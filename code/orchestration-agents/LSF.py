import boto3
import subprocess
import time
import os
import tempfile
import re

### CONFIGURATION ###

# S3 settings
AWS_ENDPOINT_URL = <YOUR AWS ENDPOINT URL>
S3_BUCKET = <YOUR S3 BUCKET NAME>  # Stores config files and all versioned artifacts
CONFIG_S3_KEY = "configs/config.env"
CONFIG_PROCESSED_S3_KEY = "configs/processed/config.env"

# DVC settings
DVC_REMOTE_URL = <YOUR DVC REMOTE URL>

# LSF settings (Adapt to your cluster configuration)
LSF_CORES = "8+1"      # cores per job
LSF_QUEUE = "x86_6h"   # queue name
LSF_MEM = "32g"        # memory per job

# Local directories
JOBS_DIR = "./downstream-tasks"  # config.env and job scripts are stored here
PREDICTIONS_DIR = "./predictions"  # Stores plotted predictions

# Polling interval (seconds)
POLL_INTERVAL = 5  # Poll for new config files

# Git settings
GIT_API_KEY = os.environ.get("GL_API_KEY") # MUST BE SET AS AN ENV VAR ON THE CLUSTER
GIT_URL = <URL OF YOUR GIT REPO>
GIT_BRANCH = <YOUR BRANCH NAME>

# Ensure the JOBS_DIR exists
if not os.path.exists(JOBS_DIR):
    os.makedirs(JOBS_DIR)

### AWS S3 Client ###
s3_client = boto3.client("s3") # AWS CREDENTIALS MUST BE SET AS ENV VARS OR VIA THE AWS CLI

### Helper Functions ###
def download_config_file():
    """
    Check for the existence of config.env in S3 (at configs/config.env).
    If found, download it (overwriting any existing version).
    """
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=CONFIG_S3_KEY)
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("No updated config file found in S3.")
            return False
        else:
            raise
    print("|------- Updated config file detected, pulling latest config and jobfiles -------|")
    subprocess.run(["git", "fetch"], check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{GIT_BRANCH}"], check=True)
    return True

def list_local_job_files():
    """
    List all .py files in the JOBS_DIR.
    """
    files = os.listdir(JOBS_DIR)
    return [os.path.join(JOBS_DIR, f) for f in files if f.endswith(".py")]

def job_completed(job_id):
    """
    Check if the given LSF job_id is no longer in the queue.
    Returns True if the job is complete.
    """
    try:
        result = subprocess.run(["bjobs", job_id], capture_output=True, text=True)
        return (job_id not in result.stdout)
    except Exception as e:
        print(f"Error checking job status: {e}")
        return False

def version_artifacts(job_files):
    """
    Version the following artifacts with DVC:
      - For each job: the <jobname>.err and <jobname>.out files.
      - All plotted predictions.
    """
    subprocess.run(["dvc", "remote", "add", "--default", "--force", "myremote", DVC_REMOTE_URL], check=True)
    subprocess.run(["dvc", "remote", "modify", "myremote", "endpointurl", AWS_ENDPOINT_URL], check=True)
    subprocess.run(["dvc", "config", "--global", "core.autostage", "true"], check=True)
    artifacts = []

    if os.path.exists("./err"):
        artifacts.append("./err")
    else:
        print("Warning: Error directory (./err) not found.")

    if os.path.exists("./out"):
        artifacts.append("./out")
    else:
        print("Warning: Output directory (./out) not found.")

    if os.path.exists(PREDICTIONS_DIR):
        artifacts.append(PREDICTIONS_DIR)
    else:
        print(f"Warning: {PREDICTIONS_DIR} not found.")

    for artifact in artifacts:
        try:
            subprocess.run(["dvc", "add", artifact], check=True)
            print(f"DVC added: {artifact}")
        except subprocess.CalledProcessError as e:
            print(f"Error adding {artifact} to DVC: {e}")

    try:
        subprocess.run(["dvc", "push"], check=True)
        print("DVC push completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error pushing DVC artifacts: {e}")

def move_config_in_s3():
    """
    Move config.env file in S3 to mark it as processed.
    """
    try:
        s3_client.copy_object(
            Bucket=S3_BUCKET,
            CopySource={'Bucket': S3_BUCKET, 'Key': CONFIG_S3_KEY},
            Key=CONFIG_PROCESSED_S3_KEY
        )
        s3_client.delete_object(Bucket=S3_BUCKET, Key=CONFIG_S3_KEY)
        print(f"Moved config file to {CONFIG_PROCESSED_S3_KEY} in S3.")
    except Exception as e:
        print(f"Error moving config file in S3: {e}")

def commit_and_push_git():
    """
    Commit and push all DVC references to the specified code repository and branch.
    """
    try:
        subprocess.run(["git", "config", "--global", "user.name", "'GitLab CI'"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "'ci@example.com'"], check=True)
        subprocess.run(["git", "fetch", "--all"], check=True)
        subprocess.run(["git", "checkout", GIT_BRANCH], check=True)
        subprocess.run(["git", "commit", "-m", "Updated DVC references. [ci skip]"], check=True)
        git_push_cmd = f"git push https://GL_API_KEY:{GIT_API_KEY}@{GIT_URL}.git {GIT_BRANCH}"
        subprocess.run(git_push_cmd, shell=True, check=True)
        print("Git commit and push completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error in git commit/push: {e}")

def submit_all_jobs(job_files):
    """
    For every job file (ending with .py) in JOBS_DIR:
      - Submit the job via jbsub.
    Returns a dictionary mapping job basenames to their LSF job IDs.
    """
    os.makedirs("./out", exist_ok=True)
    os.makedirs("./err", exist_ok=True)

    job_ids = {}
    for job_file in job_files:
        job_basename = os.path.splitext(os.path.basename(job_file))[0]
        cmd = [
            "jbsub",
            "-cores", LSF_CORES,
            "-queue", LSF_QUEUE,
            "-mem", LSF_MEM,
            "-out", f"./out/{job_basename}.out",
            "-err", f"./err/{job_basename}.err",
            "python", job_file
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            job_id = re.search(r"Job <(\d+)>", output).group(1)
            print(f"Job submitted with ID: {job_id}")
            job_ids[job_basename] = job_id
        except Exception as e:
            print(f"Error submitting job for {job_file}: {e}")
    return job_ids

def wait_for_jobs(job_ids):
    """
    Poll the LSF queue until all submitted jobs have finished.
    """
    pending = set(job_ids.values())
    while pending:
        finished = set()
        for job_id in pending:
            if job_completed(job_id):
                print(f"Job {job_id} has completed.")
                finished.add(job_id)
        pending -= finished
        if pending:
            time.sleep(POLL_INTERVAL)

def main():
    """
    Main orchestration loop:
      - Wait for the config file to appear in S3.
      - Download it and then run all local job scripts as LSF jobs.
      - After all jobs finish, version artifacts with DVC, move the config in S3,
        and commit/push the changes to git.
      - Then wait for the next config file cycle.
    """
    print("Waiting for updated config file...")
    while True:
        if download_config_file():
            job_files = list_local_job_files()
            if not job_files:
                print("No job scripts found in the local jobs directory. Waiting...")
                time.sleep(POLL_INTERVAL)
                continue

            print("Submitting jobs...")
            job_ids = submit_all_jobs(job_files)
            print("Waiting for all jobs to complete...")
            wait_for_jobs(job_ids)

            print("All jobs completed. Versioning artifacts with DVC...")
            version_artifacts(job_files)

            print("Moving processed config file in S3...")
            move_config_in_s3()

            print("Committing DVC references to git...")
            commit_and_push_git()

            print("|------- Run complete. Waiting for next config file. -------|")
        else:
            print(f"Trying again in {POLL_INTERVAL} seconds...")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
