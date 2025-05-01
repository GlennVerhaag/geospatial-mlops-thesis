import boto3
import subprocess
import time
import os
import tempfile
import re

### CONFIGURATION ###

# S3 settings
AWS_ENDPOINT_URL = <YOUR AWS ENDPOINT URL>
S3_BUCKET = <YOUR S3 BUCKET NAME> # Stores config files and all versioned artifacts
CONFIG_S3_KEY = "configs/config.env"
CONFIG_PROCESSED_S3_KEY = "configs/processed/config.env"


# DVC settings
DVC_REMOTE_URL = <YOUR DVC REMOTE URL>

# SLURM settings
SLURM_ACCOUNT = <YOUR SLURM ACCOUNT/PROJECT>
SLURM_PARTITION = <YOUR CLUSTER PARTITION>
SLURM_TIME = "02:00:00"  # Maximum run time for the job
SLURM_TASKS_PER_NODE= 1
SLURM_CPUS_PER_TASK = 48 # Configure depending on your nodes' CPU cores
SLURM_GPU = 1

# Local directories
JOBS_DIR = "./downstream-tasks" # config.env and job scripts are stored here
PREDICTIONS_DIR = "./predictions" # Stores plotted predictions

# Polling interval (seconds)
POLL_INTERVAL = 5 # Poll for new config files

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
    local_config_path = "./config.env"
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=CONFIG_S3_KEY)
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("No updated config file found in S3.")
            return False
        else:
            # some other failure
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
    job_files = [os.path.join(JOBS_DIR, f) for f in files if f.endswith(".py")]
    return job_files

def create_slurm_job_script(job_script_path, python_job_file):
    """
    Create a SLURM job script to submit the given Python job file to a compute node.
    The output (.out) and error (.err) file names are derived from the job file's name.
    
    SLURM SCRIPT MIGHT NEED TO BE ADJUSTED BASED ON YOUR SYSTEM CONFIGURATION
    """
    os.makedirs("./out", exist_ok=True)
    os.makedirs("./err", exist_ok=True)
    job_basename = os.path.splitext(os.path.basename(python_job_file))[0]
    slurm_script = f"""#!/bin/bash -x
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --output=./out/{job_basename}.out
#SBATCH --error=./err/{job_basename}.err
#SBATCH --time={SLURM_TIME}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={SLURM_TASKS_PER_NODE}
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --gres=gpu:{SLURM_GPU}

module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

source .venv/bin/activate

srun python {python_job_file}

JOB_EXIT_CODE=$?

if [ $JOB_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Job Statistics for Job ID: $SLURM_JOB_ID"
    echo "=========================================================="
    sacct -j $SLURM_JOB_ID --format=Elapsed,TotalCPU,MaxRSS,ConsumedEnergy,AllocCPUs
    echo "=========================================================="
fi

exit $JOB_EXIT_CODE

deactivate

"""
    with open(job_script_path, "w") as f:
        f.write(slurm_script)
    os.chmod(job_script_path, 0o755)

def submit_job(job_script_path):
    """
    Submit a SLURM job (using the job script) via sbatch.
    Return the job ID upon successful submission.
    """
    try:
        result = subprocess.run(["sbatch", job_script_path], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        # Expected output: "Submitted batch job <jobid>"
        job_id = re.search(r"Submitted batch job (\d+)", output).group(1)
        print(f"Job submitted with ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error submitting job: {e}")
        return None

def job_completed(job_id):
    """
    Check if the given job_id is no longer in the SLURM queue.
    Returns True if the job is complete.
    """
    try:
        result = subprocess.run(["squeue", "--job", job_id], capture_output=True, text=True)
        # If job ID is not in the output, it's completed.
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
        s3_client.copy_object(Bucket=S3_BUCKET,
                              CopySource={'Bucket': S3_BUCKET, 'Key': CONFIG_S3_KEY},
                              Key=CONFIG_PROCESSED_S3_KEY)
        s3_client.delete_object(Bucket=S3_BUCKET, Key=CONFIG_S3_KEY)
        print(f"Moved config file to {CONFIG_PROCESSED_S3_KEY} in S3.")
    except Exception as e:
        print(f"Error moving config file in S3: {e}")

def commit_and_push_git():
    """
    Commit and push all DVC references to the specifed code repository and branch.
    """
    try:
        subprocess.run(["git", "config", "--global", "user.name", "'GitLab CI'"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "'ci@example.com'"], check=True)
        subprocess.run(["git", "fetch", "--all",], check=True)
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
      - Create a temporary SLURM script.
      - Submit the job via sbatch.
    Returns a dictionary mapping job basenames to their SLURM job IDs.
    """
    job_ids = {}
    for job_file in job_files:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
            slurm_script_path = tmp.name
        create_slurm_job_script(slurm_script_path, job_file)
        job_id = submit_job(slurm_script_path)
        if job_id:
            job_basename = os.path.splitext(os.path.basename(job_file))[0]
            job_ids[job_basename] = job_id
        else:
            print(f"Job submission failed for {job_file}")
    return job_ids

def wait_for_jobs(job_ids):
    """
    Poll the SLURM queue until all submitted jobs have finished.
    """
    pending = set(job_ids.values())
    while pending:
        finished = set()
        for job_id in pending:
            if job_completed(job_id):
                print(f"Job {job_id} has completed.")
                finished.add(job_id)
        pending = pending - finished
        if pending:
            time.sleep(POLL_INTERVAL)

def main():
    """
    Main orchestration loop:
      - Wait for the config file to appear in S3.
      - Download it and then run all local job scripts as SLURM jobs.
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
 