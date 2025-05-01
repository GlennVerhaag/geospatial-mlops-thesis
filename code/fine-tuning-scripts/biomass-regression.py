import os
import sys
import numpy as np
import torch
from dotenv import load_dotenv
import logging

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

import terratorch
from terratorch.datamodules import GenericNonGeoPixelwiseRegressionDataModule
from terratorch.tasks import PixelwiseRegressionTask

import albumentations
from albumentations import Compose, Flip
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

load_dotenv(dotenv_path="./config.env")

''' Parameters adjustable via the config.env file '''
GEOFM = os.environ.get("GEOFM", "prithvi_eo_v2_300")
DATASET_PATH = os.environ.get("BIOMASS_DATASET_PATH", "./datasets/granite-geospatial-biomass-dataset")
EPOCHS = int(os.environ.get("EPOCHS", 20))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))

''' Parameters adjusted to fit the fine-tuning and cluster configuration '''
LR = 1e-3
WEIGHT_DECAY = 0.1
FREEZE_BACKBONE = True
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
NUM_FRAMES = 1
SEED = 42
OUT_DIR = "./outputs"  # for model checkpoints and log files

pl.seed_everything(SEED)

''' Logger '''
logger = TensorBoardLogger(
    save_dir=OUT_DIR,
    name="biomass-regression",
)

''' Callbacks '''
checkpoint_callback = ModelCheckpoint(
    monitor="val/RMSE",
    mode="min",
    dirpath=os.path.join(OUT_DIR, "biomass-regression", "checkpoints"),
    filename="best-{epoch:02d}",
    save_top_k=1,
)

''' Trainer '''
trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    precision="bf16-mixed",
    num_nodes=1,
    logger=logger,
    max_epochs=EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    enable_model_summary=False,
    enable_progress_bar=False,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
    limit_predict_batches=1,  # predict only in the first batch for generating plots
)

''' Augmentations '''
gaussian_noise = albumentations.GaussNoise(var_limit=(10.0, 50.0), p=0.1)
blur = albumentations.Blur(blur_limit=7, p=0.1) # Has to be odd
clouds = albumentations.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1, p=0.1)
tensor = albumentations.pytorch.transforms.ToTensorV2()
d4 = albumentations.D4()

train_transforms = [
    d4,
    tensor,
]

val_transforms = [
    d4,
    tensor,
]

test_transforms = [
    tensor,
]
''' Data module '''
data_module = GenericNonGeoPixelwiseRegressionDataModule(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    check_stackability = False,
    # Define dataset paths 
    train_data_root=os.path.join(DATASET_PATH, 'train_images/'),
    train_label_data_root=os.path.join(DATASET_PATH, 'train_labels/'),
    val_data_root=os.path.join(DATASET_PATH, 'val_images/'), 
    val_label_data_root=os.path.join(DATASET_PATH, 'val_labels/'),
    test_data_root=os.path.join(DATASET_PATH, 'test_images/'),
    test_label_data_root=os.path.join(DATASET_PATH, 'test_labels/'),
    predict_data_root=os.path.join(DATASET_PATH, 'test_images/'),
    
    img_grep='*.tif',
    label_grep='*.tif',
    
    train_transform=train_transforms,
    val_transform=val_transforms,  
    test_transform=test_transforms,
    
    means=[
      547.36707,
      898.5121,
      1020.9082,
      2665.5352,
      2340.584,
      1610.1407,
    ],
    stds=[
      411.4701,
      558.54065,
      815.94025,
      812.4403,
      1113.7145,
      1067.641,
    ],
    
    dataset_bands = [-1, "BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2", -1, -1, -1, -1],
    output_bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    rgb_indices = [2, 1, 0],
    no_data_replace=0,
    no_label_replace=-1,
)

''' Backbone args '''
backbone_args = dict(
    backbone_pretrained=True,
    backbone=GEOFM,
    backbone_bands=BANDS,
    backbone_num_frames=NUM_FRAMES,
    backbone_img_size=512
)

''' Decoder args '''
decoder_args = dict(
    decoder="UNetDecoder",
    decoder_channels=[512, 256, 128, 64]
)

''' GeoFM-specific indices '''
indices_mapping = {
    "prithvi_eo_v1_100": [2, 5, 8, 11],
    "prithvi_eo_v2_300": [5, 11, 17, 23],
    "prithvi_eo_v2_600": [7, 15, 23, 31],
}

selected_indices = indices_mapping.get(GEOFM)
if selected_indices is None:
    raise ValueError(f"Unsupported GEOFM value: {GEOFM}")

''' Neck args '''
necks = [
    dict(
            name="SelectIndices",
            indices=selected_indices
        ),
    dict(
            name="ReshapeTokensToImage",
        ),
    dict(
            name="LearnedInterpolateToPyramidal",
        )
    ]

''' Model args '''
model_args = dict(
    **backbone_args,
    **decoder_args,
    necks=necks
)

''' Model definition '''
model = PixelwiseRegressionTask(
    model_args=model_args,
    plot_on_val=True,
    loss="rmse",
    lr=LR,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
    scheduler="ReduceLROnPlateau",
    ignore_index=-1,
    freeze_backbone=FREEZE_BACKBONE,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

''' Fine-tuning '''
trainer.fit(model, datamodule=data_module)
ckpt_path = checkpoint_callback.best_model_path

''' Testing '''
test_results = trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
test_results

''' Plot Predictions '''
data_module.setup("test")
test_dataset = data_module.test_dataset

model = PixelwiseRegressionTask.load_from_checkpoint(
    ckpt_path,
    model_factory=model.hparams.model_factory,
    model_args=model.hparams.model_args,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = data_module.test_dataloader()
model.to(device)

with torch.no_grad():
    batch = next(iter(test_loader))
    images = data_module.aug(batch)
    images = batch["image"].to(device)
    masks = batch["mask"].numpy()
    preds = model(images).output

local_path = "./predictions/regression"
os.makedirs(local_path, exist_ok=True)

for i in range(BATCH_SIZE):
    sample = {key: batch[key][i] for key in batch}
    sample["prediction"] = preds[i].cpu()
    sample["image"] = sample["image"].cpu()
    sample["mask"] = sample["mask"].cpu()
    test_dataset.plot(sample, show_axes=False, save_path=local_path+"/sample_"+str(i)+".png")
    
