import os
import sys
import numpy as np
import torch
from dotenv import load_dotenv
import logging

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

import terratorch
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datasets.transforms import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels

import albumentations
from albumentations import Compose, Flip
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

load_dotenv(dotenv_path="./config.env")

''' Parameters adjustable via the config.env file '''
GEOFM = os.environ.get("GEOFM", "prithvi_eo_v2_300")
DATASET_PATH = os.environ.get("CROP_DATASET_PATH", "./datasets/multi-temporal-crop-classification")
EPOCHS = int(os.environ.get("EPOCHS", 20))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))

''' Parameters adjusted to fit the fine-tuning and cluster configuration '''
LR = 2.0e-4
WEIGHT_DECAY = 0.1
HEAD_DROPOUT=0.1
FREEZE_BACKBONE = False
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
CLASS_WEIGHTS = [
    0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462,
    1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702
]
NUM_FRAMES = 3
SEED = 0
OUT_DIR = "./outputs"  # for model checkpoints and log files

pl.seed_everything(SEED)

''' Logger '''
logger = TensorBoardLogger(
    save_dir=OUT_DIR,
    name="crop-classification",
)

''' Callbacks '''
checkpoint_callback = ModelCheckpoint(
    monitor="val/Multiclass_Jaccard_Index",
    mode="max",
    dirpath=os.path.join(OUT_DIR, "crop-classification", "checkpoints"),
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
flatten = terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
unflatten = terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=NUM_FRAMES),
gaussian_noise = albumentations.GaussNoise(var_limit=(10.0, 50.0), p=0.1)
blur = albumentations.Blur(blur_limit=7, p=0.1) # Has to be odd
clouds = albumentations.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1, p=0.1)
flip = albumentations.Flip()
tensor = albumentations.pytorch.transforms.ToTensorV2()

train_transforms = [
    flatten,
    flip,
    tensor,
    unflatten,
]

val_transforms = [
    flatten,
    flip,
    tensor,
    unflatten,
]

test_transforms = [
    flatten,
    tensor,
    unflatten,
]

''' Data module '''
data_module = MultiTemporalCropClassificationDataModule(
    batch_size=BATCH_SIZE,
    data_root=DATASET_PATH,
    train_transform=train_transforms,
    val_transform=val_transforms,
    test_transform=test_transforms,
    reduce_zero_label=True,
    expand_temporal_dimension=True,
    num_workers=NUM_WORKERS,
    use_metadata=True,
)

''' Backbone args '''
backbone_args = dict(
    backbone_pretrained=True,
    backbone=GEOFM + "_tl", # Only Prithvi v2 models support both 'time' and 'location' encoding 
    backbone_coords_encoding=["time", "location"],
    backbone_bands=BANDS,
    backbone_num_frames=NUM_FRAMES,
)

''' Decoder args '''
decoder_args = dict(
    decoder="UperNetDecoder",
    decoder_channels=256,
    decoder_scale_modules=True,
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
            effective_time_dim=NUM_FRAMES,
        )
    ]

''' Model args '''
model_args = dict(
    **backbone_args,
    **decoder_args,
    num_classes=len(CLASS_WEIGHTS),
    head_dropout=HEAD_DROPOUT,
    necks=necks,
    rescale=True,
)

''' Model definition '''
model = SemanticSegmentationTask(
    model_args=model_args,
    plot_on_val=False,
    class_weights=CLASS_WEIGHTS,
    loss="ce",
    lr=LR,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
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

''' Plot predictions '''
data_module.setup("test")
test_dataset = data_module.test_dataset

model = SemanticSegmentationTask.load_from_checkpoint(
    ckpt_path,
    model_args=model.hparams.model_args,
    model_factory=model.hparams.model_factory
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = data_module.test_dataloader()
model.to(device)
with torch.no_grad():
    batch = next(iter(test_loader))
    images = batch["image"].to(device)
    masks = batch["mask"].numpy()
    other_keys = batch.keys() - {"image", "mask", "filename"}
    rest = {k: batch[k].to(device) for k in other_keys}

    outputs = model(images, **rest)
    preds = torch.argmax(outputs.output, dim=1).cpu().numpy()

local_path = "./predictions/classification"
os.makedirs(local_path, exist_ok=True)

for i in range(BATCH_SIZE):
    sample = {key: batch[key][i] for key in batch}
    sample["prediction"] = preds[i]
    test_dataset.plot(sample, save_path=local_path+"/sample_"+str(i)+".png")
