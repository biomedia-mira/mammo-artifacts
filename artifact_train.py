import torch
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything
from artifact_detector_model import Multilabel_ArtifactDetector, MARKER_NAMES
from dataset import EMBEDMammoDataModule, ANNOTATION_FILE
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")
image_size = (512, 384)
lr = 5e-5

data = pd.read_csv(ANNOTATION_FILE)

num_classes = len(MARKER_NAMES)
data["multilabel_markers"] = data.apply(
    lambda row: np.array([row[name] for name in MARKER_NAMES]), axis=1
)
data = EMBEDMammoDataModule(
    csv_file=data, image_size=image_size, target="artifact", batch_size=32
)

model = Multilabel_ArtifactDetector(num_classes, lr)
wandb_logger = WandbLogger(save_dir="output", project="mammo-stuff")
wandb_logger.watch(model, log="all", log_freq=100)

trainer = Trainer(
    max_epochs=35,
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    num_sanity_val_steps=0,
    logger=[TensorBoardLogger("output", name="artifact-detector"), wandb_logger],
    callbacks=[
        ModelCheckpoint(monitor="val_auc", mode="max"),
        ModelCheckpoint(filename="last"),
        TQDMProgressBar(refresh_rate=10),
    ],
)

trainer.fit(model=model, datamodule=data)
trainer.validate(
    model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path
)
trainer.test(
    model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path
)
