import os
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from dataset import EMBEDMammoDataModule
from downstream_model import MammoNet


def main(hparams):

    # torch.set_float32_matmul_precision('medium')
    torch.set_float32_matmul_precision("high")

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(hparams.seed, workers=True)

    # data
    if hparams.dataset == "embed":
        data = EMBEDMammoDataModule(
            target="cancer",
            csv_file=hparams.csv_file,
            image_size=(1024, 768),
            batch_alpha=hparams.batch_alpha,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
        )
    else:
        print("Unknown dataset. Exiting.")
        return

    model = MammoNet(
        num_classes=2, backbone=hparams.model, learning_rate=hparams.learning_rate
    )

    # Create output directory
    output_dir = os.path.join(hparams.output_root, hparams.output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("")
    print("=============================================================")
    print("TRAINING...")
    print("=============================================================")
    print("")

    wandb_logger = WandbLogger(save_dir=hparams.output_root, project="mammo-stuff")
    wandb_logger.watch(model, log="all", log_freq=100)

    # train
    trainer = pl.Trainer(
        val_check_interval=1000,
        max_epochs=hparams.epochs,
        accelerator="auto",
        devices=hparams.num_devices,
        precision="16-mixed",
        num_sanity_val_steps=0,
        logger=[
            TensorBoardLogger(hparams.output_root, name=hparams.output_name),
            wandb_logger,
        ],
        callbacks=[
            ModelCheckpoint(filename="last.ckpt"),
            ModelCheckpoint(monitor="val_auc", mode="max"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )
    trainer.fit(model=model, datamodule=data)

    print("")
    print("=============================================================")
    print("VALIDATION...")
    print("=============================================================")
    print("")

    trainer.validate(
        model=model,
        datamodule=data,
        ckpt_path=trainer.checkpoint_callback.best_model_path,
    )

    print("")
    print("=============================================================")
    print("TESTING...")
    print("=============================================================")
    print("")

    trainer.test(
        model=model,
        datamodule=data,
        ckpt_path=trainer.checkpoint_callback.best_model_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_alpha", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="embed")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/vol/biomedic3/data/EMBED/tables/mammo-net-csv/embed-non-negative.csv",
    )
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--output_name", type=str, default="cancer-baseline")
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    main(args)
