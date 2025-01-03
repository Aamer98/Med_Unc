import os
import torch
import pytorch_lightning as pl
from model import NN
from dataset import MnistDataModule
import config
from pytorch_lightning.loggers import WandbLogger
from callbacks import MyPrintingCallback, ImagePredictionLogger, EarlyStopping

torch.set_float32_matmul_precision('medium')
os.environ["WANDB_API_KEY"] = "7a9cbed74d12db3de9cef466bb7b7cf08bdf1ea4"
os.environ["WANDB_MODE"] = "offline"


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="lit-wandb")
    model = NN(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
    )
    dm = MnistDataModule(
        data_dir=config.DATADIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    wandb_logger = WandbLogger()


    dm.prepare_data()
    dm.setup()

    # grab samples to log predictions on
    samples = next(iter(dm.val_dataloader()))
    breakpoint()
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), ImagePredictionLogger(samples), EarlyStopping(monitor="val/loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
