import os
import torch
import argparse
import pytorch_lightning as pl
from model import NN
import dataset
from dataset import SubpopDataModule
import config
from pytorch_lightning.loggers import WandbLogger
from callbacks import MyPrintingCallback, ImagePredictionLogger, EarlyStopping
import hparams_registry

torch.set_float32_matmul_precision('medium')
os.environ["WANDB_API_KEY"] = "7a9cbed74d12db3de9cef466bb7b7cf08bdf1ea4"
os.environ["WANDB_MODE"] = "offline"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subpopulation Shift Benchmark')
    # training
    parser.add_argument('--dataset', type=str, default="Waterbirds", choices=dataset.DATASETS)
    parser.add_argument('--model', type=str, default="ERM")
    parser.add_argument('--output_folder_name', type=str, default='debug')
    parser.add_argument('--train_attr', type=str, default="yes", choices=['yes', 'no'])
    # others
    parser.add_argument('--data_dir', type=str, default="/home/aamer98/scratch/data/subpopbench")
    parser.add_argument('--output_dir', type=str, default="/home/aamer98/projects/def-ebrahimi/aamer98/repos/Med_Unc/logs")
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--wandb_logging', action='store_true')
    # uncertainty measures
    parser.add_argument('--dropout_iters', type=int, default=5)
    # early stopping
    parser.add_argument('--use_es', action='store_true')
    parser.add_argument('--es_strategy', choices=['metric'], default='metric')
    parser.add_argument('--es_metric', type=str, default='min_group:accuracy')
    parser.add_argument('--es_patience', type=int, default=5, help='Stop after this many checkpoints w/ no improvement')
    # checkpoints
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    # architectures and pre-training sources
    parser.add_argument('--image_arch', default='resnet_sup_in1k',
                        choices=['resnet_sup_in1k', 'resnet_sup_in21k', 'resnet_simclr_in1k', 'resnet_barlow_in1k',
                                 'vit_sup_in1k', 'vit_sup_in21k', 'vit_clip_oai', 'vit_clip_laion', 'vit_sup_swag',
                                 'vit_dino_in1k', 'resnet_dino_in1k'])
    parser.add_argument('--text_arch', default='bert-base-uncased',
                        choices=['bert-base-uncased', 'gpt2', 'xlm-roberta-base',
                                 'allenai/scibert_scivocab_uncased', 'distilbert-base-uncased'])
    args = parser.parse_args()


    hparams = hparams_registry.default_hparams(args.model, args.dataset)

    wandb_logger = WandbLogger(project="lit-wandb")
    model = NN(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
    )
    dm = SubpopDataModule(
        data_dir=args.data_dir,
        batch_size=hparams['batch_size'],
        num_workers=config.NUM_WORKERS,
        dataset=args.dataset,
        hparams=hparams,
        train_attr=args.train_attr
    )
    
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
