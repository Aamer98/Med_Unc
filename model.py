import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import wandb
from pytorch_lightning.loggers import WandbLogger


class NN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate):
        super(NN, self).__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # log hyperparameters
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.valid_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        preds = logits.argmax(-1)
        accuracy = self.train_acc(preds, y)
        f1_score = self.train_f1(preds, y)
        self.log_dict(
            {
                "train/loss": loss,
                "train/accuracy": accuracy,
                "train/f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        preds = logits.argmax(-1)
        accuracy = self.valid_acc(preds, y)
        f1_score = self.valid_f1(preds, y)
        self.log_dict(
            {
                "val/loss": loss,
                "val/accuracy": accuracy,
                "val/f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )   
        self.validation_step_outputs.append(logits)     
        return logits

    def on_validation_epoch_end(self):

        validation_step_outputs = self.validation_step_outputs

        dummy_input = torch.zeros(self.hparams["input_size"], device=self.device)
        model_filename = f"weights/model_{str(self.global_step).zfill(5)}.onnx"
        torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        self.logger.experiment.log_artifact(artifact)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
            "global_step": self.global_step})


    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        preds = logits.argmax(-1)
        accuracy = self.test_acc(preds, y)
        f1_score = self.test_f1(preds, y)
        self.log_dict(
            {
                "test/loss": loss,
                "test/accuracy": accuracy,
                "test/f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )        

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        loss, logits = self.loss(x, y)
        return loss, logits, y

    # method to get loss on a batch
    def loss(self, xs, ys):
        logits = self.forward(xs) # calls self.forward
        loss = F.nll_loss(logits, ys)
        return loss, logits

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y
    
    # save model in ONNX format
    def on_test_epoch_end(self):  # args are defined as part of pl API
        dummy_input = torch.zeros(self.hparams["input_size"], device=self.device)
        model_filename = "weights/model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        wandb.log_artifact(artifact)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)