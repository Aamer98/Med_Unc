import sys
sys.path.insert(0, '/home/aamer98/projects/def-ebrahimi/aamer98/repos/Med_Unc')

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import torchvision.models  # https://github.com/pytorch/hub/issues/46
from transformers import BertModel, DistilBertModel, GPT2Model, AutoModel
import timm
from torch.hub import load_state_dict_from_url

import wandb

from  models import wide_resnet, networks, resnet50_dropout


ALGORITHMS = [
    "ERM",
    "MCDropout",
    "DeepEnsemble",
    "BayesianNN",
    "TTA"]



def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(pl.LightningModule):
    def __init__(self, data_type, input_size, num_classes, num_attributes, num_examples, h_params, grp_sizes=None):
        super(Algorithm, self).__init__()
        self.h_params = h_params
        self.input_size = input_size
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_examples = num_examples

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.valid_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError     

    def _common_step(self, batch, batch_idx):
        raise NotImplementedError

    # method to get loss on a batch
    def loss(self, xs, ys):
        raise NotImplementedError

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

    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a

        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)

        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)



class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""

    def __init__(
        self,
        data_type,
        input_size,
        num_classes,
        num_attributes,
        num_examples,
        h_params,
        grp_sizes=None,
    ):
        super(ERM, self).__init__(
            data_type,
            input_size,
            num_classes,
            num_attributes,
            num_examples,
            h_params,
            grp_sizes,
        )

        self.featurizer = networks.Featurizer(data_type, input_size, self.h_params)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.h_params["nonlinear_classifier"]
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_model()

        # log hyperparameters
        self.save_hyperparameters()

    def _init_model(self):
        self.clip_grad = (
            self.data_type == "text" and self.h_params["optimizer"] == "adamw"
        )

        self.optimizer = get_optimizers["sgd"](
            self.network, self.h_params["lr"], self.h_params["weight_decay"]
        )
        self.lr_scheduler = None

    def return_feats(self, x):
        return self.featurizer(x)

    def forward(self, x):
        return self.network(x)
    
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
        i, x, y, _a = batch
        loss, logits = self.loss(x, y)
        return loss, logits, y

    # method to get loss on a batch
    def loss(self, xs, ys):
        logits = self.forward(xs) # calls self.forward
        loss = F.nll_loss(logits, ys)
        return loss, logits


