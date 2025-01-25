import os, sys
sys.path.insert(0, '/home/aamer98/projects/def-ebrahimi/aamer98/repos/Med_Unc')

import numpy as np
import torch
import netcal.metrics
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (Accuracy, BinaryPrecision, BinaryRecall, BinaryPrecision, BinaryF1Score, MulticlassCalibrationError, BinaryAUROC, BinaryFairness, BinaryGroupStatRates)
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score, brier_score_loss, log_loss, classification_report)

import torchvision.models  
from transformers import BertModel, DistilBertModel, GPT2Model, AutoModel
import timm
from torch.hub import load_state_dict_from_url

import wandb

from  models import wide_resnet, networks, resnet50_dropout
from models.custom_metrics import BrierScore

ALGORITHMS = [
    "ERM",
    "MCDropout",
    "DeepEnsemble",
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

        # Calculated on discrete predictions
        metrics = MetricCollection([Accuracy(task="binary"), BinaryPrecision(),
                                    BinaryRecall(), BinaryF1Score(), BinaryAUROC()])

        # Calculated on probabilities
        prob_metrics = MetricCollection([MulticlassCalibrationError(num_classes=num_classes), 
                                        BrierScore(num_classes=num_classes)])

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        self.train_prob_metrics = prob_metrics.clone(prefix='train/')
        self.valid_prob_metrics = prob_metrics.clone(prefix='val/')
        self.test_prob_metrics = prob_metrics.clone(prefix='test/')

        # Track metrics for each group
        self.a_groups = [f"a={i}" for i in range(num_attributes)]
        self.c_groups = [f"y={i}" for i in range(num_classes)]
        self.sub_groups = [f"y={i},a={j}" for i in range(num_classes) for j in range(num_attributes)]

        self.group_train_metrics = {'attributes':{}, 'classes':{}, 'subgroups':{}}
        self.group_val_metrics = {'attributes':{}, 'classes':{}, 'subgroups':{}}
        self.group_test_metrics = {'attributes':{}, 'classes':{}, 'subgroups':{}}
        self.group_train_prob_metrics = {'attributes':{}, 'classes':{}, 'subgroups':{}}
        self.group_val_prob_metrics = {'attributes':{}, 'classes':{}, 'subgroups':{}}
        self.group_test_prob_metrics = {'attributes':{}, 'classes':{}, 'subgroups':{}}

        for i in self.a_groups:
            self.group_train_metrics['attributes'][i] = metrics.clone(prefix=f'train/groups/att_{i}_')
            self.group_val_metrics['attributes'][i] = metrics.clone(prefix=f'val/groups/att_{i}_')
            self.group_test_metrics['attributes'][i] = metrics.clone(prefix=f'test/groups/att_{i}_')
            self.group_train_prob_metrics['attributes'][i] = prob_metrics.clone(prefix=f'train/groups/att_{i}_')
            self.group_val_prob_metrics['attributes'][i] = prob_metrics.clone(prefix=f'val/groups/att_{i}_')
            self.group_test_prob_metrics['attributes'][i] = prob_metrics.clone(prefix=f'test/groups/att_{i}_')

        for i in self.c_groups:
            self.group_train_metrics['classes'][i] = metrics.clone(prefix=f'train/groups/class_{i}_')
            self.group_val_metrics['classes'][i] = metrics.clone(prefix=f'val/groups/class_{i}_')
            self.group_test_metrics['classes'][i] = metrics.clone(prefix=f'test/groups/class_{i}_')
            self.group_train_prob_metrics['classes'][i] = prob_metrics.clone(prefix=f'train/groups/class_{i}_')
            self.group_val_prob_metrics['classes'][i] = prob_metrics.clone(prefix=f'val/groups/class_{i}_')
            self.group_test_prob_metrics['classes'][i] = prob_metrics.clone(prefix=f'test/groups/class_{i}_')

        for i in self.sub_groups:
            self.group_train_metrics['subgroups'][i] = metrics.clone(prefix=f'train/groups/group_{i}_')
            self.group_val_metrics['subgroups'][i] = metrics.clone(prefix=f'val/groups/group_{i}_')
            self.group_test_metrics['subgroups'][i] = metrics.clone(prefix=f'test/groups/group_{i}_')
            self.group_train_prob_metrics['subgroups'][i] = prob_metrics.clone(prefix=f'train/groups/group_{i}_')
            self.group_val_prob_metrics['subgroups'][i] = prob_metrics.clone(prefix=f'val/groups/group_{i}_')
            self.group_test_prob_metrics['subgroups'][i] = prob_metrics.clone(prefix=f'test/groups/group_{i}_')

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
        dummy_size = [1]+ list(self.hparams["input_size"])
        dummy_input = torch.zeros(dummy_size, device=self.device)
        model_filename = f"weights/model_{str(self.global_step).zfill(5)}.onnx"
        os.makedirs("weights", exist_ok=True)
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
        dummy_size = [1]+ list(self.hparams["input_size"])
        dummy_input = torch.zeros(dummy_size, device=self.device)
        model_filename = "weights/model_final.onnx"
        os.makedirs("weights", exist_ok=True)
        self.to_onnx(model_filename, dummy_input, export_params=True)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        wandb.log_artifact(artifact)

    def configure_optimizers(self):
        raise NotImplementedError

    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return a list of strings of the form 'y=0,a=0'"""
        gs = []
        y = y.cpu().tolist()
        a = a.cpu().tolist()

        for i in range(len(y)):
            gs.append(f'y={y[i]},a={a[i]}')
        
        return np.array(gs)

    @staticmethod
    def return_attributes(all_a):
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)

    def group_metrics(self, targets, atts, gs, probs, preds, split):
        atts = atts.clone().cpu().numpy()
        
        a_metrics = a_prob_metrics = c_metrics = c_prob_metrics = g_metrics = g_prob_metrics = {}

        for a in np.unique(atts):
            mask = atts == a
            targets_a = targets[mask]
            probs_a = probs[mask]
            preds_a = preds[mask]
            
            if split=='train':
                a_metrics = a_metrics | self.group_train_metrics['attributes'][f'a={a}'](preds_a.cpu(), targets_a.cpu())
                a_prob_metrics = a_prob_metrics | self.group_train_prob_metrics['attributes'][f'a={a}'](probs_a.cpu(), targets_a.cpu())
            elif split=='val':
                a_metrics = a_metrics | self.group_val_metrics['attributes'][f'a={a}'](preds_a.cpu(), targets_a.cpu())
                a_prob_metrics = a_prob_metrics | self.group_val_prob_metrics['attributes'][f'a={a}'](probs_a.cpu(), targets_a.cpu()) 
            elif split=='test':
                a_metrics = a_metrics | self.group_test_metrics['attributes'][f'a={a}'](preds_a.cpu(), targets_a.cpu())
                a_prob_metrics = a_prob_metrics | self.group_test_prob_metrics['attributes'][f'a={a}'](probs_a.cpu(), targets.cpu())

        for c in torch.unique(targets):
            mask = targets == c
            mask = mask.cpu().numpy()
            targets_c = targets[mask]
            probs_c = probs[mask]
            preds_c = preds[mask]

            if split=='train':
                c_metrics = c_metrics | self.group_train_metrics['classes'][f'y={c}'](preds_c.cpu(), targets_c.cpu())
                c_prob_metrics = c_prob_metrics | self.group_train_prob_metrics['classes'][f'y={c}'](probs_c.cpu(), targets_c.cpu())
            elif split=='val':
                c_metrics = c_metrics | self.group_val_metrics['classes'][f'y={c}'](preds_c.cpu(), targets_c.cpu())
                c_prob_metrics = c_prob_metrics | self.group_val_prob_metrics['classes'][f'y={c}'](probs_c.cpu(), targets_c.cpu())
            elif split=='test':
                c_metrics = c_metrics| self.group_test_metrics['classes'][f'y={c}'](preds_c.cpu(), targets_c.cpu())
                c_prob_metrics = c_prob_metrics | self.group_test_prob_metrics['classes'][f'y={c}'](probs_c.cpu(), targets_c.cpu())

        for g in np.unique(gs):
            mask = gs == g
            targets_g = targets[mask]
            probs_g = probs[mask]
            preds_g = preds[mask]

            if split=='train':
                g_metrics = g_metrics | self.group_train_metrics['subgroups'][g](preds_g.cpu(), targets_g.cpu())
                g_prob_metrics = g_prob_metrics | self.group_train_prob_metrics['subgroups'][g](probs_g.cpu(), targets_g.cpu())
            elif split=='val':
                g_metrics = g_metrics | self.group_val_metrics['subgroups'][g](preds_g.cpu(), targets_g.cpu())
                g_prob_metrics = g_prob_metrics | self.group_val_prob_metrics['subgroups'][g](probs_g.cpu(), targets_g.cpu())
            elif split=='test':
                g_metrics = g_metrics | self.group_test_metrics['subgroups'][g](preds_g.cpu(), targets_g.cpu())
                g_prob_metrics = g_prob_metrics | self.group_test_prob_metrics['subgroups'][g](probs_g.cpu(), targets_g.cpu())
        
        return a_metrics | g_metrics | c_metrics | a_prob_metrics | g_prob_metrics | c_prob_metrics



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

        # log model and hyperparameters
        self.save_hyperparameters()

    def return_feats(self, x):
        return self.featurizer(x)

    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        loss, logits, y, atts, gs = self._common_step(batch, batch_idx)
        probs = F.softmax(logits, dim=-1)        
        preds = probs.argmax(-1)
        output = self.train_metrics(preds, y)
        prob_output = self.train_prob_metrics(probs, y)

        group_output = self.group_metrics(y, atts, gs, probs, preds, 'train')
        self.log_dict(
            output | prob_output | group_output | {'train/loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        loss, logits, y, atts, gs = self._common_step(batch, batch_idx)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(-1)
        output = self.valid_metrics(preds, y)
        prob_output = self.valid_prob_metrics(probs, y)
        group_output = self.group_metrics(y, atts, gs, probs, preds, 'val')
        self.log_dict(
            output | prob_output | group_output | {'val/loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        self.validation_step_outputs.append(probs)     
        return probs


    def test_step(self, batch, batch_idx):
        loss, logits, y, atts, gs = self._common_step(batch, batch_idx)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(-1)      
        output = self.test_metrics(preds, y)
        prob_output = self.valid_prob_metrics(probs, y)
        group_output = self.group_metrics(y, atts, gs, probs, preds, 'test')
        self.log_dict(
            output | prob_output | group_output | {'test/loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
    
    def _common_step(self, batch, batch_idx):
        i, x, y, a = batch
        gs = self.return_groups(y, a)
        loss, logits = self.loss(x, y)
        return loss, logits, y, a, gs

    # method to get loss on a batch
    def loss(self, xs, ys):
        logits = self.forward(xs) # calls self.forward
        # log_probs = F.log_softmax(logits, dim=-1)
        # loss = F.nll_loss(logits, ys)
        loss = F.cross_entropy(logits, ys)
        return loss, logits

    def configure_optimizers(self):
        return optim.SGD(
        self.network.parameters(),
        lr=self.h_params['lr'],
        weight_decay=self.h_params['weight_decay'],
        momentum=0.9)
