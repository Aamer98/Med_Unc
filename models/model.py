import os, sys
sys.path.insert(0, '/home/aamer98/projects/def-ebrahimi/aamer98/repos/Med_Unc')

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (Accuracy, BinaryPrecision, BinaryRecall, BinaryPrecision, BinaryF1Score, MulticlassCalibrationError, BinaryAUROC, BinaryFairness, BinaryGroupStatRates)

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

        metrics = MetricCollection([Accuracy(task="binary"), BinaryPrecision(),
                                    BinaryRecall(), BinaryF1Score(), BinaryAUROC()])
        
        prob_metrics = MetricCollection([MulticlassCalibrationError(num_classes=num_classes), 
                                        BrierScore(num_classes=num_classes)])
        
        group_metrics = MetricCollection([BinaryFairness(num_attributes)])

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        self.train_prob_metrics = prob_metrics.clone(prefix='train/')
        self.valid_prob_metrics = prob_metrics.clone(prefix='val/')
        self.test_prob_metrics = prob_metrics.clone(prefix='test/')

        self.train_group_metrics = group_metrics.clone(prefix='train/')
        self.valid_group_metrics = group_metrics.clone(prefix='val/')
        self.test_group_metrics = group_metrics.clone(prefix='test/')

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


    def test_metrics(algorithm, loader, train_loader, device, thres=0.5):
        
        # Get train samples
        train_targets, train_attributes, train_gs = get_samples(train_loader)

        # preds: sigmoid output
        targets, attributes, preds, gs = predict_on_set(algorithm, loader, device) # gs: group sensitive attribute: (target, attribute) pairing?
        preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
        label_set = np.unique(targets) # set of labels: but why?

        breakpoint()
        # Calculate metrics
        res = {}
        res['per_attribute'] = {}
        res['per_class'] = {} 
        res['per_group'] = {}

        ## Overall metrics
        res['overall'] = {
            **binary_metrics(targets, preds_rounded, label_set),
            **prob_metrics(targets, preds, label_set)
        }

        ## Per attribute metrics
        for a in np.unique(attributes):
            mask = attributes == a
            res['per_attribute'][str(a)] = {
                **binary_metrics(targets[mask], preds_rounded[mask], label_set),
                **prob_metrics(targets[mask], preds[mask], label_set)
            }
            train_mask = train_attributes == a
            res['per_attribute'][str(a)]['train_n_samples'] = len(train_targets[train_mask])

        ## Per class metrics
        classes_report = classification_report(targets, preds_rounded, output_dict=True, zero_division=0.)
        res['overall']['macro_avg'] = classes_report['macro avg']
        res['overall']['weighted_avg'] = classes_report['weighted avg']
        for y in np.unique(targets):
            res['per_class'][str(y)] = classes_report[str(y)]

        for c in np.unique(targets):
            mask = targets == c
            res['per_class'][f'class_{str(c)}'] = {
                **binary_metrics(targets[mask], preds_rounded[mask], label_set),
                **attribute_metrics(targets[mask], preds[mask], label_set)
            }
            train_mask = train_targets == c
            res['per_class'][f'class_{str(c)}']['train_n_samples'] = len(train_targets[train_mask])

        ## Per group metrics
        for g in np.unique(gs):
            mask = gs == g
            res['per_group'][str(g)] = {
                **binary_metrics(targets[mask], preds_rounded[mask], label_set),
                **attribute_metrics(targets[mask], preds[mask], label_set)
            }
            train_mask = train_gs == g
            res['per_group'][str(g)]['train_n_samples'] = len(train_targets[train_mask])


        res['adjusted_accuracy'] = sum([res['per_group'][str(g)]['accuracy'] for g in np.unique(gs)]) / len(np.unique(gs))
        res['min_attr']  = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
        res['max_attr']  = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
        res['min_group'] = pd.DataFrame(res['per_group']).min(axis=1).to_dict()
        res['max_group'] = pd.DataFrame(res['per_group']).max(axis=1).to_dict()

        return res



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
        loss, logits, y, atts = self._common_step(batch, batch_idx)
        probs = F.softmax(logits, dim=-1)        
        preds = probs.argmax(-1)
        output = self.train_metrics(preds, y)
        prob_output = self.train_prob_metrics(probs, y)
        group_output = self.train_group_metrics(preds, y, atts)
        breakpoint()
        self.log_dict(
            output | prob_output,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        self.log("train/loss", 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        loss, logits, y, atts = self._common_step(batch, batch_idx)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(-1)
        output = self.valid_metrics(preds, y)
        prob_output = self.valid_prob_metrics(probs, y)
        # group_output = self.train_group_metrics(preds, y, atts)
        self.log_dict(
            output | prob_output,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        self.log("val/loss", 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True) 
        self.validation_step_outputs.append(probs)     
        return probs


    def test_step(self, batch, batch_idx):
        loss, logits, y, atts = self._common_step(batch, batch_idx)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(-1)      
        output = self.test_metrics(preds, y)
        prob_output = self.valid_prob_metrics(probs, y)
        group_output = self.train_group_metrics(preds, y, atts)
        self.log_dict(
            output | prob_output,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        self.log("test/loss", 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True) 
    
    def _common_step(self, batch, batch_idx):
        i, x, y, a = batch
        loss, logits = self.loss(x, y)
        return loss, logits, y, a

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
