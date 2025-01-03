import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, ImageFile
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import hparams_


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "MIMICNoFinding",
    "MIMICNotes",
    "CXRMultisite",
    "CheXpertNoFinding"
]


class SubpopDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, dataset, hparams):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.hparams = hparams

    def prepare_data(self):
        # data downloaded here once
        # single gpu
        

    def setup(self, stage=None):
        # data is loader here
        # multi gpu
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])

        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )



class SubpopDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, train_attr='yes', subsample_type=None, duplicates=None):
        df = pd.read_csv(metadata)
        df = df[df["split"] == (self.SPLITS[split])]

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        self.a = df["a"].tolist() if train_attr == 'yes' else [0] * len(df["a"].tolist())
        self.transform_ = transform
        self._count_groups()

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(self.a))
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in self.idx:
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]])
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes)) if subsample_type == "group" else min(list(self.class_sizes))

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.num_attributes * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        return i, x, y, a

    def __len__(self):
        return len(self.idx)




class BaseImageDataset(SubpopDataset):

    def __init__(self, metadata, split, train_attr='yes', subsample_type=None, duplicates=None):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        super().__init__('/', split, metadata, transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        if self.__class__.__name__ in ['MIMICNoFinding', 'CXRMultisite'] and 'MIMIC-CXR-JPG' in x:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-5] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')

            if reduced_img_path.is_file():
                x = str(reduced_img_path.resolve())

        return self.transform_(Image.open(x).convert("RGB"))



class MIMICNoFinding(BaseImageDataset):
    N_STEPS = 20001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "MIMIC-CXR-JPG", 'physionet.org/files/mimic-cxr-jpg/2.0.0/subpop_bench_meta', "metadata_no_finding.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class CheXpertNoFinding(BaseImageDataset):
    N_STEPS = 20001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 4
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "chexpert", 'subpop_bench_meta', "metadata_no_finding.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class CXRMultisite(BaseImageDataset):
    N_STEPS = 20001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    SPLITS = {               
        'tr': 0,
        'va': 1,
        'te': 2,
        'deploy': 3
    }
    EVAL_SPLITS = ['te', 'deploy']

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        metadata = os.path.join(data_path, "MIMIC-CXR-JPG", 'physionet.org/files/mimic-cxr-jpg/2.0.0/subpop_bench_meta', "metadata_multisite.csv")
        super().__init__(metadata, split, train_attr, subsample_type, duplicates)


class MIMICNotes(SubpopDataset):
    N_STEPS = 10001
    CHECKPOINT_FREQ = 200
    INPUT_SHAPE = (10000,)

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        assert hparams['text_arch'] == 'bert-base-uncased'
        metadata = os.path.join(data_path, "mimic_notes", 'subpop_bench_meta', "metadata.csv")
        self.x_array = np.load(os.path.join(data_path, "mimic_notes", 'features.npy'))
        self.data_type = "tabular"
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, x):
        return self.x_array[int(x), :].astype('float32')