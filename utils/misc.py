import hashlib
import os
import sys
import math
from collections import OrderedDict
from numbers import Number
import operator
import numpy as np
import torch
from collections import Counter
import wandb


def wandb_init(args):
    os.environ["WANDB_API_KEY"] = "7a9cbed74d12db3de9cef466bb7b7cf08bdf1ea4"
    os.environ["WANDB_MODE"] = "offline"

    if args.algorithm == "MCDropout":
        exp_name = f"ALG_{args.algorithm}_iters_{args.mc_iters}"
    else:
        exp_name = f"ALG_{args.algorithm}"

    run = wandb.init(
        # Set the project where this run will be logged
        project=args.dataset,
        name=exp_name,
        # Track hyperparameters and run metadata
        config={
            "train_attr": args.train_attr,
            "pretrained": args.pretrained,
        },
    )
    print(f"===> Wandb initialized with run_id: {run.id}")





def prepare_folders(args):
    folders_util = [
        args.output_dir,
        os.path.join(args.output_dir, args.output_folder_name, args.store_name),
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.makedirs(folder)


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        (
            torch.cat(tuple([t.view(-1) for t in dict_1_values]))
            - torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        )
        .pow(2)
        .mean()
    )


class MovingAverage:
    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def count_samples_per_class(targets, num_labels):
    counts = Counter()
    for y in targets:
        counts[int(y)] += 1
    return [counts[i] if counts[i] else np.inf for i in range(num_labels)]


def make_balanced_weights_per_sample(targets):
    counts = Counter()
    classes = []
    for y in targets:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)
    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(targets))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb

    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


def safe_load(parsed):
    # certain metrics (e.g., AUROC) sometimes saved as a 1-element list
    if isinstance(parsed, list):
        return parsed[0]
    else:
        return parsed


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of dataset corresponding to a random split of the given dataset,
    with n data points in the first dataset and the rest in the last using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def mixup_data(x, y, alpha=1.0, device="cpu"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def accuracy(network, loader, device):
    num_labels = loader.dataset.num_labels
    num_attributes = loader.dataset.num_attributes
    corrects = torch.zeros(num_attributes * num_labels)
    totals = torch.zeros(num_attributes * num_labels)

    network.eval()
    with torch.no_grad():
        for _, x, y, a in loader:
            p = network.predict(x.to(device))
            p = (
                (p > 0).cpu().eq(y).float()
                if p.squeeze().ndim == 1
                else p.argmax(1).cpu().eq(y).float()
            )
            groups = num_attributes * y + a
            for g in groups.unique():
                corrects[g] += p[groups == g].sum()
                totals[g] += (groups == g).sum()
        corrects, totals = corrects.tolist(), totals.tolist()

        total_acc = sum(corrects) / sum(totals)
        group_acc = [c / t if t > 0 else np.inf for c, t in zip(corrects, totals)]
    network.train()

    return total_acc, group_acc


def adjust_learning_rate(optimizer, lr, step, total_steps, schedule, cos=False):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if step >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
