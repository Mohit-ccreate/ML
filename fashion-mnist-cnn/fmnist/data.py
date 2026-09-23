"""Data loading for Fashion-MNIST.

Fashion-MNIST: 70,000 grayscale 28x28 images in 10 clothing categories
(60k train / 10k test). We additionally carve a validation split out of the
official training set so that model selection never touches the test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# Per-channel mean / std of the Fashion-MNIST training set (pixel range [0, 1]).
MEAN = (0.2860,)
STD = (0.3530,)


def build_transforms(augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform).

    Training-time augmentation is deliberately mild — random crops with a small
    padding and horizontal flips. Aggressive augmentation hurts on 28x28 images.
    """
    normalize = transforms.Normalize(MEAN, STD)
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_tf = eval_tf
    return train_tf, eval_tf


def get_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    download: bool = True,
    limit_train: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / validation / test DataLoaders.

    Parameters
    ----------
    data_dir      : where torchvision stores/looks for the raw IDX files.
    batch_size    : mini-batch size used for all three loaders.
    val_fraction  : fraction of the 60k official training images held out for validation.
    augment       : apply light augmentation to the training split only.
    num_workers   : DataLoader worker processes (0 = load in the main process).
    seed          : controls the train/val split so it is reproducible.
    download      : let torchvision fetch the data if it is missing.
    limit_train   : optional cap on the number of training samples (handy for smoke tests).
    """
    train_tf, eval_tf = build_transforms(augment)

    # Two views of the same underlying files: one with augmentation (train),
    # one without (val). We then split by *index* so the two never overlap.
    full_train_aug = datasets.FashionMNIST(str(data_dir), train=True, download=download, transform=train_tf)
    full_train_plain = datasets.FashionMNIST(str(data_dir), train=True, download=False, transform=eval_tf)
    test_ds = datasets.FashionMNIST(str(data_dir), train=False, download=download, transform=eval_tf)

    n_total = len(full_train_aug)
    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)

    train_indices = list(train_idx)
    if limit_train is not None:
        train_indices = train_indices[:limit_train]

    train_ds = Subset(full_train_aug, train_indices)
    val_ds = Subset(full_train_plain, list(val_idx))

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, val_loader, test_loader
