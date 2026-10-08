"""MNIST loaders: 55k train (augmented) / 5k validation / 10k test."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .constants import MNIST_MEAN, MNIST_STD


def get_transforms(train: bool) -> transforms.Compose:
    steps = []
    if train:
        # Small random rotations/shifts/zooms: hand-drawn canvas digits are never
        # as neatly centred as MNIST, so this makes the live demo far more robust.
        steps.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    steps += [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    return transforms.Compose(steps)


def get_loaders(data_dir: str | Path = "data", batch_size: int = 128, val_size: int = 5000,
                num_workers: int = 0, seed: int = 42, pin_memory: bool = False):
    """Return (train_loader, val_loader, test_loader). Downloads MNIST (~12 MB) on first use."""
    data_dir = str(data_dir)
    train_aug = datasets.MNIST(data_dir, train=True, download=True, transform=get_transforms(True))
    train_plain = datasets.MNIST(data_dir, train=True, download=True, transform=get_transforms(False))
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=get_transforms(False))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(Subset(train_aug, train_idx), shuffle=True, **common)
    val_loader = DataLoader(Subset(train_plain, val_idx), shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader
