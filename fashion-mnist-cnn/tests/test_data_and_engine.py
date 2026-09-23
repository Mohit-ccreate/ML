"""Tests that exercise the data pipeline and the train/eval loops.

They require the Fashion-MNIST files under ``<project>/data`` and are skipped
automatically when they are absent (e.g. in an offline CI job).
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fmnist.data import CLASSES, get_dataloaders
from fmnist.engine import evaluate, predict, train_one_epoch
from fmnist.model import build_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HAS_DATA = (DATA_DIR / "FashionMNIST" / "raw" / "train-images-idx3-ubyte").exists()

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="Fashion-MNIST data not downloaded")


@pytest.fixture(scope="module")
def loaders():
    return get_dataloaders(DATA_DIR, batch_size=64, download=False, limit_train=512, num_workers=0)


def test_split_sizes_and_shapes(loaders):
    train_loader, val_loader, test_loader = loaders
    assert len(train_loader.dataset) == 512  # limited for the test
    assert len(val_loader.dataset) == 6_000  # 10% of 60k
    assert len(test_loader.dataset) == 10_000
    x, y = next(iter(train_loader))
    assert x.shape == (64, 1, 28, 28)
    assert y.dtype == torch.int64
    assert y.min() >= 0 and y.max() < len(CLASSES)


def test_train_and_val_indices_do_not_overlap():
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, download=False, num_workers=0)
    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)
    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == 60_000


def test_split_is_reproducible():
    a = get_dataloaders(DATA_DIR, download=False, seed=7, num_workers=0)[1].dataset.indices
    b = get_dataloaders(DATA_DIR, download=False, seed=7, num_workers=0)[1].dataset.indices
    assert list(a) == list(b)


def test_short_training_on_small_subset_improves_accuracy():
    """~64 optimiser steps on 2k images: a sane pipeline gets well past chance (~10%)."""
    torch.manual_seed(0)
    train_loader, val_loader, _ = get_dataloaders(
        DATA_DIR, batch_size=64, download=False, limit_train=2048, num_workers=0
    )
    model = build_model("cnn", width=8)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    before = evaluate(model, val_loader, criterion, torch.device("cpu"))
    for _ in range(2):
        train_one_epoch(model, train_loader, criterion, opt, torch.device("cpu"), progress=False)
    after = evaluate(model, val_loader, criterion, torch.device("cpu"))

    assert before.accuracy < 0.25  # untrained network ≈ chance
    assert after.accuracy > 0.45  # measured ≈ 0.66 on CPU; bound is loose for portability
    assert after.loss < before.loss


def test_predict_returns_aligned_arrays(loaders):
    _, val_loader, _ = loaders
    model = build_model("mlp")
    y_true, y_pred, probs = predict(model, val_loader, torch.device("cpu"))
    assert y_true.shape == y_pred.shape == (6_000,)
    assert probs.shape == (6_000, 10)
    assert abs(probs.sum(axis=1) - 1).max() < 1e-4
