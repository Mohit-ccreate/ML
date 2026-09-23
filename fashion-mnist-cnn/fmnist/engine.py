"""Training and evaluation loops (framework-agnostic: any nn.Module + DataLoader)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochResult:
    loss: float
    accuracy: float

    def as_dict(self) -> Dict[str, float]:
        return {"loss": self.loss, "accuracy": self.accuracy}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    progress: bool = True,
    desc: str = "train",
) -> EpochResult:
    """One full pass over `loader` in training mode.

    If `scheduler` is given it is stepped *per batch* (suited to OneCycleLR /
    cosine-with-warmup style schedules). Epoch-level schedulers should be
    stepped by the caller instead.
    """
    model.train()
    running_loss, correct, seen = 0.0, 0, 0
    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)
    for images, targets in iterator:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch = targets.size(0)
        running_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch
        iterator.set_postfix(loss=f"{running_loss / seen:.4f}", acc=f"{correct / seen:.4f}")

    return EpochResult(loss=running_loss / seen, accuracy=correct / seen)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    progress: bool = False,
    desc: str = "eval",
) -> EpochResult:
    """Average loss and accuracy over `loader` in eval mode."""
    model.eval()
    running_loss, correct, seen = 0.0, 0, 0
    for images, targets in tqdm(loader, desc=desc, leave=False, disable=not progress):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch = targets.size(0)
        running_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch
    return EpochResult(loss=running_loss / seen, accuracy=correct / seen)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, probabilities) for every sample in `loader`."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1).cpu()
        all_prob.append(probs.numpy())
        all_pred.append(probs.argmax(dim=1).numpy())
        all_true.append(targets.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_prob)
