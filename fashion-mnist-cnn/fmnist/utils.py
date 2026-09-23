"""Small utilities: reproducibility, device selection, config and checkpoints."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Pick a device. "auto" prefers CUDA, then Apple MPS, then CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    model_name: str,
    model_kwargs: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any] | None = None,
) -> None:
    """Persist weights *and* everything needed to rebuild the model later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dict produced by :func:`save_checkpoint`."""
    return torch.load(path, map_location=device, weights_only=False)
