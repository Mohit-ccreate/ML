"""Configuration loading helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Resolve commonly used dirs relative to repo root
    root = REPO_ROOT
    for key in ("processed_dir", "charts_dir", "raw_root"):
        if key in cfg.get("data", {}):
            cfg["data"][key] = str((root / cfg["data"][key]).resolve())
    for key in ("artifacts", "metrics", "plots", "models", "db"):
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = str((root / cfg["paths"][key]).resolve())
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    Path(cfg["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data"]["charts_dir"]).mkdir(parents=True, exist_ok=True)
    for key in ("artifacts", "metrics", "plots", "models"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
