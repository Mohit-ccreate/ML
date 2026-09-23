#!/usr/bin/env python
"""Train a classifier on Fashion-MNIST.

Examples
--------
    python train.py                                   # defaults from configs/default.yaml
    python train.py --epochs 3 --out runs/quick       # quick run
    python train.py --model mlp --out runs/mlp        # fully-connected baseline
    python train.py --config configs/default.yaml --lr 0.001

Outputs (in --out):
    best.pt        checkpoint with the highest validation accuracy
    last.pt        checkpoint from the final epoch
    history.json   per-epoch train/val loss & accuracy
    curves.png     loss / accuracy plots
    config.json    the fully-resolved configuration that was used
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn

from fmnist.data import get_dataloaders
from fmnist.engine import evaluate, train_one_epoch
from fmnist.model import build_model, count_parameters
from fmnist.utils import get_device, load_config, save_checkpoint, save_json, set_seed
from fmnist.visualize import plot_history

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "configs" / "default.yaml"), help="YAML config file")
    # Frequently tweaked overrides (None = keep the value from the config).
    p.add_argument("--model", choices=["cnn", "mlp"], help="architecture")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", help="auto | cpu | cuda | mps")
    p.add_argument("--no-augment", action="store_true", help="disable training augmentation")
    p.add_argument("--data-dir")
    p.add_argument("--out", help="output directory for checkpoints and plots")
    p.add_argument("--limit-train", type=int, help="use only N training samples (smoke tests)")
    p.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    if args.model:
        cfg["model"]["name"] = args.model
        if args.model == "mlp":  # CNN kwargs don't apply to the MLP
            cfg["model"]["kwargs"] = {}
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    if args.no_augment:
        cfg["data"]["augment"] = False
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    if args.out:
        cfg["output"]["dir"] = args.out
    cfg["data"]["limit_train"] = args.limit_train
    return cfg


def build_scheduler(name: str, optimizer, epochs: int, steps_per_epoch: int, lr: float):
    """Returns (scheduler, step_per_batch: bool)."""
    name = (name or "none").lower()
    if name == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.25
        )
        return sched, True
    if name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return sched, False
    if name == "none":
        return None, False
    raise ValueError(f"Unknown scheduler '{name}'")


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.json")

    # Resolve relative data dir against the project folder so the script works from anywhere.
    data_dir = Path(cfg["data"]["dir"])
    if not data_dir.is_absolute():
        data_dir = HERE / data_dir

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=cfg["data"]["batch_size"],
        val_fraction=cfg["data"]["val_fraction"],
        augment=cfg["data"]["augment"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
        limit_train=cfg["data"].get("limit_train"),
    )

    model_name = cfg["model"]["name"]
    model_kwargs = cfg["model"].get("kwargs") or {}
    model = build_model(model_name, **model_kwargs).to(device)

    tcfg = cfg["train"]
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    scheduler, per_batch = build_scheduler(
        tcfg.get("scheduler", "onecycle"), optimizer, tcfg["epochs"], len(train_loader), tcfg["lr"]
    )

    print(f"device        : {device}")
    print(f"model         : {model_name} ({count_parameters(model):,} trainable params)")
    print(f"train/val     : {len(train_loader.dataset):,} / {len(val_loader.dataset):,} samples")
    print(f"epochs        : {tcfg['epochs']}  |  batch {cfg['data']['batch_size']}  |  lr {tcfg['lr']}")
    print(f"output        : {out_dir}\n")

    history, best_acc = [], -1.0
    show_progress = not args.no_progress
    t_start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler if per_batch else None,
            progress=show_progress, desc=f"epoch {epoch}/{tcfg['epochs']}",
        )
        if scheduler is not None and not per_batch:
            scheduler.step()
        va = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": tr.loss, "train_acc": tr.accuracy,
            "val_loss": va.loss, "val_acc": va.accuracy,
            "lr": optimizer.param_groups[0]["lr"], "seconds": dt,
        }
        history.append(record)
        save_json(history, out_dir / "history.json")

        flag = ""
        if va.accuracy > best_acc:
            best_acc = va.accuracy
            flag = "  *best*"
            save_checkpoint(out_dir / "best.pt", model, model_name=model_name, model_kwargs=model_kwargs,
                            epoch=epoch, metrics={"val_loss": va.loss, "val_acc": va.accuracy}, config=cfg)
        print(
            f"epoch {epoch:2d}/{tcfg['epochs']}  "
            f"train loss {tr.loss:.4f} acc {100 * tr.accuracy:5.2f}%  |  "
            f"val loss {va.loss:.4f} acc {100 * va.accuracy:5.2f}%  |  {dt:5.1f}s{flag}"
        )

    save_checkpoint(out_dir / "last.pt", model, model_name=model_name, model_kwargs=model_kwargs,
                    epoch=tcfg["epochs"], metrics=history[-1], config=cfg)
    plot_history(history, out_dir / "curves.png")
    total = time.time() - t_start
    print(f"\ndone in {total / 60:.1f} min  |  best val acc {100 * best_acc:.2f}%  |  artifacts in {out_dir}/")


if __name__ == "__main__":
    main()
