#!/usr/bin/env python
"""Evaluate a trained checkpoint on the held-out Fashion-MNIST test set.

    python evaluate.py --checkpoint runs/cnn/best.pt

Writes into the checkpoint's directory:
    test_metrics.json      accuracy, macro-F1, per-class precision/recall/F1
    confusion_matrix.png   row-normalised confusion matrix
    predictions.png        a grid of test images with predicted vs. true labels
    mistakes.png           the most confident wrong predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

from fmnist.data import CLASSES, get_dataloaders
from fmnist.engine import evaluate, predict
from fmnist.model import build_model
from fmnist.utils import get_device, load_checkpoint, save_json, set_seed
from fmnist.visualize import plot_confusion_matrix, plot_sample_grid

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(HERE / "runs" / "cnn" / "best.pt"))
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", help="output directory (default: alongside the checkpoint)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out) if args.out else ckpt_path.parent

    ckpt = load_checkpoint(ckpt_path, device)
    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, augment=False, download=True
    )

    res = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    y_true, y_pred, probs = predict(model, test_loader, device)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, digits=4)

    print(f"checkpoint : {ckpt_path}  (epoch {ckpt['epoch']}, {ckpt['model_name']})")
    print(f"test loss  : {res.loss:.4f}")
    print(f"test acc   : {100 * res.accuracy:.2f}%")
    print(f"macro F1   : {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    save_json(
        {
            "checkpoint": str(ckpt_path),
            "model": ckpt["model_name"],
            "epoch": ckpt["epoch"],
            "test_loss": res.loss,
            "test_accuracy": res.accuracy,
            "macro_f1": macro_f1,
            "per_class": {c: report[c] for c in CLASSES},
        },
        out_dir / "test_metrics.json",
    )
    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")

    # A grid of the first 32 test images with predictions.
    images = torch.cat([test_loader.dataset[i][0].unsqueeze(0) for i in range(32)])
    plot_sample_grid(images, y_true[:32], y_pred[:32], out_dir / "predictions.png",
                     title="Test predictions (red = wrong, true label in brackets)")

    # The 24 most confident mistakes — the most instructive failure cases.
    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong):
        conf = probs[wrong, y_pred[wrong]]
        worst = wrong[np.argsort(-conf)[:24]]
        images = torch.cat([test_loader.dataset[int(i)][0].unsqueeze(0) for i in worst])
        plot_sample_grid(images, y_true[worst], y_pred[worst], out_dir / "mistakes.png",
                         title="Most confident mistakes: predicted (true)")

    print(f"artifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
