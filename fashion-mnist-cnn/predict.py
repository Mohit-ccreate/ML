#!/usr/bin/env python
"""Classify your own image(s) with a trained checkpoint.

    python predict.py path/to/shoe.jpg another.png --checkpoint runs/cnn/best.pt

Images are converted to grayscale, resized to 28x28 and normalised the same
way as the training data. Fashion-MNIST items are light objects on a dark
background; pass --invert if your photo is a dark object on a light background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from fmnist.data import CLASSES, MEAN, STD
from fmnist.model import build_model
from fmnist.utils import get_device, load_checkpoint

HERE = Path(__file__).resolve().parent


def load_image(path: str | Path, invert: bool) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if invert:
        img = ImageOps.invert(img)
    tf = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    return tf(img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="image file(s) to classify")
    p.add_argument("--checkpoint", default=str(HERE / "runs" / "cnn" / "best.pt"))
    p.add_argument("--device", default="auto")
    p.add_argument("--invert", action="store_true", help="invert colours before classifying")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    batch = torch.stack([load_image(pth, args.invert) for pth in args.images]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).cpu()

    for path, prob in zip(args.images, probs):
        top = torch.topk(prob, k=min(args.topk, len(CLASSES)))
        summary = ", ".join(f"{CLASSES[i]} {100 * v:.1f}%" for v, i in zip(top.values, top.indices))
        print(f"{path}: {summary}")


if __name__ == "__main__":
    main()
