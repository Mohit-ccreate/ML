"""Subprocess worker: train ViT and emit probabilities.

TensorFlow (LSTM) and PyTorch (ViT) destabilise this 3.8 GB sandbox when they
share a process, so the pipeline spawns this script as a separate process.

Usage:
  python scripts/vit_worker.py \
      --manifest M.parquet --labels L.parquet \
      --train-keys TR.parquet --val-keys VA.parquet \
      --eval-keys EV.parquet \
      --model-out model.pt --probs-out probs.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optionedge.config import load_config  # noqa: E402


def keys_to_set(df: pd.DataFrame) -> set:
    df = df.copy()
    df["date"] = df["date"].astype(str).str[:10]
    return set(zip(df["symbol"], df["date"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train-keys", required=True)
    ap.add_argument("--val-keys", required=False)
    ap.add_argument("--eval-keys", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--probs-out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.epochs:
        cfg["vision"]["epochs"] = int(args.epochs)

    from optionedge.vit import train_vit, predict_proba

    manifest = pd.read_csv(args.manifest)
    manifest["date"] = manifest["date"].astype(str).str[:10]
    labels = pd.read_parquet(args.labels)
    labels["date"] = labels["date"].astype(str).str[:10]

    train_keys = keys_to_set(pd.read_parquet(args.train_keys))
    val_keys = keys_to_set(pd.read_parquet(args.val_keys)) if args.val_keys else set()
    eval_df = pd.read_parquet(args.eval_keys)
    eval_df["date"] = eval_df["date"].astype(str).str[:10]
    eval_keys = list(zip(eval_df["symbol"], eval_df["date"]))

    model, info = train_vit(cfg, manifest, labels, train_keys, val_keys, args.model_out)

    man = manifest.copy()
    ev = pd.DataFrame(sorted(set(eval_keys)), columns=["symbol", "date"])
    man_eval = man.merge(ev, on=["symbol", "date"], how="inner")
    probs = predict_proba(model, man_eval, cfg["vision"]["image_size"])
    probs.to_parquet(args.probs_out, index=False)

    print(json.dumps({"status": "ok", "n_train": len(train_keys),
                      "n_val": len(val_keys), "n_eval": len(probs),
                      "best_val_f1": info.get("best_val_f1")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
