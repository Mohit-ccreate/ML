"""Random-split benchmark — reproduces the PROTOCOL used by most cited papers.

IMPORTANT: a random split over time-series rows leaks future regimes into
training and systematically **inflates** accuracy. This script exists so the
presentation can show, side by side:

  • random-split accuracy  (comparable to Sherasiya 2025 / Kayit 2025)
  • walk-forward accuracy  (the honest, tradeable number — see run_pipeline)

Usage:
  python scripts/random_split_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optionedge.config import load_config, seed_everything  # noqa: E402
from optionedge.dataset import build_feature_table, feature_columns  # noqa: E402
from optionedge.tabular_models import TabularEnsemble  # noqa: E402
from optionedge.evaluate import classification_metrics, save_metrics  # noqa: E402


def main():
    cfg = load_config()
    seed_everything(cfg["random_seed"])
    proc = Path(cfg["data"]["processed_dir"])
    fp = proc / "features.parquet"
    if not fp.exists():
        raise SystemExit("features.parquet missing — run the pipeline first")
    df = pd.read_parquet(fp)
    cols = feature_columns(df)
    X = df[cols].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int64)

    # Papers' protocol: random shuffle, 80/20
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=cfg["random_seed"], shuffle=True)

    # fast-ish bases for the benchmark (no LSTM sequences — tabular split only)
    cfg_b = json.loads(json.dumps(cfg))  # deep copy via json (dict of scalars)
    cfg_b["tabular"]["rf"]["n_estimators"] = 300
    cfg_b["tabular"]["xgb"]["n_estimators"] = 300
    cfg_b["tabular"]["lstm"]["epochs"] = 10

    ens = TabularEnsemble(cfg_b, n_features=len(cols), with_lstm=False)
    ens.fit(Xtr, ytr)
    p = ens.predict_proba(Xte)

    metrics = {
        "random_split_80_20": {
            **classification_metrics(yte, p),
            "modality": "tabular ensemble (RF+XGB)",
            "validation": "random split (paper protocol) — OPTIMISTIC",
        }
    }
    out = Path(cfg["paths"]["metrics"]) / "random_split_results.json"
    save_metrics(metrics, out)
    print(f"random-split accuracy: {metrics['random_split_80_20']['accuracy']:.4f} "
          f"(n_test={len(yte)}) → {out}")


if __name__ == "__main__":
    main()
