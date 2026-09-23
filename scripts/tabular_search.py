"""Fast tabular-only accuracy improvement search (no vision retrain).

Evaluates several strictly-causal configurations under walk-forward CV on the
tabular branch and writes a comparison JSON. Uses the same folds as the main
pipeline so numbers stay comparable.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optionedge.config import load_config, seed_everything  # noqa: E402
from optionedge.dataset import (build_feature_table, feature_columns,  # noqa: E402
                                per_symbol_frames)
from optionedge.tabular_models import TabularEnsemble, sequences_by_symbol  # noqa: E402
from optionedge.evaluate import classification_metrics, save_metrics  # noqa: E402
from optionedge.walkforward import make_walkforward_folds, _stack  # noqa: E402


def run_config(cfg, df, cols, folds, label, feature_subset=None):
    frames = per_symbol_frames(df)
    use_cols = feature_subset if feature_subset else cols
    ys, ps = [], []
    for fi, fold in enumerate(folds):
        tr_end, te_start, te_end = fold["train_end"], fold["test_start"], fold["test_end"]
        train_idx = {s: np.arange(0, tr_end) for s in frames}
        test_idx = {s: np.arange(te_start, te_end) for s in frames if te_end <= len(frames[s])}
        Xtr, ytr, _, _, _ = _stack(frames, use_cols, train_idx)
        Xte, yte, _, _, _ = _stack(frames, use_cols, test_idx)
        lookback = cfg["features"]["lookback_lstm"]
        X_seq, y_seq, meta = sequences_by_symbol(frames, use_cols, train_idx, lookback)
        seq_row = None
        if len(meta):
            base, seq_row = 0, np.empty(len(meta), dtype=int)
            for s in frames:
                mask = (meta["symbol"] == s).to_numpy()
                seq_row[mask] = base + meta.loc[mask, "pos"].to_numpy()
                base += tr_end
        ens = TabularEnsemble(cfg, n_features=len(use_cols), with_lstm=True)
        ens.fit(Xtr, ytr, X_seq=X_seq, y_seq=y_seq, seq_index_in_X=seq_row)
        # test sequences
        warm = {s: np.arange(max(0, te_start - lookback), te_end)
                for s in frames if te_end <= len(frames[s])}
        Xw, yw, metaw = sequences_by_symbol(frames, use_cols, warm, lookback)
        keep, rows = [], []
        offsets, acc = {}, 0
        for s in frames:
            if s in test_idx:
                offsets[s] = acc
                acc += te_end - te_start
        for i, m in enumerate(metaw.itertuples()):
            if m.pos >= te_start and m.symbol in offsets:
                keep.append(i)
                rows.append(offsets[m.symbol] + (int(m.pos) - te_start))
        if keep:
            p = ens.predict_proba(Xte, X_seq=Xw[keep], seq_index_in_X=np.array(rows))
        else:
            p = ens.predict_proba(Xte)
        ys.append(yte)
        ps.append(p)
        print(f"  [{label}] fold {fi}: n={len(yte)} acc={np.mean((p>=0.5)==yte):.3f}", flush=True)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    m = classification_metrics(y, p)
    print(f"[{label}] TOTAL acc={m['accuracy']:.4f} base={m['baseline_majority']:.4f} "
          f"edge={m['edge_vs_baseline']:+.4f} auc={m['roc_auc']:.4f}", flush=True)
    return m


def main():
    cfg = load_config()
    seed_everything(cfg["random_seed"])
    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "features.parquet")
    cols = feature_columns(df)
    n_per = {s: len(g) for s, g in per_symbol_frames(df).items()}
    folds = make_walkforward_folds(n_per, 5, cfg["split"]["walkforward_test_days"])

    results = {}

    # A) baseline config (matches pipeline defaults)
    c = copy.deepcopy(cfg)
    results["rf_xgb_lstm_default"] = run_config(c, df, cols, folds, "default")

    # B) stronger regularisation / more trees
    c = copy.deepcopy(cfg)
    c["tabular"]["xgb"].update(n_estimators=700, max_depth=3, learning_rate=0.03,
                               min_child_weight=6, reg_lambda=3.0, subsample=0.7,
                               colsample_bytree=0.6)
    c["tabular"]["rf"].update(n_estimators=700, max_depth=6, min_samples_leaf=12)
    results["regularized"] = run_config(c, df, cols, folds, "regularized")

    # C) feature selection: top 45 by RF importance on first fold's train (causal)
    rf_top = None
    try:
        from optionedge.tabular_models import build_rf
        frames = per_symbol_frames(df)
        tr_end = folds[0]["train_end"]
        X, y, _, _, _ = _stack(frames, cols, {s: np.arange(tr_end) for s in frames})
        rf = build_rf(cfg)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
        rf_top = list(imp.head(45).index)
        print("top features:", rf_top[:15], flush=True)
        c = copy.deepcopy(cfg)
        results["top45_rf_features"] = run_config(c, df, cols, folds, "top45",
                                                  feature_subset=rf_top)
    except Exception as e:
        print("feature select failed:", e)

    # D) momentum/oi focused hand-pick
    prefer = [c2 for c2 in cols if any(k in c2 for k in (
        "ret_", "rsi", "macd", "pcr", "iv_", "oi", "vol_", "cross", "boll",
        "max_pain", "fut_basis", "vix", "candle", "dd_from", "trend"))]
    if len(prefer) >= 20:
        c = copy.deepcopy(cfg)
        results["handpicked_domains"] = run_config(c, df, cols, folds, "handpicked",
                                                   feature_subset=prefer)

    out = Path(cfg["paths"]["metrics"]) / "tabular_search.json"
    save_metrics(results, out)
    print("saved", out)


if __name__ == "__main__":
    main()
