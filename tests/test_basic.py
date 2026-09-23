"""Minimal smoke tests for OptionEdge core modules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optionedge.config import load_config
from optionedge.fusion import meta_features, tune_threshold, FusionModel
from optionedge.options_features import bs_price, implied_vol, _max_pain
from optionedge.signals import probabilities_to_signals
from optionedge.tabular_models import make_sequences, TabularEnsemble
from optionedge.walkforward import make_walkforward_folds
from optionedge.dataset import chronological_splits, feature_columns


def test_bs_roundtrip():
    s, k, t, r, q, sigma = 100.0, 100.0, 30 / 365, 0.065, 0.01, 0.18
    px = bs_price(s, k, t, r, q, sigma, "CE")
    iv = implied_vol(np.array([px]), s, k, t, r, q, np.array(["CE"]))
    assert np.isfinite(iv[0]) and abs(iv[0] - sigma) < 1e-3, iv


def test_max_pain():
    strikes = np.array([100.0, 101.0, 102.0])
    ce = np.array([10.0, 20.0, 5.0])
    pe = np.array([5.0, 20.0, 10.0])
    mp, pay = _max_pain(strikes, ce, pe)
    assert mp in set(strikes.tolist())
    assert np.isfinite(pay)


def test_signals():
    p = np.array([0.60, 0.40, 0.50])
    s = probabilities_to_signals(p, hi=0.57, lo=0.43)
    assert s.tolist() == [1, -1, 0]


def test_meta_features_shape():
    p1 = np.array([0.5, 0.7, 0.3])
    p2 = np.array([0.55, 0.6, 0.4])
    M = meta_features(p1, p2)
    assert M.shape == (3, 7)
    M1 = meta_features(p1, None)
    assert M1.shape == (3, 1)


def test_tune_threshold():
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(0.5, 0.1, 500), 0.01, 0.99)
    y = (rng.random(500) < p).astype(int)
    thr = tune_threshold(p, y, 0.5)
    assert 0.51 <= thr["hi"] <= 0.8
    assert thr["lo"] == round(1 - thr["hi"], 4)


def test_sequences_causal():
    X = np.arange(50, dtype=float).reshape(-1, 1)
    y = (np.arange(50) % 2)
    Xs, ys, pos = make_sequences(X, y, lookback=5, positions=np.arange(50))
    assert len(Xs) == 45
    # last timestep of sequence i equals feature at position pos[i]
    assert Xs[0, -1, 0] == 5.0 and pos[0] == 5
    assert Xs[10, 0, 0] == 10.0  # window [10..15] wait: i=15 → start 10
    assert Xs[10, -1, 0] == 15.0 and pos[10] == 15


def test_walkforward_folds():
    folds = make_walkforward_folds({"NIFTY": 1000, "BANKNIFTY": 1000},
                                   n_folds=5, test_days=63)
    assert len(folds) >= 3
    for f in folds:
        assert f["test_start"] == f["train_end"]
        assert f["test_end"] - f["test_start"] == 63
    # strictly increasing, non-overlapping tests
    for a, b in zip(folds, folds[1:]):
        assert a["test_end"] <= b["test_start"]


def test_chronological_splits():
    tr, ca, te = chronological_splits(100, 0.2, 0.1)
    assert tr[-1] < ca[0] < te[0]
    assert len(tr) + len(ca) + len(te) == 100


def test_tabular_ensemble_tiny():
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    X[0, 0] = np.nan  # NaN handled
    cfg = load_config()
    cfg["tabular"]["rf"]["n_estimators"] = 30
    cfg["tabular"]["xgb"]["n_estimators"] = 30
    cfg["tabular"]["lstm"]["epochs"] = 1
    ens = TabularEnsemble(cfg, n_features=8, with_lstm=False)
    ens.fit(X, y)
    p = ens.predict_proba(X)
    assert p.shape == (n,)
    assert np.all((p >= 0) & (p <= 1))
    assert hasattr(ens, "oof_p_") and ens.oof_p_.shape == (n,)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
