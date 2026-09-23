"""Dataset assembly: features + labels + chronological splits.

Label: next-session direction of the underlying index (1 = up, 0 = down).
All features at time t use only information available at t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .technical_features import build_technicals
from .options_features import build_options_features


def build_feature_table(cfg: dict) -> pd.DataFrame:
    proc = Path(cfg["data"]["processed_dir"])
    prices = pd.read_parquet(proc / "prices.parquet")
    chain = pd.read_parquet(proc / "fo_chain.parquet")
    tech = build_technicals(prices)
    opt = build_options_features(chain, prices, r=cfg["data"]["risk_free_rate"])
    # opt has (date, symbol) possibly duplicated? ensure unique
    opt = opt.drop_duplicates(subset=["symbol", "date"])
    df = tech.merge(opt, on=["symbol", "date"], how="left", suffixes=("", "_opt"))
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Labels — NEXT SESSION open→close direction.
    #
    # The backtest enters at the next session's open and exits at its close,
    # so the training label must match that traded return (close→close labels
    # absorb overnight gaps that a next-open entry cannot capture).
    # Features at time t still use only information available at t: the future
    # open is used solely inside the label (labels may see the future).
    next_open = df.groupby("symbol")["open"].shift(-1)
    next_close = df.groupby("symbol")["close"].shift(-1)
    df["fwd_ret_1"] = next_close / next_open - 1.0          # session return
    df["fwd_ret_cc"] = df.groupby("symbol")["close"].transform(
        lambda s: s.shift(-1) / s - 1.0)                     # close→close (ref)
    df["label"] = (df["fwd_ret_1"] > 0).astype(np.float64)
    # drop rows without future label
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


FEATURE_EXCLUDE = {
    "date", "symbol", "label", "fwd_ret_1", "fwd_ret_cc",
    "open", "high", "low", "close",
    "volume", "turnover", "pe", "pb", "div_yield", "vix_close",
    # raw walls / level columns kept out of raw scale issues? keep levels as distances only
    "atm_strike", "ce_wall_strike", "pe_wall_strike", "max_pain",
    "ce_oi_wavg_strike", "pe_oi_wavg_strike", "fut_oi_total_base",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in FEATURE_EXCLUDE:
            continue
        if df[c].dtype.kind not in "fiu":
            continue
        cols.append(c)
    return cols


def chronological_splits(n: int, test_fraction: float, calib_fraction: float):
    """Return (train_idx, calib_idx, test_idx) as positional index arrays."""
    n_test = int(round(n * test_fraction))
    n_cal = int(round(n * calib_fraction))
    n_train = n - n_test - n_cal
    if n_train <= 0 or n_cal <= 0 or n_test <= 0:
        raise ValueError(f"Bad split for n={n}: test={test_fraction}, calib={calib_fraction}")
    train = np.arange(0, n_train)
    calib = np.arange(n_train, n_train + n_cal)
    test = np.arange(n_train + n_cal, n)
    return train, calib, test


def per_symbol_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}


def make_xy(frames: dict[str, pd.DataFrame], cols: list[str],
            idx_map: dict[str, np.ndarray]):
    """Assemble X, y across symbols for the given per-symbol positional indices."""
    Xs, ys, meta = [], [], []
    for sym, idx in idx_map.items():
        g = frames[sym]
        sub = g.iloc[idx]
        Xs.append(sub[cols].to_numpy(dtype=np.float32))
        ys.append(sub["label"].to_numpy(dtype=np.int64))
        for d in sub["date"]:
            meta.append({"symbol": sym, "date": d})
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return X, y, pd.DataFrame(meta)
