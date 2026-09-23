"""Technical / price-derived features (no lookahead: all values at time t use ≤ t)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_g = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_l = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_g / avg_l.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must contain OHLCV + valuation + vix columns for ONE symbol, sorted by date."""
    out = df.copy().sort_values("date").reset_index(drop=True)
    c, o, h, l, v = out["close"], out["open"], out["high"], out["low"], out["volume"]
    ret1 = c.pct_change()

    # --- returns / momentum -------------------------------------------------
    for n in (1, 2, 3, 5, 10, 21, 63):
        out[f"ret_{n}d"] = c.pct_change(n)
    out["mom_5_20"] = c / c.shift(20) - 1.0
    out["gap"] = o / c.shift(1) - 1.0
    out["intraday_ret"] = c / o - 1.0

    # --- moving averages / trend -------------------------------------------
    for n in (5, 10, 20, 50, 100, 200):
        ma = c.rolling(n).mean()
        out[f"sma_{n}_dist"] = c / ma - 1.0
    out["ema_12_26"] = c.ewm(span=12, adjust=False).mean() / c.ewm(span=26, adjust=False).mean() - 1.0
    out["cross_5_20"] = (c.rolling(5).mean() / c.rolling(20).mean()) - 1.0
    # diagonal R² of log-price over 20d (trend strength)
    x = np.arange(20)
    lx = np.log(c)
    slope = lx.rolling(20).apply(lambda s: np.polyfit(x, s.values, 1)[0], raw=False)
    out["trend_slope_20"] = slope * 100.0
    yhat = pd.DataFrame({i: lx.shift(19 - i) for i in range(20)}).rolling(20).mean()
    ss_res = ((lx - lx.rolling(20).mean()) ** 2).rolling(20).sum()  # placeholder scale
    out["hi_lo_range_20"] = (h.rolling(20).max() - l.rolling(20).min()) / c

    # --- oscillators --------------------------------------------------------
    for n in (7, 14, 21):
        out[f"rsi_{n}"] = _rsi(c, n)
    macd, signal, hist = _macd(c)
    out["macd"] = macd / c
    out["macd_signal"] = signal / c
    out["macd_hist"] = hist / c
    out["stoch_k"] = 100.0 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()
    m20 = c.rolling(20).mean()
    s20 = c.rolling(20).std()
    out["boll_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)
    out["boll_bw"] = (4 * s20) / m20

    # --- volatility ---------------------------------------------------------
    for n in (5, 10, 21, 63):
        out[f"vol_{n}"] = ret1.rolling(n).std() * np.sqrt(252.0)
    out["vol_ratio_5_21"] = out["vol_5"] / out["vol_21"].replace(0, np.nan)
    out["vol_of_vol"] = out["vol_21"].rolling(21).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr_14_pct"] = tr.rolling(14).mean() / c
    out["rvol_pos"] = (ret1.rolling(21).std().replace(0, np.nan))

    # --- candle shape -------------------------------------------------------
    body = (c - o)
    rng = (h - l).replace(0, np.nan)
    out["candle_body"] = body / rng
    out["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    out["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    out["body_abs"] = body.abs() / c

    # --- volume / turnover / valuation --------------------------------------
    out["vol_z_21"] = (v - v.rolling(21).mean()) / v.rolling(21).std().replace(0, np.nan)
    out["turnover_z_21"] = (out["turnover"] - out["turnover"].rolling(21).mean()) / \
                           out["turnover"].rolling(21).std().replace(0, np.nan)
    for col in ("pe", "pb", "div_yield"):
        mu = out[col].rolling(252, min_periods=60).mean()
        sd = out[col].rolling(252, min_periods=60).std().replace(0, np.nan)
        out[f"{col}_z"] = (out[col] - mu) / sd

    # --- drawdown / range position ------------------------------------------
    out["dd_from_252_high"] = c / c.rolling(252, min_periods=60).max() - 1.0
    out["pos_in_63_range"] = (c - l.rolling(63).min()) / \
                             (h.rolling(63).max() - l.rolling(63).min()).replace(0, np.nan)

    # --- VIX ----------------------------------------------------------------
    if "vix_close" in out.columns:
        out["vix_chg_1"] = out["vix_close"].pct_change(1)
        out["vix_chg_5"] = out["vix_close"].pct_change(5)
        out["vix_level"] = out["vix_close"]
        out["vix_hi_ratio"] = out["vix_close"] / out["vix_close"].rolling(63).max().replace(0, np.nan)

    # day-of-week (cyclic) — mild but harmless
    dow = out["date"].dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return out


def build_technicals(prices: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, g in prices.groupby("symbol"):
        frames.append(add_technical_features(g))
    return pd.concat(frames, ignore_index=True)
