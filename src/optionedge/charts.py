"""Candlestick chart image generation for the vision branch (mplfinance)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

STYLE = mpf.make_mpf_style(
    base_mpf_style="yahoo",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge={"up": "#26a69a", "down": "#ef5350"},
        wick={"up": "#26a69a", "down": "#ef5350"},
        volume="in",
    ),
    gridcolor="#e0e0e0",
    gridstyle=":",
    facecolor="white",
    figcolor="white",
    rc={"font.size": 6},
)


def render_chart(df: pd.DataFrame, path: str | Path, window: int = 60,
                 image_size: int = 128) -> bool:
    """Render the last `window` sessions of OHLCV df to a PNG at ~image_size px."""
    hist = df.tail(window)
    if len(hist) < max(20, window // 2):
        return False
    hist = hist.copy()
    hist.index = pd.to_datetime(hist["date"])
    data = hist[["open", "high", "low", "close"]].astype(float)
    # volume in index bhavcopy can be 0 — fabricate a neutral series so the panel renders
    if "volume" in hist.columns:
        vol = hist["volume"].astype(float)
        if float(vol.sum()) <= 0:
            vol = pd.Series(np.full(len(hist), 1.0), index=data.index)
        data["volume"] = vol.values
    show_vol = "volume" in data.columns
    fig, axes = mpf.plot(
        data,
        volume=show_vol,
        type="candle",
        style=STYLE,
        xrotation=0,
        tight_layout=True,
        returnfig=True,
        figsize=(image_size / 100.0, image_size / 100.0),
        warn_too_much_data=10_000,
    )
    try:
        for ax in axes:
            ax.tick_params(labelsize=5)
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        return True
    finally:
        plt.close(fig)


def _render_job(payload):
    """Worker: payload = (sym, date, path, window, size, hist_df)."""
    sym, date, path, window, size, hist = payload
    try:
        ok = render_chart(hist, path, window=window, image_size=size)
    except Exception:
        ok = False
    return {"symbol": sym, "date": date, "path": path if ok else None, "ok": ok}


def generate_all_charts(cfg: dict, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Render charts for every session of every symbol.

    Returns manifest DataFrame: symbol, date, path (only successful renders).
    """
    from multiprocessing import Pool

    d = cfg["data"]
    charts_dir = Path(d["charts_dir"])
    charts_dir.mkdir(parents=True, exist_ok=True)
    if prices is None:
        prices = pd.read_parquet(Path(d["processed_dir"]) / "prices.parquet")

    window = cfg["features"]["chart_window"]
    size = cfg["vision"]["image_size"]

    jobs = []
    for sym, g in prices.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        # rolling window needs history: start emitting once we have window//2 sessions
        for i in range(len(g)):
            row = g.iloc[i]
            out = charts_dir / sym / f"{pd.Timestamp(row['date']).strftime('%Y%m%d')}.png"
            if out.exists():
                jobs.append((sym, str(row["date"])[:10], str(out), True))
                continue
            jobs.append((sym, str(row["date"])[:10], str(out), False))

    # group by symbol slices for workers
    todo = {sym: [] for sym in prices["symbol"].unique()}
    manifest_rows = []
    for sym, date, path, exists in jobs:
        if exists:
            manifest_rows.append({"symbol": sym, "date": date, "path": path})
        else:
            todo[sym].append((date, path))

    payloads = []
    for sym, items in todo.items():
        g = prices[prices["symbol"] == sym].sort_values("date").reset_index(drop=True)
        date_idx = {str(pd.Timestamp(d).date()): i for i, d in enumerate(g["date"])}
        for date, path in items:
            payloads.append((sym, date, path, window, size,
                             g.iloc[: date_idx[date] + 1][["date", "open", "high", "low",
                                                           "close", "volume"]].copy()))

    n_workers = max(1, min(2, os.cpu_count() or 1))
    if payloads:
        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_render_job, payloads, chunksize=8)):
                if res["ok"]:
                    manifest_rows.append({"symbol": res["symbol"], "date": res["date"],
                                          "path": res["path"]})
                if (i + 1) % 200 == 0:
                    print(f"  charts: {i + 1}/{len(payloads)}")
    manifest = pd.DataFrame(manifest_rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    manifest.to_csv(charts_dir / "manifest.csv", index=False)
    return manifest
