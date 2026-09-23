"""SQLite storage for prices, features, predictions and backtest results."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _write_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False)


def store_all(cfg: dict, prices: pd.DataFrame, features: pd.DataFrame,
              walkforward_preds: pd.DataFrame | None,
              holdout_preds: pd.DataFrame | None,
              backtest_trades: pd.DataFrame | None,
              metrics: dict | None) -> None:
    conn = connect(cfg["paths"]["db"])
    try:
        _write_df(conn, prices.assign(date=prices["date"].astype(str)), "prices")
        f = features.copy()
        f["date"] = f["date"].astype(str)
        _write_df(conn, f, "features")
        if walkforward_preds is not None and len(walkforward_preds):
            _write_df(conn, walkforward_preds, "predictions_walkforward")
        if holdout_preds is not None and len(holdout_preds):
            _write_df(conn, holdout_preds, "predictions_holdout")
        if backtest_trades is not None and len(backtest_trades):
            t = backtest_trades.copy()
            for c in ("signal_date", "trade_date"):
                if c in t.columns:
                    t[c] = t[c].astype(str)
            _write_df(conn, t, "backtest_trades")
        if metrics is not None:
            m = pd.DataFrame([{"key": k, "value": str(v)} for k, v in metrics.items()])
            _write_df(conn, m, "metrics")
        conn.commit()
    finally:
        conn.close()
