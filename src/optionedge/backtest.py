"""Backtest: hold ATM option for one session using REAL contract OHLC prices.

Execution model (conservative, next-session):
  • Signal computed after close of day t (features ≤ t).
  • Enter at OPEN of ATM option contract on day t+1 (nearest expiry, ATM by t+1 spot).
  • Exit at CLOSE of the same contract on day t+1 (intraday/1-day hold), unless
    stop-loss / take-profit on intraday low/high triggers (checked with day range).
  • Costs: flat points (brokerage+slippage approximation) round-trip.

P&L is reported in index option points (lot-size agnostic) and INR using
configurable lot sizes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atm_price(chain_day: pd.DataFrame, spot: float, side: str,
               date, expiry) -> dict | None:
    """Pick ATM CE/PE row from a single day's option chain slice."""
    if chain_day.empty:
        return None
    opts = chain_day[(chain_day["instrument"] == "IDX_OPT") &
                     (chain_day["option_type"] == side)]
    if opts.empty:
        return None
    step = 50.0 if spot < 40000 else 100.0
    atm = round(spot / step) * step
    row = opts[opts["strike"] == atm]
    if row.empty:
        # nearest strike
        row = opts.iloc[[(opts["strike"] - atm).abs().argmin()]]
    r = row.iloc[0]
    return {"strike": float(r["strike"]), "open": float(r["open"]),
            "high": float(r["high"]), "low": float(r["low"]),
            "close": float(r["close"]), "expiry": r["expiry"], "oi": r["oi"]}


def run_backtest(cfg: dict, signals: pd.DataFrame, prices: pd.DataFrame,
                 chain: pd.DataFrame) -> dict:
    """signals must be time-ordered per symbol with columns date,symbol,signal,p_up."""
    b = cfg["backtest"]
    lots = cfg["data"]["lot_size"]
    chain_ix = {(s, pd.Timestamp(d).normalize()): g
                for (s, d), g in chain[chain["instrument"] == "IDX_OPT"]
                    .assign(date=lambda x: pd.to_datetime(x["date"]).dt.normalize())
                    .groupby(["symbol", "date"])}
    px_ix = {(r["symbol"], pd.Timestamp(r["date"]).normalize()): r
             for _, r in prices.iterrows()}

    trades = []
    equity = []
    cash = 0.0
    per_symbol_equity: dict[str, float] = {}

    sig = signals.sort_values(["symbol", "date"]).copy()
    for sym, g in sig.groupby("symbol"):
        g = g.reset_index(drop=True)
        per_symbol_equity[sym] = 0.0
        for i in range(len(g) - 1):          # signal day t → trade day t+1
            row = g.iloc[i]
            if int(row["signal"]) == 0:
                continue
            day_next = g.iloc[i + 1]["date"]
            t_date = pd.Timestamp(row["date"]).normalize()
            t1 = pd.Timestamp(day_next).normalize()
            side = "CE" if int(row["signal"]) == 1 else "PE"
            # spot at t+1 (use next day's index row if present)
            px_next = px_ix.get((sym, t1))
            if px_next is None:
                px_next = px_ix.get((sym, t_date))
            if px_next is None:
                continue
            # ATM strike by the session OPEN (execution moment — no look-ahead)
            spot1 = float(px_next.get("open", np.nan))
            if not np.isfinite(spot1) or spot1 <= 0:
                spot1 = float(px_next["close"])
            idx_open = float(px_next.get("open", np.nan))
            idx_close = float(px_next["close"])
            session_ret = (idx_close / idx_open - 1.0) if np.isfinite(idx_open) and idx_open > 0 else np.nan
            day_chain = chain_ix.get((sym, t1))
            if day_chain is None:
                continue
            # nearest expiry ≥ t1
            expiries = sorted(pd.to_datetime(day_chain["expiry"].dropna().unique()))
            exp1 = next((e for e in expiries if pd.Timestamp(e) >= t1),
                        expiries[0] if expiries else None)
            if exp1 is None:
                continue
            day_exp = day_chain[day_chain["expiry"] == exp1]
            legs = _atm_price(day_exp, spot1, side, t1, exp1)
            if legs is None:
                continue
            entry = legs["open"]
            if not np.isfinite(entry) or entry <= 0.05:
                continue
            hi, lo, close_px = legs["high"], legs["low"], legs["close"]
            stop = entry * (1.0 - b["stop_loss_pct"])
            target = entry * (1.0 + b["take_profit_pct"])
            # intraday path assumption: check stop first (conservative), then target
            exit_px = close_px
            exit_reason = "close"
            if lo <= stop:
                exit_px, exit_reason = stop, "stop"
            elif hi >= target:
                exit_px, exit_reason = target, "target"
            gross = exit_px - entry
            # long option P&L = exit − entry (same for CE and PE)
            costs = b["costs_points"] + b["slippage_points"]
            net = gross - costs
            lot = lots.get(sym, 75)
            per_symbol_equity[sym] += net
            trades.append({
                "symbol": sym,
                "signal_date": t_date,
                "trade_date": t1,
                "side": side,
                "strike": legs["strike"],
                "entry": entry,
                "exit": exit_px,
                "exit_reason": exit_reason,
                "gross_points": gross,
                "net_points": net,
                "net_inr": net * lot,
                "signal_p_up": float(row["p_up"]),
                "confidence": float(row["confidence"]),
                "spot_move_pct": float(session_ret),
                "direction_correct": bool(
                    (session_ret > 0 and side == "CE") or
                    (session_ret < 0 and side == "PE"))
                    if np.isfinite(session_ret) else None,
            })
            cash += net
            equity.append({"symbol": sym, "trade_date": t1, "cum_points": per_symbol_equity[sym]})

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    eq_df = pd.DataFrame(equity) if equity else pd.DataFrame()

    stats: dict = {"n_trades": int(len(trades_df))}
    if len(trades_df):
        wins = trades_df["net_points"] > 0
        stats.update(
            win_rate=float(wins.mean()),
            avg_net_points=float(trades_df["net_points"].mean()),
            total_net_points=float(trades_df["net_points"].sum()),
            total_net_inr=float(trades_df["net_inr"].sum()),
            profit_factor=float(trades_df.loc[wins, "net_points"].sum() /
                                max(-trades_df.loc[~wins, "net_points"].sum(), 1e-9)),
            avg_confidence=float(trades_df["confidence"].mean()),
            by_symbol={
                s: {
                    "n": int(len(g)),
                    "win_rate": float((g["net_points"] > 0).mean()),
                    "total_points": float(g["net_points"].sum()),
                } for s, g in trades_df.groupby("symbol")
            },
            exit_reasons=trades_df["exit_reason"].value_counts().to_dict(),
        )
        # equity curve / max drawdown on cumulative points across all trades
        cum = trades_df["net_points"].cumsum()
        peak = cum.cummax()
        stats["max_drawdown_points"] = float((cum - peak).min())
        stats["sharpe_per_trade"] = float(trades_df["net_points"].mean() /
                                          max(trades_df["net_points"].std(), 1e-9))
    return {"trades": trades_df, "equity": eq_df, "stats": stats}


def buy_hold_underlying(prices: pd.DataFrame, start, end) -> float:
    p = prices[(prices["date"] >= start) & (prices["date"] <= end)]
    if len(p) < 2:
        return float("nan")
    return float(p.iloc[-1]["close"] / p.iloc[0]["close"] - 1.0)
