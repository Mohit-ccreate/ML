"""OptionEdge — Streamlit dashboard.

Run:  streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optionedge.config import load_config  # noqa: E402

st.set_page_config(page_title="OptionEdge", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

CFG = load_config()
METRICS = Path(CFG["paths"]["metrics"])
PLOTS = Path(CFG["paths"]["plots"])
CHARTS = Path(CFG["data"]["charts_dir"])
PROC = Path(CFG["data"]["processed_dir"])

# ---------------------------------------------------------------------------
# CSS / branding
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: #0b1220; }
      h1, h2, h3 { color: #e8eefc !important; }
      .metric-card {
        background: linear-gradient(160deg, #121a2e, #0e1526);
        border: 1px solid #22304f;
        border-radius: 14px; padding: 16px 18px; margin: 6px 0;
      }
      .metric-card .value { font-size: 1.9rem; font-weight: 700; color: #7dd3fc; }
      .metric-card .label { font-size: .85rem; color: #93a4c4; letter-spacing: .04em; }
      .buy-ce { color: #26a69a; font-weight: 700; }
      .buy-pe { color: #ef5350; font-weight: 700; }
      .flat { color: #fbbf24; font-weight: 700; }
      div[data-testid="stMetricValue"] { color: #7dd3fc; }
      .stTabs [data-baseweb="tab"] { color: #cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_json(name: str):
    p = METRICS / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _load_csv(name: str) -> pd.DataFrame | None:
    p = METRICS / name
    return pd.read_csv(p) if p.exists() else None


def _load_parquet(name: str) -> pd.DataFrame | None:
    p = METRICS / name
    return pd.read_parquet(p) if p.exists() else None


def card(value: str, label: str):
    return (f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>')


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ OptionEdge")
    st.caption("Multi-modal ViT + Options-Chain Ensemble\nNIFTY 50 · BANKNIFTY")
    page = st.radio("Navigate", ["Live Signal", "Model Performance",
                                 "Backtest", "Vision Lab", "Architecture"],
                    index=0)
    st.divider()
    holdout = _load_json("holdout_results.json")
    wf = _load_json("walkforward_results.json")
    bt = _load_json("backtest_stats.json")
    if wf:
        st.caption(f"Walk-forward fused acc: **{wf.get('fused',{}).get('accuracy', float('nan')):.1%}**")
    if holdout:
        st.caption(f"Holdout fused acc: **{holdout.get('fused',{}).get('accuracy', float('nan')):.1%}**")
    st.caption("Data: NSE bhavcopy 2021–2026 · walk-forward validated")

# ---------------------------------------------------------------------------
# Page: Live Signal
# ---------------------------------------------------------------------------
if page == "Live Signal":
    st.title("📡 Latest Session Signal")
    latest = _load_json("latest_signals.json")
    if not latest:
        st.info("No signals yet — run `python scripts/run_pipeline.py` (holdout stage) first.")
    else:
        cols = st.columns(len(latest))
        for c, row in zip(cols, latest):
            sig = row.get("signal", "NO_TRADE")
            cls = {"BUY_CE": "buy-ce", "BUY_PE": "buy-pe"}.get(sig, "flat")
            arrow = {"BUY_CE": "▲", "BUY_PE": "▼"}.get(sig, "•")
            name = "NIFTY 50" if row["symbol"] == "NIFTY" else "BANKNIFTY"
            with c:
                st.markdown(card(f'<span class="{cls}">{arrow} {sig}</span>',
                                 f"{name} · {row['date']}"), unsafe_allow_html=True)
                st.progress(min(max(float(row["confidence"]), 0.0), 1.0),
                            text=f"confidence {row['confidence']:.1%}")
                st.caption(f"P(up) fused **{row['p_up_fused']:.3f}** · "
                           f"tabular {row['p_up_tabular']:.3f} · vision {row['p_up_vit']:.3f}")

        st.divider()
        st.subheader("Signal rules")
        thr = latest[0].get("thresholds", {"hi": 0.57, "lo": 0.43})
        st.markdown(
            f"""
            | Model output | Action |
            |---|---|
            | P(up) ≥ **{thr['hi']:.2f}** | **BUY CE** (ATM call, 1-day hold) |
            | P(up) ≤ **{thr['lo']:.2f}** | **BUY PE** (ATM put, 1-day hold) |
            | otherwise | **NO TRADE** |
            """
        )
        st.caption("Thresholds calibrated on the pre-test calibration window only.")

    # recent walk-forward predictions as a "recent calls" table
    wfp = _load_parquet("walkforward_predictions.parquet")
    if wfp is not None and len(wfp):
        st.subheader("Recent walk-forward calls (out-of-sample)")
        recent = wfp.sort_values("date").tail(20).copy()
        recent["pred"] = (recent["p_fused"] >= 0.5).astype(int)
        recent["hit"] = recent["pred"] == recent["y"]
        recent["direction"] = np.where(recent["pred"] == 1, "UP", "DOWN")
        recent["actual"] = np.where(recent["y"] == 1, "UP", "DOWN")
        st.dataframe(
            recent[["date", "symbol", "direction", "actual", "hit",
                    "p_fused", "p_tab", "p_vit"]].reset_index(drop=True),
            use_container_width=True)

    # underlying latest candles
    prices_p = PROC / "prices.parquet"
    if prices_p.exists():
        st.subheader("Underlying — last 120 sessions")
        pr = pd.read_parquet(prices_p)
        for sym, pretty in [("NIFTY", "NIFTY 50"), ("BANKNIFTY", "BANKNIFTY")]:
            g = pr[pr["symbol"] == sym].sort_values("date").tail(120)
            if len(g):
                st.markdown(f"**{pretty}** — close {g.iloc[-1]['close']:,.2f} "
                            f"({g.iloc[-1]['date'].date()})")
                chart_data = pd.Series(g["close"].values, index=pd.to_datetime(g["date"]))
                st.line_chart(chart_data, height=180)

# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("📊 Model Performance")
    if wf is None:
        st.info("Walk-forward results not found yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Walk-forward accuracy (fused)",
                  f"{wf['fused']['accuracy']:.2%}")
        c2.metric("Tabular only", f"{wf['tabular']['accuracy']:.2%}")
        c3.metric("Vision only", f"{wf['vision_vit']['accuracy']:.2%}")
        c4.metric("ROC-AUC (fused)", f"{wf['fused'].get('roc_auc', float('nan')):.3f}")

        if "fused_selective" in wf and wf["fused_selective"].get("n_selected", 0) > 0:
            s = wf["fused_selective"]
            st.info(f"**Selective (trade-only) accuracy:** {s['selective_accuracy']:.2%} "
                    f"on {s['coverage']:.0%} of sessions "
                    f"(n={s['n_selected']}) — thresholds respect "
                    f"hi={s.get('hi', 0.57):.2f} / lo={s.get('lo', 0.43):.2f}.")

        tabs = st.tabs(["Holdout ablation", "Benchmark vs papers",
                        "Fold details", "Curves"])
        with tabs[0]:
            if holdout:
                rows = []
                for key, label in [("tabular", "Tabular (RF+XGB+LSTM)"),
                                   ("vision_vit", "ViT (charts)"),
                                   ("fused", "Fused multi-modal")]:
                    m = holdout[key]
                    rows.append({"model": label, "accuracy": m["accuracy"],
                                 "f1": m["f1"], "precision": m["precision"],
                                 "recall": m["recall"], "roc_auc": m.get("roc_auc"),
                                 "n": m["n"]})
                st.dataframe(pd.DataFrame(rows).style.format(
                    {"accuracy": "{:.2%}", "f1": "{:.3f}", "precision": "{:.3f}",
                     "recall": "{:.3f}", "roc_auc": "{:.3f}"}),
                    use_container_width=True, hide_index=True)
                cm = np.array(holdout["fused"]["confusion_matrix"])
                st.caption("Holdout confusion (fused):")
                st.dataframe(pd.DataFrame(
                    cm, index=["actual Down", "actual Up"],
                    columns=["pred Down", "pred Up"]), use_container_width=True)
        with tabs[1]:
            bench_p = METRICS / "benchmark_table.csv"
            if bench_p.exists():
                b = pd.read_csv(bench_p)
                st.dataframe(b.style.format({"accuracy": "{:.2%}"}),
                             use_container_width=True, hide_index=True)
                st.caption(
                    "Paper numbers use random-split or proprietary protocols; "
                    "OptionEdge numbers are walk-forward / chronological holdout — "
                    "the honest apples-to-apples comparison is against Weinberg (2025) "
                    "realistic-validation figures (55–60%).")
            else:
                st.info("Benchmark table will appear after the reports stage.")
        with tabs[2]:
            folds_p = METRICS / "walkforward_folds.json"
            if folds_p.exists():
                st.json(json.loads(folds_p.read_text()))
        with tabs[3]:
            for name in ["wf_cumulative_accuracy.png", "confusion_fused.png"]:
                p = PLOTS / name
                if p.exists():
                    st.image(str(p), caption=name)

# ---------------------------------------------------------------------------
# Page: Backtest
# ---------------------------------------------------------------------------
elif page == "Backtest":
    st.title("💰 Backtest — Holdout Window")
    if bt is None:
        st.info("Backtest stats not found — run the backtest stage.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", f"{bt.get('n_trades', 0)}")
        c2.metric("Win rate", f"{bt.get('win_rate', float('nan')):.1%}")
        c3.metric("Net P&L (pts)", f"{bt.get('total_net_points', float('nan')):,.1f}")
        c4.metric("Profit factor", f"{bt.get('profit_factor', float('nan')):.2f}")
        c5, c6 = st.columns(2)
        c5.metric("Max drawdown (pts)", f"{bt.get('max_drawdown_points', float('nan')):,.1f}")
        c6.metric("Avg confidence", f"{bt.get('avg_confidence', float('nan')):.1%}")

        if "by_symbol" in bt:
            st.subheader("By index")
            st.json(bt["by_symbol"])
        if "exit_reasons" in bt:
            st.caption(f"Exits: {bt['exit_reasons']}")

        p = PLOTS / "backtest_equity.png"
        if p.exists():
            st.image(str(p), caption="Cumulative P&L (index option points)")
        trades = _load_csv("backtest_trades.csv")
        if trades is not None and len(trades):
            st.subheader("Trade log")
            show = trades.sort_values("trade_date", ascending=False).head(50)
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.download_button("Download full trade log (CSV)",
                               trades.to_csv(index=False), "optionedge_trades.csv")

# ---------------------------------------------------------------------------
# Page: Vision Lab
# ---------------------------------------------------------------------------
elif page == "Vision Lab":
    st.title("👁️ Vision Branch — Candlestick ViT")
    st.caption("60-session candlestick windows → ViT-Tiny (patch-16, dim-128) "
               "→ P(up). No pretrained weights required; trained end-to-end.")

    manifest_p = CHARTS / "manifest.csv"
    prices_p = PROC / "prices.parquet"
    wfp = _load_parquet("holdout_predictions.parquet") or _load_parquet(
        "walkforward_predictions.parquet")

    if manifest_p.exists() and prices_p.exists():
        man = pd.read_csv(manifest_p)
        pr = pd.read_parquet(prices_p)
        c1, c2 = st.columns(2)
        with c1:
            sym = st.selectbox("Index", sorted(man["symbol"].unique()))
        with c2:
            dates = sorted(man[man["symbol"] == sym]["date"].astype(str).unique())
            date = st.selectbox("Session", dates[-1:] + dates[::-1][1:] or dates)
        row = man[(man["symbol"] == sym) & (man["date"].astype(str) == date)]
        if len(row):
            st.image(row.iloc[0]["path"], width=420)
            g = pr[pr["symbol"] == sym].sort_values("date")
            gi = g[g["date"].astype(str) <= date].tail(60)
            if len(gi):
                st.caption(f"{sym} — last {len(gi)} sessions into this window, "
                           f"close {gi.iloc[-1]['close']:,.2f}")
        # vision-only recent accuracy
        if wfp is not None and "p_vit" in wfp.columns and "y" in wfp.columns:
            v = wfp.dropna(subset=["p_vit"])
            if len(v):
                acc = float(((v["p_vit"] >= 0.5).astype(int) == v["y"]).mean())
                st.metric("Vision-only accuracy on this eval set", f"{acc:.2%}")
        st.caption(f"Manifest covers {len(man)} chart images across "
                   f"{man['symbol'].nunique()} indices.")
    else:
        st.info("Charts not generated yet — run the charts stage.")

# ---------------------------------------------------------------------------
# Page: Architecture
# ---------------------------------------------------------------------------
else:
    st.title("🏗️ System Architecture")
    st.markdown(
        """
**Branch A — Vision**
1. 60-session candlestick windows rendered with *mplfinance*
2. ViT-Tiny (128×128, patch 16, 4 blocks × dim 128) trained end-to-end → P(up)

**Branch B — Options chain + technicals**
1. NSE F&O bhavcopy → OI, ΔOI, PCR, walls, max-pain, futures basis
2. Black–Scholes IV inversion on closing option premiums → IV skew, term structure, Greeks
3. Technicals: RSI, MACD, MAs, ATR, candle shape, VIX features (116 features)
4. RF + XGBoost + LSTM (20-day sequences) → logistic stacking → P(up)

**Fusion**
- Meta logistic-regression on `[p_tab, p_vit, disagreement, products, entropy]`
- Thresholds calibrated pre-test → BUY CE / BUY PE / NO TRADE

**Validation**
- Walk-forward expanding window, 5 folds × 63 sessions, models refit each fold
- Final chronological holdout (20%) for ablation + backtest
        """
    )
    st.subheader("Pipeline stages")
    st.code(
        "ingest → features → charts → holdout → walkforward → backtest → reports",
        language="bash",
    )
    st.subheader("Data sources")
    st.markdown(
        "- NSE daily *ind_close_all* bhavcopy (all indices OHLC + VIX + valuations)\n"
        "- NSE F&O bhavcopy (legacy + UDIFF formats) — OI/ΔOI/OHLC per contract\n"
        "- Study window: 2021-01-01 → 2026-09-22"
    )
    bench_p = METRICS / "benchmark_table.csv"
    if bench_p.exists():
        st.subheader("Benchmarking")
        st.dataframe(pd.read_csv(bench_p).style.format({"accuracy": "{:.2%}"}),
                     use_container_width=True, hide_index=True)
