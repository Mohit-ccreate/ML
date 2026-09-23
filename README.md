# ⚡ OptionEdge

**Multi-modal short-term direction prediction for NIFTY 50 & BANKNIFTY → Buy CE / Buy PE / No Trade signals.**

OptionEdge fuses two signal families that the literature has so far kept separate:

| Branch | Input | Model |
|---|---|---|
| **A — Vision** | 60-session candlestick chart images (mplfinance) | ViT-Tiny trained end-to-end |
| **B — Tabular** | Options-chain microstructure (OI, ΔOI, PCR, walls, max-pain, BS-IV skew, Greeks) + technicals (RSI, MACD, MAs, ATR, VIX…) — **116 features** | Random Forest + XGBoost + LSTM → logistic stacking |
| **Fusion** | `[p_tab, p_vit, disagreement, products, entropy]` | Logistic meta-classifier → calibrated thresholds → signal |

**Validation is walk-forward (time-ordered), never a random split** — expanding-window folds with full model refits, plus a final chronological 20% holdout used for the ablation table and options backtest.

---

## Results snapshot

See `artifacts/metrics/benchmark_table.csv` after a full run, and the dashboard
(**Model Performance** page) for:

* walk-forward accuracy of tabular / vision / fused models
* holdout ablation (does fusion beat each single modality?)
* selective (trade-only) accuracy at calibrated thresholds
* backtest win-rate, P&L in option points, profit factor, drawdown

Honest-verification note: papers reporting 87–96% typically use **random**
train/test splits on price rows. Under walk-forward validation, realistic
next-day index accuracy sits in the 50–62% band (Weinberg 2025: 60.14%).
OptionEdge reports both the walk-forward number and the selective accuracy on
days the model is confident enough to trade.

---

## Quick start

```bash
pip install -r requirements.txt

# Full pipeline (ingest → features → charts → holdout → walkforward → backtest → reports)
python scripts/run_pipeline.py            # ~40 min on a 2-core CPU
# Smoke test on a reduced window:
python scripts/run_pipeline.py --fast

# Dashboard
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501

# Unit tests
python tests/test_basic.py
```

Stages can also be run individually: `--stage ingest|features|charts|holdout|walkforward|backtest|reports`.

---

## Data

Real NSE daily data (no synthetic prices):

* **Index bhavcopy** `ind_close_all_*` — OHLC of *Nifty 50*, *Nifty Bank*, *India VIX*, plus P/E, P/B, dividend yield (2015–2026 mirror in `data/raw/quantdata`).
* **F&O bhavcopy** — per-contract OI, ΔOI, OHLC for NIFTY/BANKNIFTY index options & futures, both the legacy format (≤2023) and NSE UDIFF format (2024+). Implied volatility is **not** published in the bhavcopy, so OptionEdge recovers it by inverting Black–Scholes on closing premiums (r = 6.5%, q = index dividend yield), then derives **IV skew, term structure and portfolio Greeks**.

The mirror lives in `data/raw/` (git-ignored, ~2.5 GB):

```bash
git clone --filter=blob:none --sparse https://github.com/noswear94/quantdata.git data/raw/quantdata
cd data/raw/quantdata
git sparse-checkout set --no-cone 'raw/bhavcopy/index/*' 'raw/bhavcopy/fo/*'
```

Offline / restricted environments: place any CSVs with the same schema under `data/raw/quantdata/raw/bhavcopy/{index,fo}/…` and the ingest stage will pick them up. A yfinance fallback loader for `^NSEI` / `^NSEBANK` is included for environments with internet access.

---

## Repository layout

```
config.yaml               # window, splits, model hyper-parameters, thresholds
src/optionedge/
  data_ingest.py          # NSE bhavcopy → prices + normalised option chain
  technical_features.py   # 60+ price/volume/VIX features
  options_features.py     # PCR, OI walls, max-pain, BS IV inversion, Greeks
  dataset.py              # feature table, labels (next-session direction)
  charts.py               # candlestick PNG rendering
  vit.py, vit_worker.py   # ViT-Tiny (subprocess-isolated training)
  tabular_models.py       # RF + XGBoost + LSTM stacking ensemble
  fusion.py               # multi-modal meta-classifier + threshold tuning
  walkforward.py          # time-ordered refit CV of the FULL pipeline
  backtest.py             # real ATM option OHLC 1-day hold simulation
  evaluate.py             # metrics, benchmark table, plots
  db.py                   # SQLite persistence
scripts/run_pipeline.py   # orchestrator (--fast / --stage)
scripts/vit_worker.py     # torch subprocess worker
dashboard/app.py          # Streamlit UI (signals, metrics, backtest, vision lab)
tests/test_basic.py       # unit tests
docs/PRESENTATION.md      # slide-by-slide deck for the sponsor demo
```

---

## Signal contract

After the close of session *t*:

* `P(up) ≥ hi` → **BUY CE** (ATM call, entered next open, exited next close)
* `P(up) ≤ lo` → **BUY PE**
* otherwise → **NO TRADE**

`hi` / `lo` are calibrated on the pre-test calibration window only. The
backtest uses the **actual ATM option contract OHLC** from the F&O bhavcopy
(next-session open → close, with stop-loss / take-profit on the day's range),
minus round-trip costs in index-option points.

---

## Citation baselines

| Paper | Reported | Protocol |
|---|---|---|
| Sherasiya (2025) — options-chain LSTM | 90.10% | random split |
| Kayit & Ismail (2025) — voting ensemble | 95.8% | paper protocol |
| Kpereobong et al. (2025) — ViT + TFT | 96%+ | paper protocol |
| Weinberg (2025) — realistic validation | 60.14% | realistic |

OptionEdge’s walk-forward numbers should be read next to Weinberg; the
paper-split numbers are shown only as reference points in
`benchmark_table.csv`.

---

## Disclaimer

Research / educational software. Not investment advice. Trading derivatives
carries risk of loss.
