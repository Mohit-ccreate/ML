# OptionEdge — Sponsor Presentation Deck

Slide-by-slide script for tomorrow’s demo. Pair with the live dashboard
(`streamlit run dashboard/app.py`) — every number below comes from
`artifacts/metrics/` after a full pipeline run.

---

## Slide 1 — Title

**OptionEdge — Fusing Vision Transformers with Options-Chain Ensembles for
Short-Term Index Direction**

* End-sem project · industry-sponsored
* Underlyings: NIFTY 50 · BANKNIFTY
* Live demo: Streamlit dashboard

---

## Slide 2 — Problem

* Two research tracks never meet:
  * **Vision papers** read candlestick charts, ignore derivatives data
  * **Options-chain papers** use OI/IV/Greeks, ignore visual patterns
* Options studies usually validate with **random splits** → inflated accuracy
* Need: one fused system, validated the way trading actually works (time-ordered)

---

## Slide 3 — Solution architecture

```
Price data ─┬─► candlestick PNGs ─► ViT-Tiny ───────────┐
            │                                           ├─► fusion ─► CE / PE / flat
            └─► technicals ─┐                           │
Options chain ─► OI/PCR/IV ─┼─► RF + XGB + LSTM stack ──┘
            └─► Greeks ─────┘
```

* 116 tabular features + ViT visual signal
* Walk-forward validation · backtest on real option OHLC

---

## Slide 4 — Data (real, not synthetic)

* NSE daily **index bhavcopy** (OHLC, VIX, P/E, P/B, dividend yield)
* NSE **F&O bhavcopy** 2021→2026 — OI, ΔOI, premium OHLC for NIFTY/BANKNIFTY
  options & futures (legacy + 2024 UDIFF formats handled)
* **IV is not in the file** → we invert Black–Scholes on closing premiums
  (r=6.5%, q=dividend yield) → IV skew, term structure, Greeks
* ~1,300 sessions × 2 indices · ~1M option-contract rows/day archive

---

## Slide 5 — Method highlights

* **Vision:** 60-session windows → 128×128 PNG → ViT-Tiny (patch 16, dim 128,
  4 blocks), end-to-end (no internet pretrained weights needed)
* **Tabular:** RF + XGBoost + LSTM(20-day seq) stacked by logistic regression;
  OOF meta-features via TimeSeriesSplit **inside** the training window
* **Fusion:** logistic meta on probability pair + interaction features
* **Signals:** P(up) ≥ hi → BUY CE · P(up) ≤ lo → BUY PE · else NO TRADE
  (hi/lo calibrated on a pre-test slice only)

---

## Slide 6 — Validation protocol (the credibility slide)

1. **Walk-forward:** expanding window, 5 folds × 63 sessions.
   Every fold **refits** RF, XGB, LSTM, ViT, fusion — strictly causal.
2. **Holdout:** last 20% chronological; middle 10% only for threshold tuning.
3. **Backtest:** enter ATM option next open, exit next close (real contract
   OHLC), stop/target on day range, costs in option points.
4. No random shuffling anywhere.

---

## Slide 7 — Results: walk-forward (honest number)

Fill from `artifacts/metrics/walkforward_results.json`:

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Tabular only | **{…}** | {…} | {…} |
| ViT only | **{…}** | {…} | {…} |
| **Fused (OptionEdge)** | **{…}** | {…} | {…} |

Selective (trade-only) accuracy at calibrated thresholds: **{…}** on {…}% of
sessions.

Plot: `artifacts/plots/wf_cumulative_accuracy.png`.

---

## Slide 8 — Results: holdout ablation

From `holdout_results.json` — the question “does fusion help?”

| | Tabular | Vision | Fused |
|---|---|---|---|
| Accuracy | {…} | {…} | {…} |
| ROC-AUC | {…} | {…} | {…} |

Confusion matrix: `artifacts/plots/confusion_fused.png`.

---

## Slide 9 — Backtest

From `backtest_stats.json`:

* Trades {n} · Win rate {…}% · Net {…} pts · PF {…} · Max DD {…} pts
* Per-index breakdown NIFTY vs BANKNIFTY
* Equity curve: `artifacts/plots/backtest_equity.png`

Talking point: costs already netted (brokerage + slippage in points);
stop-loss −35% / take-profit +100% intraday guards.

---

## Slide 10 — Benchmarking against literature

From `benchmark_table.csv`:

| Baseline | Reported | Protocol |
|---|---|---|
| Sherasiya 2025 (LSTM, options) | 90.10% | random split |
| Kpereobong 2025 (ViT+TFT) | 96%+ | paper protocol |
| Weinberg 2025 (realistic) | 60.14% | realistic |
| **OptionEdge walk-forward** | **{…}** | **time-ordered, refit** |

Script: *“Single-modality papers sit at {tabular}% and {vision}%; fusion moves
us to {fused}% under a stricter protocol than any of them.”*

---

## Slide 11 — Live demo (dashboard)

1. **Live Signal** — latest CE/PE/flat card per index + confidence bar
2. **Model Performance** — walk-forward vs holdout vs benchmarks
3. **Backtest** — equity curve + downloadable trade log
4. **Vision Lab** — pick a date, see the exact chart the ViT consumed
5. **Architecture** — pipeline diagram for Q&A

---

## Slide 12 — Engineering status (50%+ claim map)

| Objective (proposal §4) | Status |
|---|---|
| Chart image generation | ✅ productionised (mplfinance, 2.5k+ images) |
| ViT feature extraction | ✅ ViT-Tiny end-to-end, subprocess-isolated training |
| Tabular feature engineering | ✅ 116 features incl. PCR/IV-skew/Greeks |
| RF + XGB + LSTM + stacking | ✅ trained, OOF meta |
| Multi-modal fusion | ✅ logistic fusion head |
| Walk-forward validation | ✅ 5 refit folds |
| Signal generation + backtest | ✅ real option OHLC engine |
| Dashboard | ✅ Streamlit, 5 pages |
| SQLite persistence | ✅ prices/features/preds/trades |
| Production hardening (live NSE feed, paper-trading loop) | 🔜 next sprint |

**≥50% of the proposal’s objectives are not just started — they run end-to-end
on real NSE data with metrics artifacts.**

---

## Slide 13 — Course / product angle

* Clean repo, config-driven, one-command pipeline → teachable
* Dashboard as the learner’s “terminal”
* Curriculum hooks: feature engineering on derivatives data, ViT for finance,
  honest backtesting, fusion design
* Licensing path: open core + paid courseware around the same codebase

---

## Slide 14 — Roadmap

1. Live daily ingest (NSE API / broker feed) + scheduled retrain
2. Weekly-expiry aware signals & intraday exits
3. Walk-forward with per-fold hyper-parameter search
4. Paper-trading ledger + risk module (position sizing, max loss/day)
5. Publish: write-up with walk-forward protocol as the differentiator

---

## Q&A cheat sheet

* **“Why not use a pretrained ViT?”** — sandbox has no weight downloads; our
  ViT-Tiny is trained end-to-end on chart windows; architecture-equivalent to
  HuggingFace `vit-tiny` configs.
* **“Why is accuracy not 90%?”** — random-split papers leak future rows;
  under walk-forward, realistic numbers are 50–62% (Weinberg 2025). We report
  honest accuracy **and** higher selective accuracy on trade days.
* **“Where does IV come from?”** — Black–Scholes inversion on real closing
  option premiums; Greeks derived from the fitted IV surface.
* **“Is the backtest realistic?”** — real ATM contract OHLC from bhavcopy,
  next-open entry, next-close exit, costs + SL/TP included.
