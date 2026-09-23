# OptionEdge ML Roadmap — 30-60-90 Days

## Vision
Make OptionEdge the #1 AI-powered options platform for India (Nifty/BankNifty): Real-time OI + AI + Alerts.

---

### Phase 1: Days 1-30 — Foundation & First Model

**Goal: Ship 1 working predictive alert**

- [ ] **Week 1: Data Pipeline**
  - Fetch Nifty/BankNifty OHLC (yfinance / NSE)
  - Fetch Option Chain (nsepython / NSE API) — Strike, OI, Volume, LTP
  - Calculate Greeks using `src/utils/greeks.py` (Black-Scholes)
  - Store daily in `data/raw/`

- [ ] **Week 2: Feature Engineering**
  - Technical: RSI, MACD, ATR, Bollinger, Supertrend, VWAP
  - Options: PCR (OI PCR, Volume PCR), OI Change %, Max Pain, IV, Moneyness, Time to Expiry
  - Label: Next-day direction (1 = Up >0.5%, -1 = Down <-0.5%, 0 = Sideways)

- [ ] **Week 3: Model 1 — Direction Classifier**
  - Baseline: Logistic Regression, Random Forest
  - Production: XGBoost / LightGBM
  - Metrics: Accuracy, Precision/Recall, F1, Confusion Matrix
  - Notebook: `notebooks/03_direction_prediction.ipynb`

- [ ] **Week 4: Backtest**
  - `src/strategies/backtester.py` — If model says UP → Buy CE, else PE
  - Metrics: Win Rate, Avg P&L, Sharpe, Max Drawdown
  - Telegram mock alert: "Nifty UP 72% confidence — Consider Bull Call Spread"

**Deliverable:** Working Jupyter notebook + backtest report

---

### Phase 2: Days 31-60 — Volatility & Strategy Engine

**Goal: Add 2 more AI features to OptionEdge**

- [ ] **IV Predictor**
  - Target: Predict next-day IV (regression)
  - Models: XGBoost + LSTM (sequence of 20 days)
  - Use case: "IV expansion expected — prefer Long Straddle"

- [ ] **Strategy Recommender**
  - Input: IV, Direction Confidence, Time to Expiry, VIX
  - Output: Top 3 strategies with success probability
    - e.g., High IV + Low Direction Confidence → Iron Condor (68% win)
    - e.g., Low IV + High Direction Confidence → Long CE/PE
  - Train on historical strategy payoffs

- [ ] **OI Anomaly Detector**
  - Unsupervised: Isolation Forest on OI spikes
  - Alert: "Unusual Call Writing at 25000 CE — Bearish signal"

---

### Phase 3: Days 61-90 — Production & Monetization

**Goal: Live on OptionEdge dashboard**

- [ ] **API Layer:** FastAPI endpoint `/predict` → returns {direction, confidence, strategy}
- [ ] **Dashboard Integration:** Show AI confidence badge on Option Chain
- [ ] **Telegram Live Alerts:** Real-time PCR shift + AI signal
- [ ] **Paper Trading:** Auto-forward test for 30 days, track live P&L
- [ ] **Monetization:** Free tier (delayed) vs Pro (real-time AI) — matches optionedge.info pricing

---

### Data Sources (Pick One)

| Source | Cost | Data | Best For |
|--------|------|------|----------|
| **yfinance** | Free | OHLC | Prototyping |
| **NSE India API** (nsepython) | Free | Option Chain, OI | Production (rate limited) |
| **TrueData** | Paid (~₹1500/mo) | Real-time Tick + OI | Real-time AI |
| **Dhan / Angel One API** | Free with account | Live Options | Live Trading |

**Recommendation:** Start with `yfinance + nsepython` (free), migrate to TrueData when monetizing.

---

### Model Priority for OptionEdge

1.  **Direction Classifier** → Immediate value for alerts (START HERE)
2.  IV Predictor → Strategy selection
3.  Strategy Success Predictor → Highest monetization value
4.  OI Anomaly Detector → Unique moat

---

### What To Do TODAY (Next 2 Hours)

```bash
pip install -r requirements.txt
python src/data/nse_data.py --symbol NIFTY --days 365
jupyter notebook notebooks/01_eda.ipynb
# -> See PCR vs Next Day Move plot
```

Then tell me which data source you have, and I'll wire the live pipeline.
