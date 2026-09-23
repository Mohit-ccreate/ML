# ML for OptionEdge — AI-Powered Options Intelligence

> Repo: `Mohit-ccreate/ML` | Branch: `arena/01a0cccb-ml`
> Goal: Build Machine Learning models that power **OptionEdge** — real-time OI analysis, AI strategy builder, and predictive alerts for Nifty / BankNifty.

### What is OptionEdge + ML?
OptionEdge is your options trading intelligence platform. ML makes it **predictive**, not just analytical:
- Predict **Implied Volatility (IV)** and **direction** before the move
- Recommend **strategies** (Straddle, Iron Condor, Spread) with success probability
- Generate **real-time alerts** (Telegram) with confidence scores
- Learn from **OI, PCR, Price Action, and Greeks**

---

### 🚀 What To Do Next — Your 7-Step Roadmap

This repo is ready for you. Here is the exact sequence:

#### Phase 1: Foundation (Week 1) — DO THIS NOW
1.  **Setup Environment:** `pip install -r requirements.txt`
2.  **Get Data:** Run `python src/data/nse_data.py` — fetches Nifty/BankNifty OHLC + Option Chain (NSE/Yahoo)
3.  **Explore Data:** Open `notebooks/01_eda.ipynb` — understand OI, PCR, IV patterns

#### Phase 2: Feature Engineering (Week 2)
4.  **Build Features:** `src/features/build_features.py` creates ML-ready features:
    - Price: OHLC, VWAP, Returns, ATR, RSI, MACD, Bollinger Bands
    - Options specific: OI, OI Change, PCR, IV, Greeks (Delta, Gamma, Theta, Vega), Max Pain
    - Market: VIX, Time to Expiry, Moneyness, Volume

#### Phase 3: Models (Week 3-4)
5.  **Train 3 Core Models:**
    - **A) Volatility Forecaster** (`volatility_model.py`): Predict next-day IV / HV using XGBoost + LSTM
    - **B) Direction Classifier** (`direction_model.py`): Predict Nifty direction (Up/Down/Sideways) — RandomForest + LightGBM
    - **C) Strategy Success Predictor:** Predict P&L probability for Straddle/Condor/Spread

#### Phase 4: Backtest & Deploy (Week 5+)
6.  **Backtest:** `src/strategies/backtester.py` — test strategies on historical data, calculate Sharpe, Win Rate, Max Drawdown
7.  **Deploy:** Connect to OptionEdge dashboard + Telegram alerts

```
Data → Features → Model → Backtest → Alert
 NSE API → OI/PCR/Greeks → XGBoost/LSTM → Payoff Chart → Telegram
```

---

### 📁 Repo Structure

```
ML/
├── data/
│   ├── raw/              # Raw NSE/Bhavcopy data (gitignored)
│   ├── processed/        # Features (parquet)
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_volatility_prediction.ipynb  # IV Forecasting
│   └── 03_direction_prediction.ipynb   # Nifty Direction
├── src/
│   ├── data/
│   │   ├── nse_data.py       # Fetch NSE option chain + OHLC
│   │   └── features.py       # Feature engineering
│   ├── models/
│   │   ├── volatility_model.py
│   │   └── direction_model.py
│   ├── strategies/
│   │   └── backtester.py     # Strategy backtesting + Greeks
│   └── utils/
│       ├── greeks.py         # Black-Scholes Greeks
│       └── helpers.py
├── docs/
│   └── ROADMAP.md        # Detailed 30-60-90 day plan
├── requirements.txt
└── .gitignore
```

### ⚡ Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Mohit-ccreate/ML.git
cd ML
pip install -r requirements.txt

# 2. Fetch data (Nifty example)
python src/data/nse_data.py --symbol NIFTY --days 365

# 3. Build features
python src/data/features.py

# 4. Train volatility model
python src/models/volatility_model.py

# 5. Run notebook
jupyter notebook notebooks/01_eda.ipynb
```

### 🎯 Next Action For YOU

1.  **Pick ONE model to start:** Recommendation → **Direction Classifier** (easiest win for OptionEdge alerts)
2.  Tell me your data source: Do you have **TrueData / NSE API / TradingView / Dhan / Angel One** access? I'll wire it up.
3.  Run the scaffolding I just created and push to GitHub.

Want me to build the first working model end-to-end for you? Just say: *"Build direction model"* or *"Build IV predictor"*

---
*Created: 2026-09-23 | For OptionEdge AI Intelligence*
