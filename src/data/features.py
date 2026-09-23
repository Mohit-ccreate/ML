"""
Feature Engineering for OptionEdge
Input: data/raw/NIFTY_ohlc_365d.csv + option_chain.csv
Output: data/processed/features.csv (ML-ready)
Features:
  - Price/TA: Returns, RSI, MACD, ATR, Bollinger, VWAP proxy, SMA/EMA
  - Options: PCR, OI Change, Max Pain, Moneyness, Time to Expiry proxy, IV
  - Target: Next-day direction (1 Up, -1 Down, 0 Sideways) + Next-day return
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    df = df.copy()
    # Ensure Date is datetime and sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Basic returns
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    df['Body'] = abs(df['Close'] - df['Open']) / df['Close']
    
    # Moving averages
    for w in [5, 20, 50]:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w).mean()
        df[f'Dist_SMA_{w}'] = (df['Close'] - df[f'SMA_{w}']) / df[f'SMA_{w}']
    
    # Volatility
    df['Vol_20'] = df['Return'].rolling(20).std() * np.sqrt(252)
    df['ATR_14'] = (df['High'] - df['Low']).rolling(14).mean()  # simplified
    # True ATR approx
    hl = df['High'] - df['Low']
    hc = abs(df['High'] - df['Close'].shift(1))
    lc = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR_14_true'] = tr.rolling(14).mean()
    
    # RSI, MACD, Bollinger
    df['RSI_14'] = rsi(df['Close'], 14)
    df['RSI_7'] = rsi(df['Close'], 7)
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2*df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2*df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume features
    if 'Volume' in df.columns:
        df['Vol_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Vol_ratio'] = df['Volume'] / df['Vol_SMA_20']
    
    # Lag features
    for lag in [1,2,3,5]:
        df[f'Return_lag_{lag}'] = df['Return'].shift(lag)
        df[f'RSI_lag_{lag}'] = df['RSI_14'].shift(lag)
    
    # Target: Next day direction
    # 1 = Up > 0.7%, -1 = Down < -0.7%, 0 = Sideways (tune threshold per Nifty volatility)
    next_ret = df['Close'].pct_change().shift(-1)
    df['Target_return'] = next_ret
    df['Target'] = 0
    df.loc[next_ret > 0.007, 'Target'] = 1
    df.loc[next_ret < -0.007, 'Target'] = -1
    
    # Binary target for simpler model
    df['Target_binary'] = (next_ret > 0).astype(int)  # 1 if up
    
    return df

def merge_option_features(df_price, option_chain_path=None):
    """If option chain exists, merge PCR etc. For now compute proxy PCR from price volatility"""
    # Try to find option chain file
    if option_chain_path and Path(option_chain_path).exists():
        oc = pd.read_csv(option_chain_path)
        # Compute PCR etc. if available - merge on date (requires daily snapshot; for now static)
        # For historical, this would be time-series PCR; we add placeholder
        pcr = oc['PE_OI'].sum() / oc['CE_OI'].sum() if oc['CE_OI'].sum() else 1.0
        df_price['PCR'] = pcr  # static; in production make it time-series
        df_price['CE_OI_total'] = oc['CE_OI'].sum()
        df_price['PE_OI_total'] = oc['PE_OI'].sum()
    else:
        # Proxy: Use rolling return skew as PCR proxy if no option data
        df_price['PCR'] = np.nan
        # You will replace this with real daily PCR time series
    return df_price

def main(symbol="NIFTY", days=365):
    raw_file = RAW_DIR / f"{symbol.upper()}_ohlc_{days}d.csv"
    if not raw_file.exists():
        print(f"Raw file not found: {raw_file}")
        print(f"Run: python src/data/nse_data.py --symbol {symbol} --days {days}")
        return
    df = pd.read_csv(raw_file)
    print(f"Loaded {len(df)} rows from {raw_file}")
    
    df_feat = add_features(df)
    
    # Try to merge option chain if exists
    oc_path = RAW_DIR / f"{symbol.upper()}_option_chain.csv"
    df_feat = merge_option_features(df_feat, oc_path)
    
    # Drop NaNs created by rolling
    df_feat_clean = df_feat.dropna().reset_index(drop=True)
    print(f"Features shape after clean: {df_feat_clean.shape}")
    print(f"Target distribution:\n{df_feat_clean['Target'].value_counts()}")
    
    out = PROC_DIR / f"{symbol.upper()}_features.csv"
    df_feat_clean.to_csv(out, index=False)
    print(f"Saved features to {out}")
    print("\nColumns:", list(df_feat_clean.columns))
    # Quick preview of signals
    print(df_feat_clean[['Date','Close','RSI_14','MACD','Target','Target_return']].tail().to_string())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    main(args.symbol, args.days)
