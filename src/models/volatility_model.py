"""
Volatility / IV Predictor for OptionEdge
Target: Predict next-day realized volatility (or IV if available)
Use case: "IV expansion expected -> Buy Straddle, IV crush -> Sell"
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except: HAS_XGB = False

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ['Return','Range','Vol_20','ATR_14_true','RSI_14','BB_width','Return_lag_1','Return_lag_2','Return_lag_3']

def train_volatility(symbol="NIFTY"):
    path = PROC_DIR / f"{symbol.upper()}_features.csv"
    df = pd.read_csv(path)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    # Target: next day's Vol_20
    df['Target_vol'] = df['Vol_20'].shift(-1)
    df_clean = df.dropna(subset=cols + ['Target_vol'])
    X = df_clean[cols]
    y = df_clean['Target_vol']
    
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    if HAS_XGB:
        model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"Volatility Predictor — MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    print(f"Actual vol last 5: {y_test.tail().values.round(4)}")
    print(f"Pred   last 5: {pred[-5:].round(4)}")
    
    # Direction of vol change
    vol_up_actual = (y_test.values > X_test['Vol_20'].values).mean()
    vol_up_pred = (pred > X_test['Vol_20'].values).mean()
    print(f"Actual vol expansion rate: {vol_up_actual:.1%} | Predicted: {vol_up_pred:.1%}")
    
    joblib.dump(model, MODEL_DIR / f"{symbol}_volatility.pkl")
    joblib.dump(scaler, MODEL_DIR / f"{symbol}_vol_scaler.pkl")
    
    # Next day forecast
    last = scaler.transform(X.iloc[[-1]])
    next_vol = model.predict(last)[0]
    curr_vol = df_clean.iloc[-1]['Vol_20']
    print(f"\n>>> NEXT DAY VOL FORECAST: {next_vol:.2%} vs Current {curr_vol:.2%} | {'Expansion ↑ -> Long Straddle' if next_vol > curr_vol else 'Crush ↓ -> Short Straddle'}")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NIFTY")
    args = parser.parse_args()
    train_volatility(args.symbol)
