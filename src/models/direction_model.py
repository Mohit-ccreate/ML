"""
Direction Classifier for OptionEdge
Predicts next-day Nifty direction: Up (1) / Down (-1) / Sideways (0)
or Binary: Up vs Not Up

Models: RandomForest → XGBoost → LightGBM benchmarking
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except: HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except: HAS_LGB = False

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    'Return','Range','Body','Dist_SMA_5','Dist_SMA_20','Dist_SMA_50',
    'Vol_20','ATR_14_true','RSI_14','RSI_7','MACD','MACD_hist','BB_width','BB_pct',
    'Vol_ratio','Return_lag_1','Return_lag_2','Return_lag_3','Return_lag_5',
    'RSI_lag_1','RSI_lag_2'
]

def load_data(symbol="NIFTY", days=365):
    path = PROC_DIR / f"{symbol.upper()}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}. Run python src/data/features.py --symbol {symbol}")
    df = pd.read_csv(path)
    # Keep only cols that exist
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0)
    # Use 3-class target
    y = df['Target']  # -1,0,1
    # Alternative: y_binary = df['Target_binary']
    return X, y, cols, df

def train(symbol="NIFTY", model_type="xgboost"):
    X, y, cols, df = load_data(symbol)
    print(f"Data: {X.shape} | Classes: {y.value_counts().to_dict()}")
    print(f"Features: {cols}")
    
    # Time-series split: no shuffling! train on past, test on future
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | Split date: {df.iloc[split_idx]['Date']}")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Model selection
    if model_type == "randomforest":
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train)  # RF doesn't need scaling
        y_pred = model.predict(X_test)
    elif model_type == "xgboost" and HAS_XGB:
        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
    elif model_type == "lightgbm" and HAS_LGB:
        model = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
    else:
        print(f"Model {model_type} not available, falling back to RandomForest")
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {model_type.upper()} Accuracy: {acc:.3f} ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix (rows=actual, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({'feature': cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        print("\nTop 10 Features:")
        print(imp.head(10).to_string(index=False))
    
    # Save
    joblib.dump(model, MODEL_DIR / f"{symbol}_{model_type}_direction.pkl")
    joblib.dump(scaler, MODEL_DIR / f"{symbol}_scaler.pkl")
    joblib.dump(cols, MODEL_DIR / f"{symbol}_feature_cols.pkl")
    print(f"\nSaved model to models/{symbol}_{model_type}_direction.pkl")
    
    # Demo: predict next day
    last_row = X.iloc[[-1]]
    if model_type in ["xgboost","lightgbm"]:
        last_row_s = scaler.transform(last_row)
        pred = model.predict(last_row_s)[0]
        prob = model.predict_proba(last_row_s)[0].max() if hasattr(model, 'predict_proba') else 0
    else:
        pred = model.predict(last_row)[0]
        prob = model.predict_proba(last_row)[0].max() if hasattr(model, 'predict_proba') else 0
    
    label_map = {-1: "DOWN 🔴", 0: "SIDEWAYS 🟡", 1: "UP 🟢"}
    print(f"\n>>> NEXT DAY PREDICTION: {label_map.get(pred, pred)} | Confidence: {prob:.2%}")
    print(f"Last Close: {df.iloc[-1]['Close']:.1f} | RSI: {df.iloc[-1]['RSI_14']:.1f}")
    print(f"Signal for OptionEdge: {'Buy CE / Bull Spread' if pred==1 else 'Buy PE / Bear Spread' if pred==-1 else 'Iron Condor / Straddle'}")
    
    return model, acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--model", default="xgboost", choices=["randomforest","xgboost","lightgbm"])
    args = parser.parse_args()
    train(args.symbol, args.model)
