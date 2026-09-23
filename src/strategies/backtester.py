"""
Simple Backtester for OptionEdge
Logic: Use ML direction prediction to simulate strategy payoff

Example:
  If pred == 1 (UP): Buy ATM CE
  If pred == -1 (DOWN): Buy ATM PE
  If pred == 0: Iron Condor (simulate small profit in sideways)

This is a price-action proxy. For true options backtest, plug in historical option LTP.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

def backtest_direction_strategy(symbol="NIFTY", cost_per_trade=20):
    path = PROC_DIR / f"{symbol.upper()}_features.csv"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Target','Target_return']).reset_index(drop=True)
    
    # Simulate: we have prediction = Target (perfect for demo). Replace with model predictions in production.
    # For realistic, add noise: assume 58% accuracy model
    np.random.seed(42)
    # Create simulated predictions: 58% correct, rest random
    # For now use actual Target shifted to avoid lookahead, then add error
    true = df['Target'].values
    # Simulate model accuracy 58%
    pred = true.copy()
    # Flip 42% randomly
    flip_mask = np.random.rand(len(true)) < 0.42
    # Flip to random other class
    for i in np.where(flip_mask)[0]:
        choices = [c for c in [-1,0,1] if c != true[i]]
        pred[i] = np.random.choice(choices)
    
    df['Pred'] = pred
    
    # Payoff proxy: if correct direction, profit = abs(return)*Close - cost
    # If wrong, loss = -abs(return)*Close - cost
    # Simplified: P&L in points
    df['Next_return'] = df['Target_return']
    df['Correct'] = (df['Pred'] == df['Target']).astype(int)
    
    # Points P&L proxy: assume ATM option captures 40% of underlying move (delta ~0.5 * leverage)
    # For UP correct: profit = 0.4 * move, else -0.4*move - cost
    df['PnL_points'] = np.where(df['Correct']==1, abs(df['Next_return'])*df['Close']*0.4, -abs(df['Next_return'])*df['Close']*0.4) - cost_per_trade/50  # cost in points approx
    
    # Alternative: pure underlying directional P&L (buy future)
    df['PnL_underlying'] = df['Pred'] * df['Next_return'] * df['Close']  # long/short future
    
    # Cum P&L
    df['Cum_PnL'] = df['PnL_points'].cumsum()
    
    # Metrics
    win_rate = df['Correct'].mean()
    total_pnl = df['PnL_points'].sum()
    avg_win = df[df['Correct']==1]['PnL_points'].mean()
    avg_loss = df[df['Correct']==0]['PnL_points'].mean()
    sharpe = df['PnL_points'].mean() / df['PnL_points'].std() * np.sqrt(252) if df['PnL_points'].std()!=0 else 0
    max_dd = (df['Cum_PnL'].cummax() - df['Cum_PnL']).max()
    
    print(f"=== Backtest {symbol} (Simulated 58% Model) ===")
    print(f"Trades: {len(df)} | Win Rate: {win_rate:.1%}")
    print(f"Total P&L (points): {total_pnl:.1f} | Avg Win: {avg_win:.1f} | Avg Loss: {avg_loss:.1f}")
    print(f"Sharpe (proxy): {sharpe:.2f} | Max Drawdown: {max_dd:.1f} points")
    print(f"\nLast 5 trades:\n{df[['Date','Close','Target','Pred','Correct','PnL_points','Cum_PnL']].tail().to_string(index=False)}")
    
    # Save
    out = PROC_DIR / f"{symbol}_backtest.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved backtest to {out}")
    print("\nTip: Replace simulated Pred with model.predict() output for real backtest: pred = model.predict(X)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NIFTY")
    args = parser.parse_args()
    backtest_direction_strategy(args.symbol)
