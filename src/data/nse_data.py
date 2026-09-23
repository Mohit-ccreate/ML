"""
NSE Data Fetcher for OptionEdge
Fetches: OHLC (yfinance) + Option Chain (nsepython)
Usage:
  python src/data/nse_data.py --symbol NIFTY --days 365
  python src/data/nse_data.py --symbol BANKNIFTY --days 180 --option-chain
"""
import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys

# Symbol mapping yfinance
SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "^CNXFIN",
    "SENSEX": "^BSESN",
    "RELIANCE": "RELIANCE.NS",
}

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_ohlc(symbol="NIFTY", days=365, interval="1d"):
    ticker = SYMBOL_MAP.get(symbol.upper(), symbol)
    print(f"Fetching {symbol} ({ticker}) last {days} days...")
    df = yf.download(ticker, period=f"{days}d", interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        print(f"Failed to fetch {ticker}. Check symbol or network.")
        return pd.DataFrame()
    # Flatten multi-index if present (yfinance new version)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    df.reset_index(inplace=True)
    # Save
    out = RAW_DIR / f"{symbol.upper()}_ohlc_{days}d.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df.tail(3).to_string())
    return df

def fetch_option_chain(symbol="NIFTY"):
    """Fetch live option chain via nsepython (requires internet, NSE may block)"""
    try:
        from nsepython import nse_optionchain_scrapper
    except ImportError:
        print("nsepython not installed. Run pip install nsepython")
        return None
    try:
        print(f"Fetching option chain for {symbol}...")
        chain = nse_optionchain_scrapper(symbol)
        # chain has 'records' -> 'data' list
        records = chain.get('records', {}).get('data', [])
        rows = []
        for r in records:
            strike = r.get('strikePrice')
            ce = r.get('CE', {})
            pe = r.get('PE', {})
            rows.append({
                'strike': strike,
                'expiry': r.get('expiryDate'),
                'CE_OI': ce.get('openInterest'),
                'CE_OI_change': ce.get('changeinOpenInterest'),
                'CE_LTP': ce.get('lastPrice'),
                'CE_volume': ce.get('totalTradedVolume'),
                'CE_IV': ce.get('impliedVolatility'),
                'CE_bidQty': ce.get('bidQty'),
                'PE_OI': pe.get('openInterest'),
                'PE_OI_change': pe.get('changeinOpenInterest'),
                'PE_LTP': pe.get('lastPrice'),
                'PE_volume': pe.get('totalTradedVolume'),
                'PE_IV': pe.get('impliedVolatility'),
            })
        df = pd.DataFrame(rows)
        out = RAW_DIR / f"{symbol.upper()}_option_chain.csv"
        df.to_csv(out, index=False)
        print(f"Saved option chain {len(df)} strikes to {out}")
        # Compute PCR
        total_ce_oi = df['CE_OI'].sum()
        total_pe_oi = df['PE_OI'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi else 0
        print(f"PCR (OI): {pcr:.3f} | Total CE OI: {total_ce_oi} | PE OI: {total_pe_oi}")
        return df
    except Exception as e:
        print(f"Option chain fetch failed: {e}")
        print("Tip: NSE blocks cloud IPs. Try running locally or use Dhan/TrueData API for production.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--option-chain", action="store_true", help="Also fetch option chain")
    args = parser.parse_args()
    
    df = fetch_ohlc(args.symbol, args.days, args.interval)
    if args.option_chain:
        fetch_option_chain(args.symbol)
    
    print("\nNext: python src/data/features.py")
