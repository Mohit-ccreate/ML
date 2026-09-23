"""Ingest NSE bhavcopy archives (index + F&O) into clean tables.

Data source: noswear94/quantdata GitHub mirror of NSE daily bhavcopy files.
Handles both the legacy F&O format (2019-2023) and the NSE UDIFF format (2024+).
"""
from __future__ import annotations

import glob
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SYM_TO_INDEX_ROW = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank"}

# ---------------------------------------------------------------------------
# Index prices (OHLCV + valuation) + India VIX
# ---------------------------------------------------------------------------

def load_index_prices(raw_root: str | Path, start: str, end: str) -> pd.DataFrame:
    """Return long-format DataFrame with columns:
    date, symbol, open, high, low, close, volume, turnover, pe, pb, div_yield
    for symbols NIFTY / BANKNIFTY plus vix_close attached later.
    """
    files = sorted(glob.glob(str(Path(raw_root) / "index" / "*" / "ind_close_all_*.csv")))
    if not files:
        raise FileNotFoundError(f"No ind_close_all files under {raw_root}/index")
    rows = []
    want = set(SYM_TO_INDEX_ROW.values()) | {"India VIX"}
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            continue
        df = df[df["Index Name"].isin(want)]
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["Index Date"], format="%d-%m-%Y", errors="coerce")
        df = df.dropna(subset=["date"])
        for col in ["Open Index Value", "High Index Value", "Low Index Value",
                    "Closing Index Value", "Volume", "Turnover (Rs. Cr.)",
                    "P/E", "P/B", "Div Yield"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        rows.append(df)
    raw = pd.concat(rows, ignore_index=True)
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]

    # Build per-index frames then assemble
    inv_row = {v: k for k, v in SYM_TO_INDEX_ROW.items()}
    out_frames = []
    for _, r in raw.iterrows():
        name = r["Index Name"]
        if name == "India VIX":
            continue
        sym = inv_row.get(name)
        if not sym:
            continue
        out_frames.append({
            "date": r["date"], "symbol": sym,
            "open": r["Open Index Value"], "high": r["High Index Value"],
            "low": r["Low Index Value"], "close": r["Closing Index Value"],
            "volume": r["Volume"], "turnover": r["Turnover (Rs. Cr.)"],
            "pe": r["P/E"], "pb": r["P/B"], "div_yield": r["Div Yield"],
        })
    prices = pd.DataFrame(out_frames)
    prices = prices.drop_duplicates(subset=["date", "symbol"]).sort_values(["symbol", "date"])
    prices = prices.reset_index(drop=True)

    vix = raw[raw["Index Name"] == "India VIX"][["date", "Closing Index Value"]].copy()
    vix["vix_close"] = vix["Closing Index Value"]
    vix = vix.drop(columns=["Closing Index Value"]).drop_duplicates("date")
    prices = prices.merge(vix, on="date", how="left")
    prices["vix_close"] = prices.groupby("symbol")["vix_close"].ffill()
    return prices


# ---------------------------------------------------------------------------
# F&O bhavcopy — normalise legacy + UDIFF formats
# ---------------------------------------------------------------------------

def _read_zip_csv(fp: str) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(fp) as z:
            name = z.namelist()[0]
            with z.open(name) as fh:
                return pd.read_csv(fh, low_memory=False)
    except Exception:
        return None


def _normalise_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy header:
    INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,
    SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP
    """
    df = df.copy()
    df = df[df["SYMBOL"].isin(SYM_TO_INDEX_ROW)].copy()
    df["instrument"] = df["INSTRUMENT"].map(
        {"OPTIDX": "IDX_OPT", "FUTIDX": "IDX_FUT",
         "OPTSTK": "STK_OPT", "FUTSTK": "STK_FUT"}).fillna(df["INSTRUMENT"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y", errors="coerce"),
        "symbol": df["SYMBOL"],
        "instrument": df["instrument"],
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce"),
        "strike": pd.to_numeric(df["STRIKE_PR"], errors="coerce"),
        "option_type": df["OPTION_TYP"].replace({"XX": None}),
        "open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH"], errors="coerce"),
        "low": pd.to_numeric(df["LOW"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "settle": pd.to_numeric(df["SETTLE_PR"], errors="coerce"),
        "contracts": pd.to_numeric(df["CONTRACTS"], errors="coerce"),
        "turnover": pd.to_numeric(df["VAL_INLAKH"], errors="coerce"),
        "oi": pd.to_numeric(df["OPEN_INT"], errors="coerce"),
        "chg_oi": pd.to_numeric(df["CHG_IN_OI"], errors="coerce"),
        "underlying": np.nan,
    })
    return out


def _normalise_udiff(df: pd.DataFrame) -> pd.DataFrame:
    """UDIFF header (2024+):
    TradDt,...,FinInstrmTp,...,TckrSymb,...,XpryDt,...,StrkPric,OptnTp,...,
    OpnPric,HghPric,LwPric,ClsPric,...,UndrlygPric,SttlmPric,OpnIntrst,
    ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,...
    IDO = index options, IDF = index futures.
    """
    df = df.copy()
    df = df[df["TckrSymb"].isin(SYM_TO_INDEX_ROW)].copy()
    df = df[df["FinInstrmTp"].isin(["IDO", "IDF"])].copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"], errors="coerce"),
        "symbol": df["TckrSymb"],
        "instrument": df["FinInstrmTp"].map({"IDO": "IDX_OPT", "IDF": "IDX_FUT"}),
        "expiry": pd.to_datetime(df["XpryDt"], errors="coerce"),
        "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
        "option_type": df["OptnTp"].replace({"": None}),
        "open": pd.to_numeric(df["OpnPric"], errors="coerce"),
        "high": pd.to_numeric(df["HghPric"], errors="coerce"),
        "low": pd.to_numeric(df["LwPric"], errors="coerce"),
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "settle": pd.to_numeric(df["SttlmPric"], errors="coerce"),
        "contracts": np.nan,
        # TtlTrfVal is in ₹ lakh in NSE UDIFF files
        "turnover": pd.to_numeric(df["TtlTrfVal"], errors="coerce"),
        "oi": pd.to_numeric(df["OpnIntrst"], errors="coerce"),
        "chg_oi": pd.to_numeric(df["ChngInOpnIntrst"], errors="coerce"),
        "underlying": pd.to_numeric(df["UndrlygPric"], errors="coerce"),
    })
    return out


def load_fo_chain(raw_root: str | Path, start: str, end: str,
                  symbols: list[str] | None = None) -> pd.DataFrame:
    """Load NIFTY/BANKNIFTY index options + futures from all F&O bhavcopy zips."""
    files = sorted(glob.glob(str(Path(raw_root) / "fo" / "*" / "*.zip")))
    if not files:
        raise FileNotFoundError(f"No F&O bhavcopy zips under {raw_root}/fo")
    parts = []
    for fp in files:
        df = _read_zip_csv(fp)
        if df is None or df.empty:
            continue
        cols = set(df.columns)
        if "INSTRUMENT" in cols:
            norm = _normalise_legacy(df)
        elif "FinInstrmTp" in cols:
            norm = _normalise_udiff(df)
        else:
            continue
        norm = norm.dropna(subset=["date"])
        parts.append(norm)
    chain = pd.concat(parts, ignore_index=True)
    chain = chain[(chain["date"] >= start) & (chain["date"] <= end)]
    if symbols:
        chain = chain[chain["symbol"].isin(symbols)]
    chain = chain.sort_values(["symbol", "date", "expiry", "strike"]).reset_index(drop=True)
    return chain


def load_prices_yfinance(symbols: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Fallback loader for environments with internet access (Yahoo Finance).

    symbols maps internal ids ('nifty'/'banknifty') to Yahoo tickers.
    """
    import yfinance as yf
    frames = []
    for sid, ticker in symbols.items():
        sym = "NIFTY" if sid == "nifty" else "BANKNIFTY" if sid == "banknifty" else sid
        df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                         progress=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"})
        df["symbol"] = sym
        df["turnover"] = np.nan
        df["pe"] = np.nan
        df["pb"] = np.nan
        df["div_yield"] = np.nan
        df["vix_close"] = np.nan
        frames.append(df[["date", "symbol", "open", "high", "low", "close",
                          "volume", "turnover", "pe", "pb", "div_yield",
                          "vix_close"]])
    if not frames:
        raise RuntimeError("yfinance returned no data")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def ingest(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full ingest → (prices, chain) dataframes; also writes processed parquet files."""
    d = cfg["data"]
    proc = Path(d["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    prices = load_index_prices(d["raw_root"], d["start_date"], d["end_date"])
    chain = load_fo_chain(d["raw_root"], d["start_date"], d["end_date"],
                          symbols=list(d["symbols"].values()))
    prices.to_parquet(proc / "prices.parquet", index=False)
    chain.to_parquet(proc / "fo_chain.parquet", index=False)
    return prices, chain


if __name__ == "__main__":
    from optionedge.config import load_config, ensure_dirs
    cfg = load_config()
    ensure_dirs(cfg)
    p, c = ingest(cfg)
    print("prices", p.shape, "chain", c.shape)
    print(p.head())
    print(c.head())
