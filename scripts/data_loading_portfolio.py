"""
data_loading_portfolio.py

Robust 10-year OHLCV downloader for top US stocks + SPY benchmark.
- Ticker normalization (handles BRK-B and common quirks)
- Retry logic with exponential backoff
- Conservative concurrency to avoid throttling
- Uses yfinance download for reliable Adj Close
- Graceful error handling, clean CSV output

Outputs: data/portfolio_data.csv
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


# ------------------------------
# Configuration
# ------------------------------

# Top 50 US stocks by market cap + SPY benchmark
TOP_TICKERS = [
    # Mega-cap Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",

    # Mega-cap Finance & Berkshire
    "BRK-B", "JPM", "V", "MA", "BAC", "GS",

    # Healthcare & Pharma
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "ABT", "TMO", "AMGN",

    # Energy & Materials
    "XOM", "CVX", "COP", "MPC", "PSX",

    # Consumer & Retail
    "PG", "KO", "HD", "WMT", "MCD", "NKE", "COST", "TJX", "SBUX", "AZO",

    # Industrial & Defense
    "CAT", "BA", "HON", "LMT", "RTX", "GE",

    # Semiconductors & Tech Services
    "AVGO", "TXN", "QCOM", "ADBE", "CRM", "NOW", "INTU", "IBM", "ORCL", "ACN",

    # Utilities & Infrastructure
    "NEE", "DUK", "SO", "D", "LIN",

    # Communications
    "VZ", "T", "DIS",

    # Benchmark
    "SPY"
]

# Ticker normalization map and alternates to try if a symbol fails
TICKER_ALIASES: Dict[str, List[str]] = {
    # Known Yahoo quirks: sometimes hyphen vs dot
    "BRK-B": ["BRK-B", "BRK.B"],
    # Keep primary for others; add alternates if you find resolution issues
    "GOOGL": ["GOOGL"],
    "META": ["META"],      # formerly FB
    "SPY": ["SPY"],        # S&P 500 ETF
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "portfolio_data.csv")

# Concurrency and retries
MAX_WORKERS = 5            # conservative to avoid throttling
RETRIES = 3
BACKOFF_BASE_SECONDS = 2   # exponential backoff: 2, 4, 8


# ------------------------------
# Helpers
# ------------------------------

def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def get_date_window_days(years: int = 10) -> int:
    # Add a small cushion to be safe across leap years
    return years * 365 + 30


def fmt_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def candidate_symbols(ticker: str) -> List[str]:
    # Return list of candidates to try for a given ticker
    if ticker in TICKER_ALIASES:
        return TICKER_ALIASES[ticker]
    return [ticker]


def normalize_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # Ensure standard columns and add symbol, with adjusted_close present
    df = df.reset_index()
    # yfinance.download returns: Date, Open, High, Low, Close, Adj Close, Volume
    cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        # If Adj Close missing (rare with download), fallback to Close
        if "Adj Close" in missing and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
            missing.remove("Adj Close")
        # If still missing essentials, return empty to signal failure
        essential = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if any(m in essential for m in missing):
            return pd.DataFrame()

    df["symbol"] = symbol
    df = df[["Date", "symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.columns = ["date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume"]

    # Sort by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


# ------------------------------
# Core download logic
# ------------------------------

def fetch_single_symbol(symbol: str, start: str, end: str, retries: int = RETRIES) -> pd.DataFrame:
    """
    Fetch OHLCV for one concrete symbol using yfinance.download with retries.
    Returns normalized dataframe or empty on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                threads=False,  # single ticker; avoid internal threads
                timeout=30,
                interval="1d"
            )
            if df is not None and not df.empty:
                return normalize_df(df, symbol)
            # Empty response: backoff and retry
            time.sleep(BACKOFF_BASE_SECONDS ** attempt)
        except Exception as e:
            # Backoff and retry
            time.sleep(BACKOFF_BASE_SECONDS ** attempt)
    return pd.DataFrame()


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Try multiple candidate symbols for a ticker (aliases), with retries per candidate.
    Returns first successful normalized dataframe or empty.
    """
    candidates = candidate_symbols(ticker)
    for i, sym in enumerate(candidates, start=1):
        df = fetch_single_symbol(sym, start, end, retries=RETRIES)
        if not df.empty:
            # If the alias differs, set original ticker in the symbol column for consistency
            df["symbol"] = ticker
            print(f"✅ {ticker}: {len(df)} rows downloaded (resolved as '{sym}')")
            return df
        else:
            print(f"⚠️ {ticker}: No data for candidate '{sym}' (attempt {i}/{len(candidates)})")
    print(f"❌ {ticker}: Failed after trying {len(candidates)} candidate symbol(s)")
    return pd.DataFrame()


def main(tickers: Optional[List[str]] = None, max_symbols: int = 50) -> pd.DataFrame:
    ensure_dirs()

    tickers = tickers or TOP_TICKERS
    tickers = tickers[:max_symbols]

    # Date window: last 10 years to today (most recent trading date will be handled by yfinance)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=get_date_window_days(10))
    start_str, end_str = fmt_date(start_dt), fmt_date(end_dt)

    print("\n" + "=" * 70)
    print("📊 PORTFOLIO DATA LOADER")
    print("=" * 70)
    print(f"📥 Downloading 10-year OHLCV data for {len(tickers)} tickers...")
    print(f"⏱️  Start: {start_str}")
    print(f"⏱️  End:   {end_str}")
    print("=" * 70 + "\n")

    dfs: List[pd.DataFrame] = []
    completed = 0

    # Conservative concurrency to avoid throttling
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_ticker, t, start_str, end_str): t for t in tickers}
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                if not result.empty:
                    dfs.append(result)
            except Exception as e:
                t = futures[future]
                print(f"❌ {t}: Unhandled exception - {str(e)[:80]}")
            if completed % 10 == 0 or completed == len(tickers):
                print(f"   Progress: {completed}/{len(tickers)} tickers completed")

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

        df.to_csv(CSV_PATH, index=False)

        print("\n" + "=" * 70)
        print("✅ DATA DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"📊 Rows: {len(df):,}")
        print(f"📊 Symbols: {df['symbol'].nunique()}")
        print(f"📊 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"📁 Saved: {CSV_PATH}")
        print("=" * 70 + "\n")
        return df
    else:
        print("\n❌ No data downloaded. Check ticker symbols, yfinance installation, and internet connection.")
        return pd.DataFrame()


def load_portfolio_data() -> pd.DataFrame:
    """
    Load portfolio data from CSV if present; otherwise download fresh.
    """
    ensure_dirs()
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        print(f"📂 Loaded: {CSV_PATH} ({len(df):,} rows, {df['symbol'].nunique()} symbols)")
        return df
    else:
        print("📥 CSV not found, downloading...")
        return main(TOP_TICKERS, max_symbols=min(50, len(TOP_TICKERS)))


if __name__ == "__main__":
    # Optional: quick environment sanity check
    print(f"🐍 Python: {sys.version.split()[0]} | yfinance: {getattr(yf, '__version__', 'unknown')}")
    df = main(TOP_TICKERS, max_symbols=min(50, len(TOP_TICKERS)))
    if not df.empty:
        print("\n📈 Sample data (first 5 rows):")
        print(df.head())
        print("\n📊 Data Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Symbols: {sorted(df['symbol'].unique().tolist())}")
