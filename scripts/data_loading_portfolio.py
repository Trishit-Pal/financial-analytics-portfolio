"""
data_loading_portfolio.py

Portfolio data loader using yfinance (Yahoo Finance).
- Fetch up to N stocks with ~10 years of daily data
- Output schema:
  date, symbol, open, high, low, close, adjusted_close, volume
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf  # Multi-ticker market data [web:325][web:329]

PROCESSED_DATA_PATH = "data/processed/portfolio_data.csv"
os.makedirs("data/processed", exist_ok=True)

# Up to 40 tickers (you can extend to 50 if desired)
TOP_50_TICKERS = [
    # Finance
    "JPM", "GS", "BLK", "BK", "MS", "PNC", "USB", "COF", "AXP", "CFG",
    # Tech
    "IBM", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "AMD", "ADBE", "CRM",
    # Energy
    "XOM", "CVX", "MPC", "PSX", "VLO",
    # Healthcare
    "UNH", "JNJ", "LLY", "MRK", "ABT", "ABBV", "PFE", "BMY", "AMGN", "GILD",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "AMZN",
    "SPY", 
]


def fetch_yahoo_portfolio(
    symbols=None,
    start: str | None = None,
    end: str | None = None,
    max_symbols: int = 50,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for multiple symbols using yfinance.

    Parameters
    ----------
    symbols : list[str]
        Tickers to load. Default: TOP_50_TICKERS (truncated to max_symbols).
    start : str
        Start date (YYYY-MM-DD). Default: 10 years ago from today.
    end : str
        End date (YYYY-MM-DD). Default: today.
    max_symbols : int
        Limit number of tickers to fetch.

    Returns
    -------
    DataFrame with columns:
    - date, open, high, low, close, adjusted_close, volume, symbol
    """
    if symbols is None:
        symbols = TOP_50_TICKERS[:max_symbols]

    if end is None:
        end_dt = datetime.today()
    else:
        end_dt = datetime.strptime(end, "%Y-%m-%d")

    if start is None:
        start_dt = end_dt - timedelta(days=365 * 10)
    else:
        start_dt = datetime.strptime(start, "%Y-%m-%d")

    print(f"\n=== FETCHING PORTFOLIO DATA VIA YFINANCE ===")
    print(f"Symbols: {len(symbols)} | From {start_dt.date()} to {end_dt.date()}")

    # Download once for all tickers; full OHLCV including Adj Close [web:344][web:356]
    data = yf.download(
        symbols,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=False,      # keep both Close and Adj Close [web:347][web:351]
        group_by="ticker",
        progress=False,
    )

    if data.empty:
        raise RuntimeError("yfinance returned no data. Check tickers or internet connection.")

    frames = []

    # MultiIndex columns: (symbol, field)
    for symbol in symbols:
        if symbol not in data.columns.get_level_values(0):
            print(f"  ⚠ {symbol}: not found in Yahoo data, skipping.")
            continue

        df_sym = data[symbol].reset_index()  # Date, Open, High, Low, Close, Adj Close, Volume
        if df_sym.empty:
            print(f"  ⚠ {symbol}: empty data, skipping.")
            continue

        df_sym["symbol"] = symbol
        df_sym = df_sym.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )

        # Ensure numeric
        for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            df_sym[col] = pd.to_numeric(df_sym[col], errors="coerce")

        df_sym = df_sym.dropna(subset=["adjusted_close", "volume"])
        frames.append(df_sym)

        print(f"  ✓ {symbol}: {len(df_sym)} rows")

    if not frames:
        raise RuntimeError("No portfolio data could be built from yfinance (all tickers empty).")

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["symbol", "date"])

    full.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nSaved combined portfolio data to: {PROCESSED_DATA_PATH}")
    print(f"Total rows: {len(full)}, Symbols: {full['symbol'].nunique()}")
    print(f"Date range: {full['date'].min().date()} to {full['date'].max().date()}\n")

    return full


def main(max_symbols: int = 50):
    df = fetch_yahoo_portfolio(max_symbols=max_symbols)
    return df


if __name__ == "__main__":
    main(max_symbols=50)
