"""
portfolio_analytics.py

Performance & risk analytics:
- Total return, TWR (via geometric return), annualized return
- Volatility, Sharpe ratio, beta, max drawdown
- Top gainers/losers by period
- Portfolio-level metrics (simple equal-weight approximation)
- Alpha (Jensen's alpha vs benchmark)
- VaR (historical Value at Risk)
- Fundamentals (P/E, EPS, dividend yield, market cap) via yfinance
"""

import os
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf  # for fundamentals

RISK_FREE_RATE = 0.045  # approximate annual risk-free rate
BENCHMARK_SYMBOL = "SPY"  # used for alpha/beta if present in data


# ========= Data loading =========

def load_portfolio_data(csv_path: str = "data/processed/portfolio_data.csv") -> pd.DataFrame:
    """Load combined portfolio CSV with basic validation."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run 'python -m scripts.pipeline_portfolio' first to generate it."
        )
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"])


# ========= Fundamentals =========

def get_fundamentals(symbol: str) -> Dict[str, Optional[float]]:
    """
    Fetch fundamentals for a single symbol via yfinance:
    - P/E ratio
    - EPS
    - Dividend yield
    - Market cap

    If data not available, returns None for that KPI.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return {
            "pe_ratio": None,
            "eps": None,
            "dividend_yield": None,
            "market_cap": None,
        }


# ========= Performance Metrics =========

def calculate_total_return(prices: pd.Series) -> float:
    if len(prices) < 2 or prices.iloc[0] == 0:
        return 0.0
    return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]


def calculate_twr(prices: pd.Series) -> float:
    if len(prices) < 2 or prices.iloc[0] == 0:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0]) - 1.0


def calculate_annualized_return(prices: pd.Series, days: int) -> float:
    if days <= 0:
        return 0.0
    total_return = calculate_total_return(prices)
    years = days / 365.25
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    daily_vol = returns.std()
    if annualize:
        return daily_vol * np.sqrt(252)
    return daily_vol


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    if returns.empty:
        return 0.0
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    if annual_vol == 0:
        return 0.0
    return (annual_return - risk_free_rate) / annual_vol


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    covariance = np.cov(stock_returns.fillna(0), market_returns.fillna(0))[0][1]
    market_variance = market_returns.var()
    if market_variance == 0:
        return 1.0
    return covariance / market_variance


def calculate_max_drawdown(prices: pd.Series) -> tuple:
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    min_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    peak_idx = prices[:trough_idx].idxmax()
    return (min_dd, peak_idx, trough_idx)


def calculate_alpha(stock_returns: pd.Series, benchmark_returns: pd.Series, beta: float) -> float:
    """
    Jensen's Alpha:
    alpha = (R_s - R_f) - beta * (R_m - R_f)
    where R_s: stock return, R_m: market return, R_f: risk-free rate.
    """
    if stock_returns.empty or benchmark_returns.empty:
        return 0.0
    stock_excess = stock_returns.mean() * 252 - RISK_FREE_RATE
    benchmark_excess = benchmark_returns.mean() * 252 - RISK_FREE_RATE
    return stock_excess - beta * benchmark_excess


def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
    """
    Historical Value at Risk (VaR) at given confidence level (e.g., 0.05 for 95%).
    Returns a negative number representing potential loss.
    """
    if returns.dropna().empty:
        return 0.0
    return np.percentile(returns.dropna(), confidence * 100)


# ========= Per-symbol Period Metrics =========

def get_period_returns(
    df: pd.DataFrame,
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> dict | None:
    """
    Compute per-symbol metrics for a given period:
    - total_return, annualized_return
    - volatility, sharpe_ratio
    - max_drawdown
    - optional: beta, alpha, VaR (if benchmark + returns available)
    """
    symbol_df = df[df["symbol"] == symbol].copy()
    if symbol_df.empty:
        return None

    if start_date:
        symbol_df = symbol_df[symbol_df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        symbol_df = symbol_df[symbol_df["date"] <= pd.to_datetime(end_date)]

    if len(symbol_df) < 2:
        return None

    prices = symbol_df["adjusted_close"].values
    daily_returns = np.diff(prices) / prices[:-1]
    daily_returns_series = pd.Series(daily_returns, index=symbol_df["date"].iloc[1:])

    num_days = (symbol_df["date"].max() - symbol_df["date"].min()).days
    total_return = calculate_total_return(pd.Series(prices))
    ann_return = calculate_annualized_return(pd.Series(prices), num_days) if num_days > 0 else 0.0
    volatility = calculate_volatility(daily_returns_series)
    sharpe = calculate_sharpe_ratio(daily_returns_series)
    dd_pct, dd_peak, dd_trough = calculate_max_drawdown(pd.Series(prices))

    beta = None
    alpha = None
    var_95 = None

    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat(
            [daily_returns_series, benchmark_returns],
            axis=1,
            join="inner",
        ).dropna()
        if not aligned.empty:
            stock_r = aligned.iloc[:, 0]
            mkt_r = aligned.iloc[:, 1]
            beta = calculate_beta(stock_r, mkt_r)
            alpha = calculate_alpha(stock_r, mkt_r, beta)
    # VaR (95%) from stock's own returns
    if not daily_returns_series.empty:
        var_95 = calculate_var(daily_returns_series, confidence=0.05)

    return {
        "symbol": symbol,
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd_pct,
        "max_drawdown_peak": dd_peak,
        "max_drawdown_trough": dd_trough,
        "num_days": num_days,
        "start_date": symbol_df["date"].min(),
        "end_date": symbol_df["date"].max(),
        "beta": beta,
        "alpha": alpha,
        "var_95": var_95,
    }


# ========= Top movers =========

def get_top_gainers_losers(
    df: pd.DataFrame,
    metric: str = "total_return",
    n: int = 5,
    start_date: str = None,
    end_date: str = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-symbol metrics for all symbols and return Top N gainers/losers.
    """
    results = []
    for symbol in df["symbol"].unique():
        result = get_period_returns(df, symbol, start_date, end_date, benchmark_returns)
        if result:
            results.append(result)
    if not results:
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(results)
    top_gainers = results_df.nlargest(n, metric)
    top_losers = results_df.nsmallest(n, metric)
    return top_gainers, top_losers


# ========= Portfolio metrics (equal weight) =========

def calculate_portfolio_metrics(
    df: pd.DataFrame,
    weights: dict | None = None,
    start_date: str = None,
    end_date: str = None,
    benchmark_symbol: str = BENCHMARK_SYMBOL,
) -> dict:
    """
    Portfolio metrics (equal-weight approximation) plus alpha/beta if benchmark available.
    """
    symbols = df["symbol"].unique()
    if weights is None:
        weights = {s: 1 / len(symbols) for s in symbols}

    # Benchmark returns if present
    bench_df = df[df["symbol"] == benchmark_symbol].copy()
    benchmark_returns = None
    if not bench_df.empty:
        bench_df = bench_df.sort_values("date")
        if start_date:
            bench_df = bench_df[bench_df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            bench_df = bench_df[bench_df["date"] <= pd.to_datetime(end_date)]
        if len(bench_df) > 1:
            bench_prices = bench_df["adjusted_close"].values
            b_ret = np.diff(bench_prices) / bench_prices[:-1]
            benchmark_returns = pd.Series(b_ret, index=bench_df["date"].iloc[1:])

    portfolio_total_return = 0.0
    portfolio_annual_return = 0.0
    vol_components = []
    betas = []
    alphas = []
    vars_95 = []

    for symbol in symbols:
        result = get_period_returns(df, symbol, start_date, end_date, benchmark_returns)
        if result:
            w = weights.get(symbol, 0)
            portfolio_total_return += result["total_return"] * w
            portfolio_annual_return += result["annualized_return"] * w
            vol_components.append(result["volatility"] * w)
            if result["beta"] is not None:
                betas.append(result["beta"] * w)
            if result["alpha"] is not None:
                alphas.append(result["alpha"] * w)
            if result["var_95"] is not None:
                vars_95.append(result["var_95"] * w)

    portfolio_volatility = np.sqrt(sum(v**2 for v in vol_components))
    portfolio_sharpe = (
        (portfolio_annual_return - RISK_FREE_RATE) / portfolio_volatility
        if portfolio_volatility > 0
        else 0.0
    )
    portfolio_beta = sum(betas) if betas else None
    portfolio_alpha = sum(alphas) if alphas else None
    portfolio_var_95 = sum(vars_95) if vars_95 else None

    return {
        "total_return": portfolio_total_return,
        "annualized_return": portfolio_annual_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": portfolio_sharpe,
        "num_symbols": len(symbols),
        "beta": portfolio_beta,
        "alpha": portfolio_alpha,
        "var_95": portfolio_var_95,
    }


# ========= Main (CLI) =========

def main(start_date: str = None, end_date: str = None):
    print("\n=== PORTFOLIO ANALYTICS ===")
    df = load_portfolio_data()

    if end_date is None:
        end_date = df["date"].max().strftime("%Y-%m-%d")
    if start_date is None:
        default_start = df["date"].max() - timedelta(days=365 * 10)
        start_date = default_start.strftime("%Y-%m-%d")

    metrics = calculate_portfolio_metrics(df, start_date=start_date, end_date=end_date)
    print(f"\nPortfolio Metrics ({start_date} to {end_date}):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.4f}")
        else:
            print(f"  {k}: {v}")

    # Benchmark returns for gainers/losers
    bench_df = df[df["symbol"] == BENCHMARK_SYMBOL].copy()
    benchmark_returns = None
    if not bench_df.empty:
        bench_df = bench_df.sort_values("date")
        bench_df = bench_df[
            (bench_df["date"] >= pd.to_datetime(start_date))
            & (bench_df["date"] <= pd.to_datetime(end_date))
        ]
        if len(bench_df) > 1:
            bench_prices = bench_df["adjusted_close"].values
            b_ret = np.diff(bench_prices) / bench_prices[:-1]
            benchmark_returns = pd.Series(b_ret, index=bench_df["date"].iloc[1:])

    gainers, losers = get_top_gainers_losers(
        df,
        "total_return",
        n=5,
        start_date=start_date,
        end_date=end_date,
        benchmark_returns=benchmark_returns,
    )

    print("\nTop 5 Gainers:")
    if not gainers.empty:
        print(gainers[["symbol", "total_return", "sharpe_ratio", "volatility"]].to_string())
    else:
        print("  None")

    print("\nTop 5 Losers:")
    if not losers.empty:
        print(losers[["symbol", "total_return", "sharpe_ratio", "volatility"]].to_string())
    else:
        print("  None")

    return df, metrics, gainers, losers


if __name__ == "__main__":
    main()
