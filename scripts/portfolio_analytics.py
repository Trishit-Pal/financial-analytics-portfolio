import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

RISK_FREE_RATE = 0.045
BENCHMARK_SYMBOL = "SPY"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "portfolio_data.csv")


def load_portfolio_data():
    """Load portfolio data from CSV. Download if not found."""
    if not os.path.exists(DATA_PATH):
        from data_loading_portfolio import main
        main()
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def get_fundamentals(symbol):
    """Get P/E, EPS, dividend yield, market cap for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield", 0),
            "market_cap": info.get("marketCap")
        }
    except:
        return {"pe_ratio": None, "eps": None, "dividend_yield": None, "market_cap": None}


def calculate_returns(prices):
    """Calculate daily returns from price series."""
    return prices.pct_change().dropna()


def calculate_portfolio_metrics(df_filtered, start_date, end_date):
    """Calculate equal-weight portfolio KPIs."""
    daily_returns = df_filtered.pivot(
        index="date", columns="symbol", values="adjusted_close"
    ).pct_change().mean(axis=1).dropna()
    
    if len(daily_returns) < 2:
        return {
            "total_return": 0, "annualized_return": 0, "volatility": 0,
            "sharpe_ratio": 0, "beta": np.nan, "alpha": np.nan, "var_95": 0
        }
    
    total_return = (1 + daily_returns).prod() - 1
    days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    ann_return = (1 + total_return) ** (365 / max(days, 1)) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (ann_return - RISK_FREE_RATE) / max(volatility, 0.0001)
    
    # Beta/Alpha vs benchmark
    bench_prices = df_filtered[df_filtered["symbol"] == BENCHMARK_SYMBOL]["adjusted_close"]
    beta, alpha = np.nan, np.nan
    if len(bench_prices) > 30:
        bench_ret = calculate_returns(bench_prices)
        port_ret = daily_returns.reindex(bench_ret.index).dropna()
        bench_aligned = bench_ret.reindex(port_ret.index).dropna()
        if len(port_ret) > 30 and len(bench_aligned) > 30:
            slope, intercept, _, _, _ = stats.linregress(bench_aligned, port_ret)
            beta, alpha = slope, intercept * 252
    
    var_95 = np.percentile(daily_returns.dropna(), 5)
    
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "beta": beta,
        "alpha": alpha,
        "var_95": var_95
    }


def get_period_returns(df_filtered, symbol, start_date, end_date, benchmark_returns=None):
    """Calculate KPIs for a single symbol over period."""
    sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
    if sym_data.empty or len(sym_data) < 2:
        return None
    
    prices = sym_data["adjusted_close"]
    returns = calculate_returns(prices)
    
    total_return = (1 + returns).prod() - 1
    days = len(returns)
    ann_return = (1 + total_return) ** (365 / max(days, 1)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (ann_return - RISK_FREE_RATE) / max(volatility, 0.0001)
    
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    max_dd = drawdown.min()
    
    beta, alpha = np.nan, np.nan
    if benchmark_returns is not None and len(returns) > 30:
        aligned_bench = benchmark_returns.reindex(returns.index).dropna()
        aligned_ret = returns.reindex(aligned_bench.index).dropna()
        if len(aligned_ret) > 30:
            slope, intercept, _, _, _ = stats.linregress(aligned_bench, aligned_ret)
            beta, alpha = slope, intercept * 252
    
    var_95 = np.percentile(returns.dropna(), 5) if len(returns) > 0 else 0
    
    return {
        "symbol": symbol,
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "beta": beta,
        "alpha": alpha,
        "max_drawdown": max_dd,
        "var_95": var_95
    }


def get_top_gainers_losers(df_filtered, metric="total_return", n=5, start_date=None, end_date=None, benchmark_returns=None):
    """Get top N gainers and losers."""
    symbols = df_filtered["symbol"].unique()
    results = []
    
    for symbol in symbols:
        result = get_period_returns(df_filtered, symbol, start_date, end_date, benchmark_returns)
        if result:
            results.append(result)
    
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    gainers = df_results.nlargest(n, metric)
    losers = df_results.nsmallest(n, metric)
    
    return gainers, losers


if __name__ == "__main__":
    df = load_portfolio_data()
    print(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(df.head())