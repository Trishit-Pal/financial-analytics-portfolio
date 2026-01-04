"""
portfolio_dashboard.py

Institutional portfolio analytics dashboard with:
- Multi-symbol selection with proper reactivity
- Persistent filters via URL query parameters (survive browser refresh)
- Dynamic portfolio-level KPIs (return, volatility, Sharpe, beta, alpha, VaR)
- Top gainers/losers within the filtered universe
- Risk metrics table and correlation heatmap
- Matplotlib-based charts saved as .png on each page render
"""

import os
import sys
from datetime import datetime, timedelta
from urllib.parse import parse_qs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional: plotly is still imported if you want to extend later
import plotly.graph_objects as go  # noqa: F401

# Make scripts importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts import portfolio_analytics as pa  # noqa: E402

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Streamlit page config & basic styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Institutional Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS hook (extend as needed)
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helper: query param persistence
# -----------------------------------------------------------------------------
def get_query_params():
    """Return current query params as a dict of lists."""
    # st.query_params in recent Streamlit; fallback using script run context if needed
    try:
        return st.query_params
    except Exception:
        # Fallback: parse from environment if available (very defensive)
        return parse_qs("")


def sync_query_params(selected_symbols, start_date, end_date):
    """Write selected filters into URL query params to persist across refresh."""
    try:
        st.query_params.update(
            {
                "symbols": ",".join(selected_symbols),
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }
        )
    except Exception:
        # If not supported, silently ignore; state still lives in session_state
        pass


def load_state_from_query(df: pd.DataFrame):
    """Initialize symbols and dates from URL query params, with safe fallbacks."""
    params = get_query_params()
    all_symbols = sorted(df["symbol"].unique().tolist())

    # Symbols
    symbols_param = ""
    if isinstance(params, dict) and "symbols" in params:
        # st.query_params returns a dict-like, values are str or list-like
        val = params.get("symbols")
        symbols_param = val[0] if isinstance(val, list) else str(val)
    if symbols_param:
        selected_symbols = [s for s in symbols_param.split(",") if s in all_symbols]
    else:
        selected_symbols = all_symbols[:3] if len(all_symbols) >= 3 else all_symbols

    # Dates
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_start = max_date - timedelta(days=365 * 5)

    def _get_param(key, default_str):
        if isinstance(params, dict) and key in params:
            val = params.get(key)
            val = val[0] if isinstance(val, list) else str(val)
            return val or default_str
        return default_str

    start_str = _get_param("start", default_start.strftime("%Y-%m-%d"))
    end_str = _get_param("end", max_date.strftime("%Y-%m-%d"))

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    except ValueError:
        start_date = default_start

    try:
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        end_date = max_date

    # Clamp to valid range
    start_date = max(min_date, min(start_date, max_date))
    end_date = max(min_date, min(end_date, max_date))

    return selected_symbols, start_date, end_date


# -----------------------------------------------------------------------------
# Matplotlib-based savers
# -----------------------------------------------------------------------------
def _safe_filename(title: str) -> str:
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{safe_title} {timestamp}.png"


def save_price_trends_image(
    df_filtered: pd.DataFrame, selected_symbols: list[str], title: str
):
    """Save normalized price trends as PNG."""
    if df_filtered.empty or not selected_symbols:
        return
    path = os.path.join(CHARTS_DIR, _safe_filename(title))
    try:
        plt.figure(figsize=(12, 5))
        for symbol in selected_symbols:
            sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
            if not sym_data.empty:
                first_price = sym_data["adjusted_close"].iloc[0]
                normalized = sym_data["adjusted_close"] / first_price * 100
                plt.plot(sym_data["date"], normalized, label=symbol, linewidth=2)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price (base=100)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception:
        plt.close()


def save_cumulative_returns_image(
    df_filtered: pd.DataFrame, selected_symbols: list[str], title: str
):
    """Save cumulative returns as PNG."""
    if df_filtered.empty or not selected_symbols:
        return
    path = os.path.join(CHARTS_DIR, _safe_filename(title))
    try:
        plt.figure(figsize=(12, 5))
        for symbol in selected_symbols:
            sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
            if not sym_data.empty:
                returns = sym_data["adjusted_close"].pct_change()
                cum_returns = (1 + returns).cumprod() - 1
                plt.plot(sym_data["date"], cum_returns * 100, label=symbol, linewidth=2)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception:
        plt.close()


def save_risk_heatmap_image(risk_df: pd.DataFrame, title: str):
    """Save risk metrics heatmap as PNG."""
    if risk_df.empty:
        return
    path = os.path.join(CHARTS_DIR, _safe_filename(title))
    try:
        metrics = ["Annual Return", "Volatility", "Sharpe", "Max DD"]
        data = risk_df[metrics].copy()
        data["Annual Return"] *= 100
        data["Volatility"] *= 100
        data["Max DD"] *= 100

        plt.figure(figsize=(max(10, 0.8 * len(risk_df)), 5))
        im = plt.imshow(data.T.values, aspect="auto", cmap="RdYlGn")
        plt.colorbar(im, label="Value")
        plt.yticks(
            range(len(metrics)),
            ["Annual Return (%)", "Volatility (%)", "Sharpe", "Max DD (%)"],
        )
        plt.xticks(
            range(len(risk_df["Symbol"])),
            risk_df["Symbol"],
            rotation=45,
            ha="right",
        )
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception:
        plt.close()


def save_rolling_vol_image(
    df_filtered: pd.DataFrame, selected_symbols: list[str], title: str
):
    """Save rolling 20-day volatility as PNG."""
    if df_filtered.empty or not selected_symbols:
        return
    path = os.path.join(CHARTS_DIR, _safe_filename(title))
    try:
        plt.figure(figsize=(12, 5))
        for symbol in selected_symbols:
            sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
            if not sym_data.empty and len(sym_data) > 20:
                returns = sym_data["adjusted_close"].pct_change()
                rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                plt.plot(sym_data["date"], rolling_vol * 100, label=symbol, linewidth=2)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Annualized Volatility (%)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception:
        plt.close()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    return pa.load_portfolio_data()


df = load_data()

# -----------------------------------------------------------------------------
# Sidebar: filters with URL-based persistence
# -----------------------------------------------------------------------------
saved_symbols, saved_start, saved_end = load_state_from_query(df)

st.sidebar.title("📊 Configuration")
st.sidebar.markdown("---")

all_symbols = sorted(df["symbol"].unique().tolist())
selected_symbols = st.sidebar.multiselect(
    "📌 Select companies (up to 10)",
    all_symbols,
    default=saved_symbols,
    help="Choose stocks to analyze. Metrics update automatically.",
)
if len(selected_symbols) > 10:
    st.sidebar.warning("⚠️ Maximum 10 companies allowed. Showing first 10.")
    selected_symbols = selected_symbols[:10]

if not selected_symbols:
    st.warning("⚠️ Please select at least one stock to begin analysis.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Analysis Period")

min_date = df["date"].min().date()
max_date = df["date"].max().date()
col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input(
        "From",
        saved_start,
        min_value=min_date,
        max_value=max_date,
    )
with col_end:
    end_date = st.date_input(
        "To",
        saved_end,
        min_value=min_date,
        max_value=max_date,
    )

if start_date > end_date:
    st.sidebar.error("❌ Start date must be before end date.")
    st.stop()

# Sync filters into URL so refresh keeps them
sync_query_params(selected_symbols, start_date, end_date)

# Filtered data
df_filtered = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (df["symbol"].isin(selected_symbols))
].copy()

if df_filtered.empty:
    st.error("❌ No data available for selected symbols and date range.")
    st.stop()

# -----------------------------------------------------------------------------
# Portfolio-level KPIs (real-time)
# -----------------------------------------------------------------------------
metrics = pa.calculate_portfolio_metrics(
    df[
        (df["symbol"].isin(selected_symbols))
        & (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
    ]
)

total_return = metrics["total_return"]
annualized_return = metrics["annualized_return"]
volatility = metrics["volatility"]
sharpe_ratio = metrics["sharpe_ratio"]
beta = metrics.get("beta")
alpha = metrics.get("alpha")
var_95 = metrics.get("var_95")
num_symbols = metrics["num_symbols"]

st.markdown(
    f"📊 Analyzing **{num_symbols}** "
    f"{'stock' if num_symbols == 1 else 'stocks'} "
    f"from **{start_date}** to **{end_date}**"
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Return", f"{total_return * 100:,.2f} %")
with kpi2:
    st.metric("Annualized Return", f"{annualized_return * 100:,.2f} %")
with kpi3:
    st.metric("Volatility (ann.)", f"{volatility * 100:,.2f} %")
with kpi4:
    st.metric("Sharpe Ratio", f"{sharpe_ratio:,.2f}")

kpi5, kpi6, kpi7 = st.columns(3)
with kpi5:
    st.metric("Portfolio Beta", f"{beta:,.2f}" if beta is not None else "N/A")
with kpi6:
    st.metric(
        "Jensen's Alpha",
        f"{alpha * 100:,.2f} %" if alpha is not None else "N/A",
    )
with kpi7:
    st.metric(
        "95% VaR (daily)",
        f"{var_95 * 100:,.2f} %" if var_95 is not None else "N/A",
    )

# -----------------------------------------------------------------------------
# Top gainers / losers within filtered symbols
# -----------------------------------------------------------------------------
bench_df = df[df["symbol"] == pa.BENCHMARK_SYMBOL].copy()
benchmark_returns = None
if not bench_df.empty:
    bench_df = bench_df.sort_values("date")
    bench_df = bench_df[
        (bench_df["date"].dt.date >= start_date)
        & (bench_df["date"].dt.date <= end_date)
    ]
    if len(bench_df) > 1:
        bench_prices = bench_df["adjusted_close"].values
        b_ret = np.diff(bench_prices) / bench_prices[:-1]
        benchmark_returns = pd.Series(b_ret, index=bench_df["date"].iloc[1:])

gainers, losers = pa.get_top_gainers_losers(
    df[df["symbol"].isin(selected_symbols)],
    metric="total_return",
    n=5,
    start_date=start_date.strftime("%Y-%m-%d"),
    end_date=end_date.strftime("%Y-%m-%d"),
    benchmark_returns=benchmark_returns,
)

top1, top2 = st.columns(2)
with top1:
    st.subheader("Top 5 Gainers")
    if not gainers.empty:
        st.dataframe(
            gainers[["symbol", "total_return", "sharpe_ratio", "volatility"]]
            .rename(
                columns={
                    "symbol": "Symbol",
                    "total_return": "Total Return",
                    "sharpe_ratio": "Sharpe",
                    "volatility": "Volatility",
                }
            )
            .assign(
                **{
                    "Total Return": gainers["total_return"] * 100,
                    "Volatility": gainers["volatility"] * 100,
                }
            )
            .style.format(
                {
                    "Total Return": "{:.2f} %",
                    "Volatility": "{:.2f} %",
                    "Sharpe": "{:.2f}",
                }
            )
        )
    else:
        st.write("No data.")
with top2:
    st.subheader("Top 5 Losers")
    if not losers.empty:
        st.dataframe(
            losers[["symbol", "total_return", "sharpe_ratio", "volatility"]]
            .rename(
                columns={
                    "symbol": "Symbol",
                    "total_return": "Total Return",
                    "sharpe_ratio": "Sharpe",
                    "volatility": "Volatility",
                }
            )
            .assign(
                **{
                    "Total Return": losers["total_return"] * 100,
                    "Volatility": losers["volatility"] * 100,
                }
            )
            .style.format(
                {
                    "Total Return": "{:.2f} %",
                    "Volatility": "{:.2f} %",
                    "Sharpe": "{:.2f}",
                }
            )
        )
    else:
        st.write("No data.")

# -----------------------------------------------------------------------------
# Risk metrics by symbol
# -----------------------------------------------------------------------------
risk_rows = []
for sym in selected_symbols:
    sym_df = df_filtered[df_filtered["symbol"] == sym].sort_values("date")
    if len(sym_df) < 2:
        continue
    prices = sym_df["adjusted_close"].values
    rets = sym_df["adjusted_close"].pct_change().dropna()
    num_days = (sym_df["date"].max() - sym_df["date"].min()).days
    ann_ret = pa.calculate_annualized_return(pd.Series(prices), num_days)
    vol = pa.calculate_volatility(rets)
    sharpe = pa.calculate_sharpe_ratio(rets)
    max_dd, _, _ = pa.calculate_max_drawdown(pd.Series(prices))
    risk_rows.append(
        {
            "Symbol": sym,
            "Annual Return": ann_ret,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Max DD": max_dd,
        }
    )

risk_df = pd.DataFrame(risk_rows)
if not risk_df.empty:
    st.subheader("Risk Metrics by Symbol")
    st.dataframe(
        risk_df.assign(
            **{
                "Annual Return": risk_df["Annual Return"] * 100,
                "Volatility": risk_df["Volatility"] * 100,
                "Max DD": risk_df["Max DD"] * 100,
            }
        ).style.format(
            {
                "Annual Return": "{:.2f} %",
                "Volatility": "{:.2f} %",
                "Sharpe": "{:.2f}",
                "Max DD": "{:.2f} %",
            }
        )
    )
    # Save risk heatmap PNG (created on each render/refresh)
    save_risk_heatmap_image(risk_df, "Risk metrics heatmap")

# -----------------------------------------------------------------------------
# Time-series Matplotlib charts (and PNG saving)
# -----------------------------------------------------------------------------
# Normalized price trends
st.subheader("Normalized Price Trends (base = 100)")
price_title = "Normalized price trends"
save_price_trends_image(df_filtered, selected_symbols, price_title)

fig_price, ax_price = plt.subplots(figsize=(12, 5))
for symbol in selected_symbols:
    sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
    if not sym_data.empty:
        first_price = sym_data["adjusted_close"].iloc[0]
        normalized = sym_data["adjusted_close"] / first_price * 100
        ax_price.plot(sym_data["date"], normalized, label=symbol, linewidth=2)
ax_price.set_title(price_title)
ax_price.set_xlabel("Date")
ax_price.set_ylabel("Normalized Price (base=100)")
ax_price.legend(loc="best")
ax_price.grid(alpha=0.3)
st.pyplot(fig_price)
plt.close(fig_price)

# Cumulative returns
st.subheader("Cumulative Returns")
cum_title = "Cumulative returns"
save_cumulative_returns_image(df_filtered, selected_symbols, cum_title)

fig_cum, ax_cum = plt.subplots(figsize=(12, 5))
for symbol in selected_symbols:
    sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
    if not sym_data.empty:
        rets = sym_data["adjusted_close"].pct_change()
        cum_ret = (1 + rets).cumprod() - 1
        ax_cum.plot(sym_data["date"], cum_ret * 100, label=symbol, linewidth=2)
ax_cum.set_title(cum_title)
ax_cum.set_xlabel("Date")
ax_cum.set_ylabel("Cumulative Return (%)")
ax_cum.legend(loc="best")
ax_cum.grid(alpha=0.3)
st.pyplot(fig_cum)
plt.close(fig_cum)

# Rolling 20-day volatility
st.subheader("Rolling 20-day Volatility")
vol_title = "Rolling 20-day volatility"
save_rolling_vol_image(df_filtered, selected_symbols, vol_title)

fig_vol, ax_vol = plt.subplots(figsize=(12, 5))
for symbol in selected_symbols:
    sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
    if not sym_data.empty and len(sym_data) > 20:
        rets = sym_data["adjusted_close"].pct_change()
        rolling_vol = rets.rolling(20).std() * np.sqrt(252) * 100
        ax_vol.plot(sym_data["date"], rolling_vol, label=symbol, linewidth=2)
ax_vol.set_title(vol_title)
ax_vol.set_xlabel("Date")
ax_vol.set_ylabel("Annualized Volatility (%)")
ax_vol.legend(loc="best")
ax_vol.grid(alpha=0.3)
st.pyplot(fig_vol)
plt.close(fig_vol)

# -----------------------------------------------------------------------------
# Correlation heatmap
# -----------------------------------------------------------------------------
st.subheader("Return Correlation Heatmap")
corr_df = (
    df_filtered.pivot(index="date", columns="symbol", values="adjusted_close")
    .pct_change()
    .corr()
)
if corr_df.shape[0] > 1:
    fig_corr, ax_corr = plt.subplots(
        figsize=(0.8 * corr_df.shape[0], 0.8 * corr_df.shape[0])
    )
    im = ax_corr.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(corr_df.columns)))
    ax_corr.set_yticks(range(len(corr_df.columns)))
    ax_corr.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax_corr.set_yticklabels(corr_df.columns)
    plt.colorbar(im, ax=ax_corr, label="Correlation")
    plt.tight_layout()
    st.pyplot(fig_corr)

    corr_path = os.path.join(CHARTS_DIR, _safe_filename("Correlation heatmap"))
    fig_corr.savefig(corr_path, dpi=150)
    plt.close(fig_corr)
else:
    st.write("Not enough symbols for correlation matrix.")
