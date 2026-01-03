"""
portfolio_dashboard.py

Institutional portfolio analytics dashboard with:
- Multi-symbol selection (persisted with st.session_state, max 10 symbols)
- 10-year date range analysis
- Performance & risk KPIs:
  - Total Return, Annualized Return, Volatility, Sharpe Ratio
  - Alpha, Beta, VaR(95%), Max Drawdown
  - Fundamentals (P/E, EPS, Dividend Yield, Market Cap) when available
- Dynamic Top N / Top 5 gainers & losers tables based on number of selections
- Robust chart saving (silent fail) to outputs/charts with title + timestamp
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts import portfolio_analytics as pa

CHARTS_DIR = "outputs/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

st.set_page_config(
    page_title="Institutional Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def save_plotly_fig(fig, title: str):
    """Try to save Plotly figure as PNG; skip silently if image export is unavailable."""
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{safe_title} {timestamp}.png"
    path = os.path.join(CHARTS_DIR, filename)
    try:
        fig.write_image(path)
    except Exception:
        # If kaleido or image export fails, ignore and continue
        pass


@st.cache_data(ttl=3600)
def load_data():
    return pa.load_portfolio_data()


df = load_data()

# ---------------- Sidebar: symbol selection with persistence ---------------- #

st.sidebar.title("📊 Configuration")

all_symbols = sorted(df["symbol"].unique().tolist())

# Initialize session state for selections
if "selected_symbols" not in st.session_state:
    st.session_state.selected_symbols = all_symbols[:5] if len(all_symbols) >= 5 else all_symbols

# Multiselect with persisted default
selected_symbols = st.sidebar.multiselect(
    "Select up to 10 symbols",
    all_symbols,
    default=st.session_state.selected_symbols,
)

# Enforce max 10 symbols
if len(selected_symbols) > 10:
    st.sidebar.warning("Maximum number of companies that can be selected at a single time is 10.")
    selected_symbols = selected_symbols[:10]

# Update session state
st.session_state.selected_symbols = selected_symbols

if not selected_symbols:
    st.warning("Please select at least one symbol.")
    st.stop()

# ---------------- Sidebar: date range ---------------- #

st.sidebar.subheader("Date range")
min_date = df["date"].min().date()
max_date = df["date"].max().date()
default_start = max_date - timedelta(days=365 * 5)

start_date = st.sidebar.date_input(
    "Start date",
    default_start,
    min_value=min_date,
    max_value=max_date,
)
end_date = st.sidebar.date_input(
    "End date",
    max_date,
    min_value=min_date,
    max_value=max_date,
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

df_filtered = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (df["symbol"].isin(selected_symbols))
].copy()

if df_filtered.empty:
    st.error("No data for selected symbols and date range.")
    st.stop()

num_selected = len(selected_symbols)

st.title("💼 Institutional Portfolio Analytics")
st.caption(f"Analysis from {start_date} to {end_date} | {num_selected} symbols")

# ---------------- Portfolio KPIs ---------------- #

st.header("Key Performance Indicators")

portfolio_metrics = pa.calculate_portfolio_metrics(
    df_filtered, start_date=str(start_date), end_date=str(end_date)
)

# Compute benchmark returns if available for alpha/beta use in gainers/losers
BENCH = pa.BENCHMARK_SYMBOL
bench_df = df_filtered[df_filtered["symbol"] == BENCH].sort_values("date")
benchmark_returns = None
if not bench_df.empty and len(bench_df) > 1:
    bp = bench_df["adjusted_close"].values
    b_ret = np.diff(bp) / bp[:-1]
    benchmark_returns = pd.Series(b_ret, index=bench_df["date"].iloc[1:])

# Top movers; dynamic N based on selections
top_n = num_selected if num_selected < 5 else 5
gainers, losers = pa.get_top_gainers_losers(
    df_filtered,
    "total_return",
    n=top_n,
    start_date=str(start_date),
    end_date=str(end_date),
    benchmark_returns=benchmark_returns,
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        "Total Return",
        f"{portfolio_metrics['total_return'] * 100:.2f}%",
        delta=f"{portfolio_metrics['annualized_return'] * 100:.2f}% annualized",
    )

with kpi2:
    st.metric(
        "Volatility (Ann.)",
        f"{portfolio_metrics['volatility'] * 100:.2f}%",
    )

with kpi3:
    sharpe = portfolio_metrics.get("sharpe_ratio", 0.0)
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

with kpi4:
    beta_val = portfolio_metrics.get("beta")
    st.metric("Portfolio Beta (vs SPY)", f"{beta_val:.2f}" if beta_val is not None else "N/A")

with kpi5:
    alpha_val = portfolio_metrics.get("alpha")
    st.metric("Portfolio Alpha", f"{alpha_val:.2f}" if alpha_val is not None else "N/A")

kpi6, kpi7, kpi8, kpi9, kpi10 = st.columns(5)

with kpi6:
    var_val = portfolio_metrics.get("var_95")
    st.metric("VaR 95% (daily)", f"{var_val * 100:.2f}%" if var_val is not None else "N/A")

# Fundamentals (from first selected symbol, if available)
first_symbol = selected_symbols[0]
fund = pa.get_fundamentals(first_symbol)

with kpi7:
    pe = fund.get("pe_ratio")
    st.metric(f"{first_symbol} P/E", f"{pe:.2f}" if pe is not None else "N/A")

with kpi8:
    eps = fund.get("eps")
    st.metric(f"{first_symbol} EPS", f"{eps:.2f}" if eps is not None else "N/A")

with kpi9:
    dy = fund.get("dividend_yield")
    st.metric(
        f"{first_symbol} Dividend Yield",
        f"{dy * 100:.2f}%" if dy is not None else "N/A",
    )

with kpi10:
    mc = fund.get("market_cap")
    st.metric(
        f"{first_symbol} Market Cap",
        f"{mc/1e9:.2f} B" if mc is not None else "N/A",
    )

st.markdown("---")

# ---------------- Charts row 1: Price Trends & Cumulative Returns ---------------- #

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Trends (Normalized to 100)")
    fig_price = go.Figure()
    for symbol in selected_symbols:
        sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
        if not sym_data.empty:
            first_price = sym_data["adjusted_close"].iloc[0]
            normalized = sym_data["adjusted_close"] / first_price * 100
            fig_price.add_trace(
                go.Scatter(
                    x=sym_data["date"],
                    y=normalized,
                    name=symbol,
                    mode="lines",
                )
            )
    fig_price.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Normalized Price (base=100)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)
    save_plotly_fig(fig_price, "Price Trends (Normalized to 100)")

with col2:
    st.subheader("Cumulative Returns")
    fig_cum = go.Figure()
    for symbol in selected_symbols:
        sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
        if not sym_data.empty:
            returns = sym_data["adjusted_close"].pct_change()
            cum_returns = (1 + returns).cumprod() - 1
            fig_cum.add_trace(
                go.Scatter(
                    x=sym_data["date"],
                    y=cum_returns * 100,
                    name=symbol,
                    mode="lines",
                )
            )
    fig_cum.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)
    save_plotly_fig(fig_cum, "Cumulative Returns")

st.markdown("---")

# ---------------- Risk Metrics Heatmap ---------------- #

st.subheader("Risk Metrics Heatmap")

risk_rows = []
for symbol in selected_symbols:
    result = pa.get_period_returns(
        df_filtered,
        symbol,
        start_date=str(start_date),
        end_date=str(end_date),
        benchmark_returns=benchmark_returns,
    )
    if result:
        risk_rows.append({
            "Symbol": symbol,
            "Annual Return": result["annualized_return"],
            "Volatility": result["volatility"],
            "Sharpe": result["sharpe_ratio"],
            "Max DD": result["max_drawdown"],
        })

risk_df = pd.DataFrame(risk_rows)

if not risk_df.empty:
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=risk_df[["Annual Return", "Volatility", "Sharpe", "Max DD"]].T.values * 100,
            x=risk_df["Symbol"],
            y=["Annual Return (%)", "Volatility (%)", "Sharpe Ratio", "Max DD (%)"],
            colorscale="RdYlGn",
            colorbar=dict(title="Value"),
        )
    )
    fig_heatmap.update_layout(height=300)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    save_plotly_fig(fig_heatmap, "Risk Metrics Heatmap")
else:
    st.write("Not enough data to compute risk metrics.")

st.markdown("---")

# ---------------- Top movers with dynamic N ---------------- #

col3, col4 = st.columns(2)
label_suffix = f"Top {top_n}" if num_selected >= 5 else f"Top {num_selected}"

with col3:
    st.subheader(f"🔝 {label_suffix} Gainers")
    if not gainers.empty:
        gainers_display = gainers[
            ["symbol", "total_return", "annualized_return", "sharpe_ratio", "volatility"]
        ].copy()
        gainers_display.columns = ["Symbol", "Total Return", "Annual Return", "Sharpe", "Volatility"]
        gainers_display["Total Return"] *= 100
        gainers_display["Annual Return"] *= 100
        gainers_display["Volatility"] *= 100
        st.dataframe(gainers_display, use_container_width=True)
    else:
        st.write("No data.")

with col4:
    st.subheader(f"🔽 {label_suffix} Losers")
    if not losers.empty:
        losers_display = losers[
            ["symbol", "total_return", "annualized_return", "sharpe_ratio", "volatility"]
        ].copy()
        losers_display.columns = ["Symbol", "Total Return", "Annual Return", "Sharpe", "Volatility"]
        losers_display["Total Return"] *= 100
        losers_display["Annual Return"] *= 100
        losers_display["Volatility"] *= 100
        st.dataframe(losers_display, use_container_width=True)
    else:
        st.write("No data.")

st.markdown("---")

# ---------------- Rolling Volatility ---------------- #

st.subheader("Rolling 20-Day Volatility")
fig_vol = go.Figure()
for symbol in selected_symbols:
    sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
    if not sym_data.empty:
        returns = sym_data["adjusted_close"].pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        fig_vol.add_trace(
            go.Scatter(
                x=sym_data["date"],
                y=rolling_vol * 100,
                name=symbol,
                mode="lines",
            )
        )
fig_vol.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Annualized Volatility (%)",
    hovermode="x unified",
)
st.plotly_chart(fig_vol, use_container_width=True)
save_plotly_fig(fig_vol, "Rolling 20-Day Volatility")

st.markdown("---")

# ---------------- Detailed Risk Metrics Table ---------------- #

st.subheader("Detailed Risk Metrics Table")
if not risk_df.empty:
    risk_df_display = risk_df.copy()
    risk_df_display["Annual Return"] = risk_df_display["Annual Return"] * 100
    risk_df_display["Volatility"] = risk_df_display["Volatility"] * 100
    risk_df_display["Max DD"] = risk_df_display["Max DD"] * 100
    st.dataframe(risk_df_display, use_container_width=True)

st.caption(
    "KPIs shown only where data is available. "
    "Sharpe: risk-adjusted return; Beta/Alpha vs SPY; "
    "VaR(95%): worst expected daily loss at 95% confidence; "
    "Max Drawdown: largest peak-to-trough decline."
)
