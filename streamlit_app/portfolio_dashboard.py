import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts import portfolio_analytics as pa

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Generate timestamped subfolder for this refresh
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CHARTS_SUBDIR = os.path.join(CHARTS_DIR, TIMESTAMP)
os.makedirs(CHARTS_SUBDIR, exist_ok=True)

st.set_page_config(
    page_title="Institutional Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ENHANCED PROFESSIONAL CSS
st.markdown("""
    <style>
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --accent-color: #FFD700;
            --success-color: #4CAF50;
            --danger-color: #FF5252;
            --dark-bg: #0F0F23;
            --light-bg: #F5F7FA;
            --text-dark: #1A1A2E;
            --text-light: #FFFFFF;
            --border-color: #E0E7FF;
        }
        
        html, body {
            scroll-behavior: smooth;
        }
        
        .main {
            padding-top: 1rem;
            background-color: var(--light-bg);
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            padding: 2.5rem 2rem;
            border-radius: 15px;
            color: var(--text-light);
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .dashboard-header h1 {
            margin: 0;
            font-size: clamp(1.8rem, 5vw, 3rem);
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        
        .dashboard-header p {
            margin: 0.8rem 0 0 0;
            font-size: clamp(0.9rem, 2vw, 1.1rem);
            opacity: 0.95;
            font-weight: 300;
        }
        
        .selection-info {
            background-color: #F0F4FF;
            border-left: 5px solid var(--primary-color);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        }
        
        .selection-info strong {
            color: var(--text-dark);
            font-weight: 700;
        }
        
        .companies-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 0.8rem;
        }
        
        .company-badge {
            background-color: #E8EFFE;
            color: var(--text-dark);
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.95rem;
            border: 2px solid var(--primary-color);
            display: inline-block;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.15);
        }
        
        .section-header {
            color: var(--primary-color);
            font-size: clamp(1.3rem, 3vw, 1.8rem);
            font-weight: 800;
            margin-top: 2.5rem;
            margin-bottom: 1.8rem;
            padding-bottom: 1rem;
            border-bottom: 3px solid var(--primary-color);
            letter-spacing: -0.3px;
        }
        
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(clamp(150px, 20vw, 220px), 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .indicator-box {
            background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
            border: 2px solid var(--primary-color);
            border-radius: 12px;
            padding: clamp(1rem, 2vw, 1.5rem);
            text-align: center;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.08);
            transition: all 0.3s ease;
        }
        
        .indicator-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
        }
        
        .indicator-label {
            font-size: clamp(0.75rem, 1.5vw, 0.95rem);
            color: var(--text-dark);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 0.8rem;
        }
        
        .indicator-value {
            font-size: clamp(1.5rem, 4vw, 2.5rem);
            font-weight: 800;
            color: var(--primary-color);
            margin: 0.8rem 0;
            line-height: 1.2;
        }
        
        .chart-description {
            background: linear-gradient(135deg, #F0F4FF 0%, #F5EBFF 100%);
            padding: 1.2rem 1.5rem;
            border-left: 5px solid var(--primary-color);
            border-radius: 8px;
            margin-top: 1rem;
            font-size: 0.95rem;
            color: var(--text-dark);
            line-height: 1.7;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.08);
        }
        
        .subheader-text {
            font-size: clamp(1.1rem, 2.5vw, 1.5rem);
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 1rem;
        }
        
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F5F7FA 0%, #FFFFFF 100%);
        }
        
        .footer-text {
            text-align: center;
            color: #999;
            margin-top: 3rem;
            padding: 2rem;
            border-top: 2px solid var(--border-color);
            font-size: 0.9rem;
        }
        
        @media (max-width: 1200px) {
            .dashboard-header {
                padding: 2rem 1.5rem;
            }
        }
        
        @media (max-width: 768px) {
            .dashboard-header h1 {
                font-size: 1.5rem;
            }
            
            .kpi-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 0.8rem;
            }
            
            .indicator-box {
                padding: 0.8rem;
            }
            
            .indicator-value {
                font-size: 1.3rem;
            }
            
            .companies-list {
                gap: 0.6rem;
            }
            
            .company-badge {
                font-size: 0.85rem;
                padding: 0.4rem 0.8rem;
            }
        }
        
        @media (max-width: 480px) {
            .dashboard-header {
                padding: 1.5rem 1rem;
                margin-bottom: 1.5rem;
            }
            
            .kpi-grid {
                grid-template-columns: 1fr;
                gap: 0.6rem;
            }
            
            .indicator-label {
                font-size: 0.7rem;
            }
            
            .indicator-value {
                font-size: 1.2rem;
            }
            
            .selection-info {
                padding: 1rem;
                margin-bottom: 1.5rem;
            }
        }
    </style>
""", unsafe_allow_html=True)


# ---------- Chart savers with timestamped subfolder ---------- #

def _safe_filename(title: str, subdir: str) -> str:
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    return os.path.join(subdir, f"{safe_title}.png")


def save_price_trends_image(df_filtered: pd.DataFrame, selected_symbols: list[str], subdir: str):
    """Save normalized price trends to timestamped subfolder."""
    if df_filtered.empty or not selected_symbols:
        return
    path = _safe_filename("Price_Trends_Normalized_to_100", subdir)
    try:
        plt.figure(figsize=(14, 6))
        for symbol in selected_symbols:
            sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
            if not sym_data.empty:
                first_price = sym_data["adjusted_close"].iloc[0]
                normalized = sym_data["adjusted_close"] / first_price * 100
                plt.plot(sym_data["date"], normalized, label=symbol, linewidth=2.5)
        plt.title("Price Trends (Normalized to 100)", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontweight='bold')
        plt.ylabel("Normalized Price (base=100)", fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        st.sidebar.success(f"✅ Saved: {os.path.basename(path)}")
    except Exception:
        plt.close()


def save_cumulative_returns_image(df_filtered: pd.DataFrame, selected_symbols: list[str], subdir: str):
    """Save cumulative returns to timestamped subfolder."""
    if df_filtered.empty or not selected_symbols:
        return
    path = _safe_filename("Cumulative_Returns", subdir)
    try:
        plt.figure(figsize=(14, 6))
        for symbol in selected_symbols:
            sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
            if not sym_data.empty:
                returns = sym_data["adjusted_close"].pct_change()
                cum_returns = (1 + returns).cumprod() - 1
                plt.plot(sym_data["date"], cum_returns * 100, label=symbol, linewidth=2.5)
        plt.title("Cumulative Returns", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontweight='bold')
        plt.ylabel("Cumulative Return (%)", fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        st.sidebar.success(f"✅ Saved: {os.path.basename(path)}")
    except Exception:
        plt.close()


def save_risk_heatmap_image(risk_df: pd.DataFrame, subdir: str):
    """Save risk metrics heatmap to timestamped subfolder."""
    if risk_df.empty:
        return
    path = _safe_filename("Risk_Metrics_Heatmap", subdir)
    try:
        metrics = ["Annual Return", "Volatility", "Sharpe", "Max DD"]
        data = risk_df[metrics].copy()
        data["Annual Return"] *= 100
        data["Volatility"] *= 100
        data["Max DD"] *= 100

        plt.figure(figsize=(max(12, len(risk_df) * 1.2), 6))
        im = plt.imshow(data.T.values, aspect="auto", cmap="RdYlGn", interpolation='nearest')
        plt.colorbar(im, label="Value (%)")
        plt.yticks(range(len(metrics)), ["Annual Return (%)", "Volatility (%)", "Sharpe Ratio", "Max DD (%)"], fontsize=11)
        plt.xticks(range(len(risk_df["Symbol"])), risk_df["Symbol"], rotation=45, ha="right")
        plt.title("Risk Metrics Heatmap", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        st.sidebar.success(f"✅ Saved: {os.path.basename(path)}")
    except Exception:
        plt.close()


def save_rolling_vol_image(df_filtered: pd.DataFrame, selected_symbols: list[str], subdir: str):
    """Save rolling volatility to timestamped subfolder."""
    if df_filtered.empty or not selected_symbols:
        return
    path = _safe_filename("Rolling_20Day_Volatility", subdir)
    try:
        plt.figure(figsize=(14, 6))
        for symbol in selected_symbols:
            sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
            if not sym_data.empty and len(sym_data) > 20:
                returns = sym_data["adjusted_close"].pct_change()
                rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                plt.plot(sym_data["date"], rolling_vol * 100, label=symbol, linewidth=2.5)
        plt.title("Rolling 20-Day Annualized Volatility", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontweight='bold')
        plt.ylabel("Volatility (%)", fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        st.sidebar.success(f"✅ Saved: {os.path.basename(path)}")
    except Exception:
        plt.close()


def save_top_5_kpis_image(portfolio_metrics: dict, subdir: str):
    """Save Top 5 KPIs as horizontal bar chart to timestamped subfolder."""
    path = _safe_filename("Top_5_Portfolio_KPIs", subdir)
    try:
        kpis = {
            "Total Return (%)": portfolio_metrics.get("total_return", 0) * 100,
            "Annual Return (%)": portfolio_metrics.get("annualized_return", 0) * 100,
            "Volatility (%)": portfolio_metrics.get("volatility", 0) * 100,
            "Sharpe Ratio": portfolio_metrics.get("sharpe_ratio", 0),
            "VaR 95% (%)": portfolio_metrics.get("var_95", 0) * 100
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        kpi_names = list(kpis.keys())
        kpi_values = list(kpis.values())
        
        bars = ax.barh(kpi_names, kpi_values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'])
        ax.set_xlabel("Value", fontweight='bold', fontsize=12)
        ax.set_title("Top 5 Portfolio KPIs", fontsize=16, fontweight='bold', pad=20)
        
        for bar, val in zip(bars, kpi_values):
            width = bar.get_width()
            ax.text(width + (0.01 if val >= 0 else -0.05 * abs(val)), bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', ha='left' if val >= 0 else 'right', va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        st.sidebar.success(f"✅ Saved: {os.path.basename(path)}")
    except Exception:
        pass


# ---------- Data loading ---------- #

@st.cache_data(ttl=3600)
def load_data():
    return pa.load_portfolio_data()


df = load_data()

# ---------- Sidebar: symbol selection ---------- #

st.sidebar.title("📊 Configuration")
st.sidebar.markdown("---")

all_symbols = sorted(df["symbol"].unique().tolist())

if "selected_symbols" not in st.session_state:
    st.session_state.selected_symbols = all_symbols[:3] if len(all_symbols) >= 3 else all_symbols

selected_symbols = st.sidebar.multiselect(
    "📌 Select companies (up to 10)",
    all_symbols,
    default=st.session_state.selected_symbols,
    help="Choose stocks to analyze. Metrics update automatically.",
)

if len(selected_symbols) > 10:
    st.sidebar.warning("⚠️ Maximum 10 companies allowed. Showing first 10.")
    selected_symbols = selected_symbols[:10]

st.session_state.selected_symbols = selected_symbols

if not selected_symbols:
    st.warning("⚠️ Please select at least one stock to begin analysis.")
    st.stop()

# ---------- Sidebar: date range ---------- #

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Analysis Period")
min_date = df["date"].min().date()
max_date = df["date"].max().date()
default_start = max_date - timedelta(days=365 * 5)

col_start, col_end = st.sidebar.columns(2)

with col_start:
    start_date = st.date_input(
        "From",
        default_start,
        min_value=min_date,
        max_value=max_date,
    )

with col_end:
    end_date = st.date_input(
        "To",
        max_date,
        min_value=min_date,
        max_value=max_date,
    )

if start_date > end_date:
    st.sidebar.error("❌ Start date must be before end date.")
    st.stop()

df_filtered = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (df["symbol"].isin(selected_symbols))
].copy()

if df_filtered.empty:
    st.error("❌ No data available for selected symbols and date range.")
    st.stop()

num_selected = len(selected_symbols)

# ---------- Dashboard Header ---------- #

st.markdown(f"""
    <div class="dashboard-header">
        <h1>💼 Institutional Portfolio Analytics</h1>
        <p>📊 Analyzing {num_selected} {'stock' if num_selected == 1 else 'stocks'} from {start_date} to {end_date}</p>
    </div>
""", unsafe_allow_html=True)

# Selection info with high contrast
companies_html = " ".join([f'<span class="company-badge">{s}</span>' for s in selected_symbols])
st.markdown(f"""
    <div class="selection-info">
        <strong>✅ Selected Companies:</strong>
        <div class="companies-list">
            {companies_html}
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------- Portfolio KPIs ---------- #

st.markdown("<div class='section-header'>📊 Key Performance Indicators</div>", unsafe_allow_html=True)

portfolio_metrics = pa.calculate_portfolio_metrics(
    df_filtered, start_date=str(start_date), end_date=str(end_date)
)

BENCH = getattr(pa, "BENCHMARK_SYMBOL", "SPY")
bench_df = df_filtered[df_filtered["symbol"] == BENCH].sort_values("date")
benchmark_returns = None
if not bench_df.empty and len(bench_df) > 1:
    bp = bench_df["adjusted_close"].values
    b_ret = np.diff(bp) / bp[:-1]
    benchmark_returns = pd.Series(b_ret, index=bench_df["date"].iloc[1:])

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
        delta=f"{portfolio_metrics['annualized_return'] * 100:.2f}% annual",
    )

with kpi2:
    st.metric(
        "Volatility",
        f"{portfolio_metrics['volatility'] * 100:.2f}%",
        help="Annualized price fluctuation"
    )

with kpi3:
    sharpe = portfolio_metrics.get("sharpe_ratio", 0.0)
    st.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Risk-adjusted return")

with kpi4:
    beta_val = portfolio_metrics.get("beta")
    st.metric("Portfolio Beta", f"{beta_val:.2f}" if beta_val is not None else "N/A")

with kpi5:
    alpha_val = portfolio_metrics.get("alpha")
    st.metric("Portfolio Alpha", f"{alpha_val:.2f}" if alpha_val is not None else "N/A")

kpi6, kpi7, kpi8, kpi9, kpi10 = st.columns(5)

with kpi6:
    var_val = portfolio_metrics.get("var_95")
    st.metric("VaR 95%", f"{var_val * 100:.2f}%" if var_val is not None else "N/A")

first_symbol = selected_symbols[0]
fund = pa.get_fundamentals(first_symbol)

with kpi7:
    pe = fund.get("pe_ratio")
    st.metric(f"{first_symbol} P/E", f"{pe:.2f}" if pe is not None else "N/A")

with kpi8:
    eps = fund.get("eps")
    st.metric(f"{first_symbol} EPS", f"${eps:.2f}" if eps is not None else "N/A")

with kpi9:
    dy = fund.get("dividend_yield")
    st.metric("Div. Yield", f"{dy * 100:.2f}%" if dy is not None else "N/A")

with kpi10:
    mc = fund.get("market_cap")
    st.metric("Market Cap", f"${mc/1e9:.1f}B" if mc is not None else "N/A")

st.markdown("---")

# Save Top 5 KPIs chart
save_top_5_kpis_image(portfolio_metrics, CHARTS_SUBDIR)

# ---------- Performance Range Indicators (RESPONSIVE) ---------- #

st.markdown("<div class='section-header'>🎯 Performance Range Indicators</div>", unsafe_allow_html=True)

price_trends = {}
cum_returns = {}

for symbol in selected_symbols:
    sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
    if not sym_data.empty:
        first_price = sym_data["adjusted_close"].iloc[0]
        normalized = sym_data["adjusted_close"] / first_price * 100
        price_trends[symbol] = {
            "min": normalized.min(),
            "max": normalized.max(),
            "current": normalized.iloc[-1]
        }
        
        returns = sym_data["adjusted_close"].pct_change()
        cum_ret = (1 + returns).cumprod() - 1
        cum_returns[symbol] = {
            "min": cum_ret.min() * 100,
            "max": cum_ret.max() * 100,
            "current": cum_ret.iloc[-1] * 100
        }

ind_col1, ind_col2 = st.columns(2)

with ind_col1:
    st.markdown('<div class="subheader-text">Price Trends (Normalized to 100)</div>', unsafe_allow_html=True)
    if price_trends:
        overall_min = min([v["min"] for v in price_trends.values()])
        overall_max = max([v["max"] for v in price_trends.values()])
        overall_current = sum([v["current"] for v in price_trends.values()]) / len(price_trends)
        
        col_min, col_max, col_cur = st.columns(3)
        with col_min:
            st.markdown(f"""
            <div class="indicator-box">
                <div class="indicator-label">📉 Minimum</div>
                <div class="indicator-value">{overall_min:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_max:
            st.markdown(f"""
            <div class="indicator-box">
                <div class="indicator-label">📈 Maximum</div>
                <div class="indicator-value">{overall_max:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_cur:
            st.markdown(f"""
            <div class="indicator-box">
                <div class="indicator-label">💹 Current</div>
                <div class="indicator-value">{overall_current:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

with ind_col2:
    st.markdown('<div class="subheader-text">Cumulative Returns</div>', unsafe_allow_html=True)
    if cum_returns:
        overall_min = min([v["min"] for v in cum_returns.values()])
        overall_max = max([v["max"] for v in cum_returns.values()])
        overall_current = sum([v["current"] for v in cum_returns.values()]) / len(cum_returns)
        
        col_min, col_max, col_cur = st.columns(3)
        with col_min:
            st.markdown(f"""
            <div class="indicator-box">
                <div class="indicator-label">📉 Minimum</div>
                <div class="indicator-value">{overall_min:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_max:
            st.markdown(f"""
            <div class="indicator-box">
                <div class="indicator-label">📈 Maximum</div>
                <div class="indicator-value">{overall_max:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_cur:
            st.markdown(f"""
            <div class="indicator-box">
                <div class="indicator-label">💹 Current</div>
                <div class="indicator-value">{overall_current:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ---------- Charts: Price Trends & Cumulative Returns ---------- #

st.markdown("<div class='section-header'>📊 Performance Charts</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="subheader-text">Price Trends (Normalized to 100)</div>', unsafe_allow_html=True)
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
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Price: %{y:.2f}<extra></extra>',
                )
            )
    fig_price.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Normalized Price (base=100)",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig_price, use_container_width=True)
    save_price_trends_image(df_filtered, selected_symbols, CHARTS_SUBDIR)
    
    st.markdown("""
    <div class="chart-description">
        <strong>📌 What it shows:</strong> Tracks how stock prices move relative to a baseline of 100 at the start of the period. 
        A value of 110 means the stock is 10% above its starting price. Useful for comparing relative performance of multiple stocks 
        with different price scales.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="subheader-text">Cumulative Returns</div>', unsafe_allow_html=True)
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
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>',
                )
            )
    fig_cum.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig_cum, use_container_width=True)
    save_cumulative_returns_image(df_filtered, selected_symbols, CHARTS_SUBDIR)
    
    st.markdown("""
    <div class="chart-description">
        <strong>📌 What it shows:</strong> Displays the total percentage gain or loss from the start of the period, accounting for 
        all price changes. A 25% return means every dollar invested grew to $1.25. Shows how much wealth was created or destroyed 
        over time.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- Risk Metrics Heatmap ---------- #

st.markdown("<div class='section-header'>⚠️ Risk Analysis</div>", unsafe_allow_html=True)

st.markdown('<div class="subheader-text">Risk Metrics Heatmap</div>', unsafe_allow_html=True)

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
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>',
        )
    )
    fig_heatmap.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    save_risk_heatmap_image(risk_df, CHARTS_SUBDIR)
    
    st.markdown("""
    <div class="chart-description">
        <strong>📌 What it shows:</strong> Color-coded comparison of risk metrics across selected stocks. Green indicates better values 
        (higher returns, lower volatility, higher Sharpe). Helps identify which stocks offer the best risk-adjusted returns.
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Not enough data to compute risk metrics.")

st.markdown("---")

# ---------- Top Movers ---------- #

st.markdown("<div class='section-header'>🏆 Top Performers</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)
label_suffix = f"Top {top_n}" if num_selected >= 5 else f"All {num_selected}"

with col3:
    st.markdown(f'<div class="subheader-text">🔝 {label_suffix} Gainers</div>', unsafe_allow_html=True)
    if not gainers.empty:
        gainers_display = gainers[
            ["symbol", "total_return", "annualized_return", "sharpe_ratio", "volatility"]
        ].copy()
        gainers_display.columns = ["Symbol", "Total Return", "Annual Return", "Sharpe", "Volatility"]
        gainers_display["Total Return"] = gainers_display["Total Return"].apply(lambda x: f"{x*100:.2f}%")
        gainers_display["Annual Return"] = gainers_display["Annual Return"].apply(lambda x: f"{x*100:.2f}%")
        gainers_display["Sharpe"] = gainers_display["Sharpe"].apply(lambda x: f"{x:.2f}")
        gainers_display["Volatility"] = gainers_display["Volatility"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(gainers_display, use_container_width=True, hide_index=True)
    else:
        st.info("No data available.")

with col4:
    st.markdown(f'<div class="subheader-text">🔽 {label_suffix} Losers</div>', unsafe_allow_html=True)
    if not losers.empty:
        losers_display = losers[
            ["symbol", "total_return", "annualized_return", "sharpe_ratio", "volatility"]
        ].copy()
        losers_display.columns = ["Symbol", "Total Return", "Annual Return", "Sharpe", "Volatility"]
        losers_display["Total Return"] = losers_display["Total Return"].apply(lambda x: f"{x*100:.2f}%")
        losers_display["Annual Return"] = losers_display["Annual Return"].apply(lambda x: f"{x*100:.2f}%")
        losers_display["Sharpe"] = losers_display["Sharpe"].apply(lambda x: f"{x:.2f}")
        losers_display["Volatility"] = losers_display["Volatility"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(losers_display, use_container_width=True, hide_index=True)
    else:
        st.info("No data available.")

st.markdown("---")

# ---------- Rolling Volatility ---------- #

st.markdown("<div class='section-header'>📈 Volatility Trends</div>", unsafe_allow_html=True)

st.markdown('<div class="subheader-text">Rolling 20-Day Volatility Analysis</div>', unsafe_allow_html=True)
fig_vol = go.Figure()
for symbol in selected_symbols:
    sym_data = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
    if not sym_data.empty and len(sym_data) > 20:
        returns = sym_data["adjusted_close"].pct_change()
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        fig_vol.add_trace(
            go.Scatter(
                x=sym_data["date"],
                y=rolling_vol * 100,
                name=symbol,
                mode="lines",
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>',
            )
        )

fig_vol.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Annualized Volatility (%)",
    hovermode="x unified",
    template="plotly_white",
)
st.plotly_chart(fig_vol, use_container_width=True)
save_rolling_vol_image(df_filtered, selected_symbols, CHARTS_SUBDIR)

st.markdown("""
<div class="chart-description">
    <strong>📌 What it depicts:</strong> Shows how price volatility (variability/risk) changes over time using a 20-day rolling window. 
    Peaks indicate periods of high market uncertainty and larger price swings; valleys show calm periods with smaller movements. 
    <strong>Higher volatility = Higher risk but also higher potential returns.</strong> Useful for timing entries/exits and understanding 
    market stress periods.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- Comprehensive Risk Metrics Table ---------- #

st.markdown("<div class='section-header'>📋 Comprehensive Risk Metrics</div>", unsafe_allow_html=True)

st.markdown('<div class="subheader-text">Detailed Risk Analysis for All Selected Companies</div>', unsafe_allow_html=True)

all_risk_rows = []
for symbol in selected_symbols:
    result = pa.get_period_returns(
        df_filtered,
        symbol,
        start_date=str(start_date),
        end_date=str(end_date),
        benchmark_returns=benchmark_returns,
    )
    if result:
        all_risk_rows.append({
            "Company": symbol,
            "Total Return (%)": f"{result['total_return']*100:.2f}%",
            "Annual Return (%)": f"{result['annualized_return']*100:.2f}%",
            "Volatility (%)": f"{result['volatility']*100:.2f}%",
            "Sharpe Ratio": f"{result['sharpe_ratio']:.2f}",
            "Beta": f"{result.get('beta', np.nan):.2f}" if result.get('beta') is not None else "N/A",
            "Alpha": f"{result.get('alpha', np.nan):.2f}" if result.get('alpha') is not None else "N/A",
            "Max Drawdown (%)": f"{result['max_drawdown']*100:.2f}%",
            "VaR 95% (%)": f"{result.get('var_95', 0)*100:.2f}%" if result.get('var_95') is not None else "N/A",
        })

all_risk_df = pd.DataFrame(all_risk_rows)

if not all_risk_df.empty:
    st.dataframe(all_risk_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="chart-description">
        <strong>📌 Metric Definitions:</strong>
        <ul style="margin-left: 1.5rem; margin-top: 0.8rem;">
            <li><strong>Total Return:</strong> Overall percentage gain/loss over the period</li>
            <li><strong>Annual Return:</strong> Annualized return rate for comparison</li>
            <li><strong>Volatility:</strong> Standard deviation of returns; higher = riskier</li>
            <li><strong>Sharpe Ratio:</strong> Risk-adjusted return; higher = better (>1 is good, >2 is excellent)</li>
            <li><strong>Beta:</strong> Sensitivity to market moves; <1 = less volatile than market</li>
            <li><strong>Alpha:</strong> Excess return vs benchmark (SPY)</li>
            <li><strong>Max Drawdown:</strong> Largest peak-to-trough decline in value</li>
            <li><strong>VaR 95%:</strong> Worst expected daily loss at 95% confidence</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No data available for risk metrics.")

st.markdown("---")

# Sidebar shows chart export summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**📁 Charts saved to:**")
st.sidebar.markdown(f"`{CHARTS_SUBDIR}`")
st.sidebar.markdown("""**Files:**
- Price_Trends_Normalized_to_100.png
- Cumulative_Returns.png
- Risk_Metrics_Heatmap.png
- Rolling_20Day_Volatility.png
- Top_5_Portfolio_KPIs.png""")

# Footer
st.markdown("""
    <div class="footer-text">
        <strong>Institutional Portfolio Analytics Dashboard</strong><br>
        <small>Last updated: January 2026 | Data refreshes on each page reload | Charts exported to timestamped subfolder</small>
    </div>
""", unsafe_allow_html=True)