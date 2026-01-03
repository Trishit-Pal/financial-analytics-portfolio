"""
pipeline_portfolio.py

Run complete portfolio analytics pipeline:
1. Fetch portfolio data via yfinance
2. Compute performance & risk metrics
"""

from scripts import data_loading_portfolio as dlp
from scripts import portfolio_analytics as pa


def main(max_symbols: int = 50):
    print("\n" + "=" * 60)
    print("ENTERPRISE PORTFOLIO ANALYTICS PIPELINE")
    print("=" * 60)

    df = dlp.main(max_symbols=max_symbols)
    df_analysis, metrics, gainers, losers = pa.main()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print("=" * 60)
    print("Ready for dashboard: streamlit run streamlit_app/portfolio_dashboard.py")
    return df, metrics, gainers, losers


if __name__ == "__main__":
    main(max_symbols=50)
