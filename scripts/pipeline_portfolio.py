"""
pipeline_portfolio.py

Orchestration script for portfolio analytics pipeline.
Stages:
1. Download/load 10-year data (concurrent, ~2-3 min)
2. Validate data quality
3. Display summary statistics
"""

import os
import sys
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loading_portfolio import load_portfolio_data, main as download_data
from portfolio_analytics import calculate_portfolio_metrics, get_top_gainers_losers


def main():
    print("\n" + "="*80)
    print("🚀 INSTITUTIONAL PORTFOLIO ANALYTICS PIPELINE")
    print("="*80)
    
    # Stage 1: Load/Download Data
    print("\n📊 STAGE 1: Data Loading")
    print("-" * 80)
    start_time = datetime.now()
    
    df = load_portfolio_data()
    
    if df.empty:
        print("❌ Pipeline failed: No data available")
        return False
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✅ Data loaded in {elapsed:.1f}s")
    
    # Stage 2: Data Validation
    print("\n📊 STAGE 2: Data Validation")
    print("-" * 80)
    print(f"   Rows: {len(df):,}")
    print(f"   Symbols: {df['symbol'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Columns: {', '.join(df.columns)}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicates: {df.duplicated().sum()}")
    
    # Stage 3: Summary Analytics
    print("\n📊 STAGE 3: Summary Analytics")
    print("-" * 80)
    
    start_date = df['date'].min().date()
    end_date = df['date'].max().date()
    
    metrics = calculate_portfolio_metrics(df, str(start_date), str(end_date))
    
    print(f"   Portfolio Total Return: {metrics['total_return']*100:.2f}%")
    print(f"   Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"   Volatility: {metrics['volatility']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Beta: {metrics['beta']:.2f}" if not pd.isna(metrics['beta']) else "   Beta: N/A")
    print(f"   Alpha: {metrics['alpha']:.2f}" if not pd.isna(metrics['alpha']) else "   Alpha: N/A")
    
    gainers, losers = get_top_gainers_losers(df, "total_return", n=5, start_date=str(start_date), end_date=str(end_date))
    
    print(f"\n   Top 5 Gainers:")
    if not gainers.empty:
        for _, row in gainers.iterrows():
            print(f"      {row['symbol']}: {row['total_return']*100:+.2f}%")
    
    print(f"\n   Top 5 Losers:")
    if not losers.empty:
        for _, row in losers.iterrows():
            print(f"      {row['symbol']}: {row['total_return']*100:+.2f}%")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE - Ready for Dashboard Launch")
    print("="*80)
    print("\n📊 Next: streamlit run streamlit_app/portfolio_dashboard.py\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)