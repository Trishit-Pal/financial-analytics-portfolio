# 💼 Institutional Portfolio Analytics System

**Enterprise-grade multi-asset portfolio analysis platform** leveraging institutional-quality quantitative metrics and risk analytics.

## ✨ Features

### 📊 Performance Analytics
- **Total Return**: Overall profit/loss from initial investment
- **Time-Weighted Return (TWR)**: Return metric accounting for cash flow timing
- **Annualized Return**: Normalized returns for fair period comparison
- **Cumulative Returns**: Running total return visualization

### ⚠️ Risk Metrics
- **Volatility (Annualized)**: Price variability = √252 × daily std dev
- **Sharpe Ratio**: Risk-adjusted return = (Annual Return - Risk-Free Rate) / Volatility
- **Beta**: Systematic risk = Covariance(stock, market) / Variance(market)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Rolling Volatility**: 20-day rolling annualized volatility

### 🏆 Portfolio Features
- **50+ Large-Cap Stocks**: Finance, tech, energy, healthcare, consumer sectors
- **Top 5 Movers**: Gainers and losers by period and risk-adjusted metrics
- **Multi-Symbol Selection**: Compare up to 50 stocks simultaneously
- **Date Range Filtering**: Analyze any 10-year historical period
- **Normalized Price Charts**: Base=100 for easy visual comparison
- **Risk Heatmaps**: Color-coded risk metric visualization

### 🔄 Data & Deployment
- **Alpha Vantage API**: Live real-time data with full-history (`outputsize=full`)
- **Intelligent Fallback**: Local JSON cache + CSV fallback when API unavailable
- **Rate-Limit Handling**: 0.25s delays respect API free tier (12 calls/minute)
- **10-Year Data**: Full historical data from Alpha Vantage [web:94]
- **GitHub Pages**: Static portfolio site with metrics explanations
- **Streamlit Cloud**: Interactive live dashboard (free hosting)
