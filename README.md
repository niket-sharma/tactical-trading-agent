# Tactical Trading Agent (QQQ / TQQQ / SQQQ)

Research-grade backtester and signal generator for allocating between TQQQ (3x long NASDAQ) and SQQQ (3x inverse 3x), driven by QQQ price action. Supports rule-based, ML, and hybrid modes. **For research/education only. Not investment advice.**

## Features
- Data loader (yfinance) with caching.
- Indicator pipeline (SMAs, returns, volatility, momentum).
- Policies: Rule, ML (next-day QQQ direction), Hybrid (blend).
- Backtester with equity curves and metrics (CAGR, Sharpe, Max DD, Vol).
- Daily signal export (allocations TQQQ vs SQQQ).
- Python API for programmatic use.
- Streamlit UI for non-coders.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI Backtest
```bash
# Rule / ML / Hybrid
python -m trading_agent.main_backtest --mode rule
python -m trading_agent.main_backtest --mode ml --model-type logistic_regression
python -m trading_agent.main_backtest --mode ml --model-type random_forest
python -m trading_agent.main_backtest --mode ml --model-type xgboost
python -m trading_agent.main_backtest --mode hybrid --model-type random_forest

# Save daily signals to CSV
python -m trading_agent.main_backtest --mode rule --save-signals signals.csv

# Optional date range
python -m trading_agent.main_backtest --mode rule --start-date 2014-01-01 --end-date 2024-12-31
```

## Python API
```python
from trading_agent.api import backtest, get_signals

output = backtest(mode="rule", start_date="2014-01-01", end_date=None, save_signals_path="signals.csv")
results = output["results"]     # DataFrame with equity curve + weights (weight_tqqq, weight_sqqq)
metrics = output["metrics"]     # dict with cagr, sharpe, max_dd, vol

# Specify model_type for ML/Hybrid
output = backtest(mode="ml", model_type="random_forest", start_date="2020-01-01", end_date="2023-12-31")
signals = get_signals(mode="hybrid", model_type="xgboost", start_date="2020-01-01", end_date="2023-12-31")
```

## Streamlit App
```bash
streamlit run trading_agent/streamlit_app.py
```
Select mode, dates, run the backtest, view metrics and equity curves, download signals CSV.

## Tests
```bash
python -m pytest
```
- Unit tests use synthetic data (offline).
- `tests/test_integration_live.py` will use cached data or try to download via yfinance; skips if unavailable.

## Safety
- Leveraged ETFs are path-dependent and risky. Do not use this code for live trading without thorough research, validation, and professional advice.
 ##Test Ping