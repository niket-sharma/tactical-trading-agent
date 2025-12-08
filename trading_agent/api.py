"""Public Python API for running backtests and exporting signals."""

from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd

from trading_agent.config import DEFAULT_END_DATE, DEFAULT_START_DATE
from trading_agent.main_backtest import build_policy, load_prices
from trading_agent.features.indicators import compute_indicators
from trading_agent.strategy.backtest import backtest_strategy
from trading_agent.strategy.metrics import (
    compute_cagr,
    compute_daily_volatility,
    compute_max_drawdown,
    compute_sharpe,
)


def backtest(
    mode: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = DEFAULT_END_DATE,
    save_signals_path: Optional[str] = None,
    model_type: str = "logistic_regression",
) -> Dict[str, Any]:
    """Run a backtest and return results + metrics.

    Returns:
        {
            "results": pd.DataFrame,
            "metrics": {"cagr": float, "sharpe": float, "max_dd": float, "vol": float},
        }
    """
    prices = load_prices(start_date, end_date)
    features = compute_indicators(prices)

    policy, _ = build_policy(mode, features, prices, model_type=model_type)
    results = backtest_strategy(prices, features, policy)
    metrics = {}
    if not results.empty:
        metrics = {
            "cagr": compute_cagr(results["total_value"]),
            "sharpe": compute_sharpe(results["total_return"]),
            "max_dd": compute_max_drawdown(results["total_value"]),
            "vol": compute_daily_volatility(results["total_return"]),
        }
        if save_signals_path:
            cols = [
                "weight_tqqq",
                "weight_sqqq",
                "total_value",
                "total_return",
                "lev_capital",
                "QQQ_buyhold_value",
                "TQQQ_buyhold_value",
                "SQQQ_buyhold_value",
            ]
            export_df = results[[c for c in cols if c in results.columns]].copy()
            export_df.to_csv(save_signals_path, index=True, date_format="%Y-%m-%d")

    return {"results": results, "metrics": metrics}


def get_signals(
    mode: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = DEFAULT_END_DATE,
    model_type: str = "logistic_regression",
) -> pd.DataFrame:
    """Convenience helper that returns the daily allocation signals."""
    output = backtest(mode=mode, start_date=start_date, end_date=end_date, model_type=model_type)
    return output["results"][["weight_tqqq", "weight_sqqq"]] if not output["results"].empty else pd.DataFrame()
