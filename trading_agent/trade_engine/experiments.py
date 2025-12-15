"""Parameter sweep utilities for the discrete trade system."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Dict, List

import pandas as pd

from trading_agent.trade_engine.config import TradeSystemConfig
from trading_agent.trade_engine.backtest_trades import backtest_trade_system
from trading_agent.strategy.metrics import (
    compute_cagr,
    compute_daily_volatility,
    compute_max_drawdown,
    compute_sharpe,
)


def _default_param_grid() -> Dict[str, List[Any]]:
    """Return a compact default sweep grid."""
    return {
        "rsi_long_threshold": [50.0, 55.0],
        "rsi_short_threshold": [50.0, 55.0],
        "pullback_max_pct": [0.05, 0.08],
        "stop_loss_pct": [0.08, 0.10],
        "take_profit_max_pct": [0.12, 0.15],
        "trailing_stop_pct": [0.05, 0.07],
        "max_position_fraction": [0.20, 0.25],
        "typical_position_fraction": [0.10, 0.15],
    }


def _compute_metrics(equity_df: pd.DataFrame, trades: List[Any]) -> Dict[str, Any]:
    """Compute summary metrics for a single backtest run."""
    daily_return = equity_df["total_equity"].pct_change().fillna(0.0)
    cagr = compute_cagr(equity_df["total_equity"])
    sharpe = compute_sharpe(daily_return)
    max_dd = compute_max_drawdown(equity_df["total_equity"])
    vol = compute_daily_volatility(daily_return)

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "daily_vol": vol,
        "trades": len(trades),
        "final_equity": equity_df["total_equity"].iloc[-1],
        "start": equity_df.index.min().date().isoformat(),
        "end": equity_df.index.max().date().isoformat(),
    }


def run_param_sweep(
    price_df: pd.DataFrame,
    base_config: TradeSystemConfig,
    param_grid: Dict[str, List[Any]] | None = None,
    max_runs: int | None = None,
) -> pd.DataFrame:
    """Sweep over provided parameters and collect performance metrics.

    Args:
        price_df: MultiIndex price DataFrame (QQQ/TQQQ/SQQQ).
        base_config: Starting configuration to clone per run.
        param_grid: Dict of parameter -> list of values. Uses defaults if None.
        max_runs: Optional cap on number of combinations (for quick smoke tests).

    Returns:
        DataFrame with one row per run, including parameters and metrics.
    """
    grid = param_grid or _default_param_grid()
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    if max_runs is not None:
        combos = combos[:max_runs]

    rows: List[Dict[str, Any]] = []
    for idx, values in enumerate(combos):
        params = dict(zip(keys, values))
        cfg = replace(base_config, **params)
        # Ensure min take-profit stays below max; if not, adjust minimally.
        if cfg.take_profit_min_pct >= cfg.take_profit_max_pct:
            cfg = replace(cfg, take_profit_min_pct=cfg.take_profit_max_pct * 0.6)

        equity_df, trades = backtest_trade_system(price_df, cfg)
        if equity_df.empty:
            rows.append({**params, "run": idx, "error": "empty_equity"})
            continue

        metrics = _compute_metrics(equity_df, trades)
        rows.append({"run": idx, **params, **metrics, "error": ""})

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        sort_cols = [c for c in ["cagr", "sharpe", "max_dd", "daily_vol"] if c in result_df.columns]
        asc = [False, False, True, True][: len(sort_cols)]
        if sort_cols:
            result_df.sort_values(by=sort_cols, ascending=asc, inplace=True)

        # Reorder columns for easier CSV analysis
        ordered_params = [col for col in keys if col in result_df.columns]
        metric_cols = ["cagr", "sharpe", "max_dd", "daily_vol", "trades", "final_equity"]
        bookkeeping = ["run", "start", "end", "error"]
        col_order = ["run"] + ordered_params + metric_cols + [c for c in bookkeeping if c in result_df.columns and c != "run"]
        result_df = result_df[[col for col in col_order if col in result_df.columns]]
    return result_df
