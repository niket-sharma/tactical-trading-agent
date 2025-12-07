"""Performance metrics and visualization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity.empty:
        return float("nan")
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    return float(total_return ** (1 / years) - 1) if years > 0 else float("nan")


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def compute_daily_volatility(returns: pd.Series) -> float:
    return float(returns.std())


def count_allocation_changes(weights: pd.Series) -> int:
    """Count how often allocation materially shifts."""
    return int((weights.diff().abs() > 0.05).sum())


def plot_equity_curves(result_df: pd.DataFrame):
    """Plot equity curves for strategy and references."""
    plt.figure(figsize=(10, 6))
    plt.plot(result_df.index, result_df["total_value"], label="Strategy")
    plt.plot(result_df.index, result_df["QQQ_buyhold_value"], label="QQQ buy&hold")
    plt.plot(result_df.index, result_df["TQQQ_buyhold_value"], label="TQQQ buy&hold", alpha=0.6)
    plt.plot(result_df.index, result_df["SQQQ_buyhold_value"], label="SQQQ buy&hold", alpha=0.6)
    plt.legend()
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    return plt
