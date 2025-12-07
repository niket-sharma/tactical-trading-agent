"""Rule-based allocation logic."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from trading_agent.strategy.config import StrategyConfig


def classify_trend(row: pd.Series, config: StrategyConfig) -> str:
    """Classify market regime using QQQ close vs. moving averages."""
    close = row.get("QQQ_Close")
    sma_50 = row.get("QQQ_SMA_50")
    sma_100 = row.get("QQQ_SMA_100")

    if any(pd.isna(val) for val in (close, sma_50, sma_100)):
        return "neutral"

    if close > sma_50 > sma_100:
        return "bull"
    if close < sma_50 < sma_100:
        return "bear"
    return "neutral"


def volatility_filter(row: pd.Series, config: StrategyConfig) -> float:
    """Return damping factor based on recent volatility."""
    vol = row.get("QQQ_vol_20")
    if pd.isna(vol):
        return 1.0
    if vol > config.vol_cutoff:
        return 0.5
    return 1.0


def compute_rule_allocation(
    row: pd.Series,
    config: StrategyConfig,
    prev_weights: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Compute TQQQ/SQQQ weights based on trend and volatility."""
    prev_wt = prev_weights or (0.5, 0.5)
    trend = classify_trend(row, config)
    if trend == "bull":
        w_t, w_s = config.bull_bias, 1 - config.bull_bias
    elif trend == "bear":
        w_t, w_s = 1 - config.bear_bias, config.bear_bias
    else:
        w_t, w_s = config.neutral_bias, 1 - config.neutral_bias

    damp = volatility_filter(row, config)
    if damp < 1.0:
        w_t = 0.5 + (w_t - 0.5) * damp
        w_s = 0.5 + (w_s - 0.5) * damp

    if any(math.isnan(val) for val in (w_t, w_s)):
        w_t, w_s = prev_wt

    w_t = float(np.clip(w_t, 0.0, 1.0))
    w_s = float(np.clip(w_s, 0.0, 1.0))
    total = w_t + w_s
    if total == 0:
        return 0.5, 0.5
    return w_t / total, w_s / total
