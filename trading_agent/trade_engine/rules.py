"""Rule set for the trend + mean reversion trade engine."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from trading_agent.trade_engine.config import TradeSystemConfig


def _pick_take_profit_pct(config: TradeSystemConfig) -> float:
    """Choose a take-profit target between min and max."""
    return (config.take_profit_min_pct + config.take_profit_max_pct) / 2.0


def determine_entry(row: pd.Series, config: TradeSystemConfig) -> Tuple[Optional[str], str]:
    """Return symbol to trade (TQQQ/SQQQ) or None if no entry."""
    if row.isnull().any():
        return None, "incomplete_indicators"

    close = row["QQQ_Close"]
    ema = row["QQQ_EMA"]
    rsi = row["QQQ_RSI"]
    pullback = row["QQQ_pullback"]
    bounce = row["QQQ_bounce"]

    in_uptrend = close > ema
    in_downtrend = close < ema

    # Long bias: uptrend + pullback + oversold RSI
    if in_uptrend and pullback <= -config.pullback_min_pct and pullback >= -config.pullback_max_pct:
        if rsi <= config.rsi_long_threshold:
            return config.long_symbol, "uptrend_pullback"

    # Short bias: downtrend + bounce + overbought RSI -> use SQQQ as inverse long
    if in_downtrend and bounce >= config.pullback_min_pct and bounce <= config.pullback_max_pct:
        if rsi >= config.rsi_short_threshold:
            return config.short_symbol, "downtrend_bounce"

    return None, "flat"


def compute_stops(entry_price: float, config: TradeSystemConfig) -> Tuple[float, float, float]:
    """Return stop_loss, take_profit, trailing_stop prices for a long position."""
    stop_loss = entry_price * (1 - config.stop_loss_pct)
    take_profit_pct = _pick_take_profit_pct(config)
    take_profit = entry_price * (1 + take_profit_pct)
    trailing_stop = entry_price * (1 - config.trailing_stop_pct) if config.trailing_stop_pct > 0 else np.nan
    return stop_loss, take_profit, trailing_stop
