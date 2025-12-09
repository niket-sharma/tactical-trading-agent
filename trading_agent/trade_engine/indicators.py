"""Indicator helpers for the discrete trade engine."""

from __future__ import annotations

import pandas as pd

from trading_agent.trade_engine.config import TradeSystemConfig


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_trade_indicators(price_df: pd.DataFrame, config: TradeSystemConfig) -> pd.DataFrame:
    """Return indicator DataFrame focused on QQQ trend and pullbacks."""
    trend_ticker = config.trend_ticker
    qqq_close = price_df[(trend_ticker, "Close")].copy()
    qqq_open = price_df[(trend_ticker, "Open")].copy()

    ema = qqq_close.ewm(span=config.trend_ema_period, adjust=False).mean()
    rsi = compute_rsi(qqq_close, config.rsi_period)

    window = max(config.trend_ema_period, 10)
    rolling_high = qqq_close.rolling(window=window, min_periods=window // 2).max()
    rolling_low = qqq_close.rolling(window=window, min_periods=window // 2).min()

    pullback_from_high = (qqq_close / rolling_high) - 1
    bounce_from_low = (qqq_close / rolling_low) - 1

    indicators = pd.DataFrame(
        {
            "QQQ_Open": qqq_open,
            "QQQ_Close": qqq_close,
            "QQQ_EMA": ema,
            "QQQ_RSI": rsi,
            "QQQ_pullback": pullback_from_high,
            "QQQ_bounce": bounce_from_low,
        }
    )
    return indicators
