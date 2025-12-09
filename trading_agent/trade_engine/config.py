"""Configuration for the discrete trade engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeSystemConfig:
    """Parameters controlling the rule-based TQQQ/SQQQ trade engine."""

    initial_equity: float = 10000.0

    # Risk / position sizing
    max_position_fraction: float = 0.25  # max 25% of account in a single trade
    typical_position_fraction: float = 0.15  # default 15% position

    # Trend filter (daily QQQ)
    trend_ema_period: int = 20  # 20-day EMA for trend
    trend_ticker: str = "QQQ"

    # Pullback filter
    pullback_min_pct: float = 0.01  # 1% pullback
    pullback_max_pct: float = 0.05  # allow deeper pullbacks/bounces for entries

    # Mean reversion "RSI-like" filter (daily RSI)
    rsi_period: int = 14
    rsi_long_threshold: float = 55.0  # oversold threshold for TQQQ entries
    rsi_short_threshold: float = 50.0  # overbought threshold for SQQQ entries

    # Stop-loss & take-profit
    stop_loss_pct: float = 0.10  # 10% stop-loss
    take_profit_min_pct: float = 0.05  # 5% minimum profit target
    take_profit_max_pct: float = 0.15  # 15% maximum profit target
    trailing_stop_pct: float = 0.07  # optional trailing stop (7%)

    # Trade management
    allow_overnight: bool = True  # keep positions overnight by default
    max_open_days: int = 10  # optional: close trade after N days

    # Symbols
    long_symbol: str = "TQQQ"
    short_symbol: str = "SQQQ"
