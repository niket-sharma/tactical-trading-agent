"""Dataclasses used by the discrete trade engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    size_fraction: float
    shares: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    max_price: float = 0.0
    days_held: int = 0


@dataclass
class TradeResult:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_pct: float
    reason: str
