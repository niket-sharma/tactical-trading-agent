"""Configuration for portfolio-level allocations and explanations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    """Parameters controlling macro/swing classification and risk sizing."""

    ma_short: int = 50
    ma_long: int = 200
    swing_lookback: int = 10
    swing_threshold: float = 0.02

    base_risk: float = 0.4
    risk_step: float = 0.4
    max_leverage: float = 1.0

    core_qqq_weight: float = 0.50
    tqqq_hard_cap: float = 0.40
    tqqq_min_weight: float = 0.0
    sqqq_min_weight: float = 0.0
    cash_risk_off_floor: float = 0.20

    neutral_exposure: float = 0.25
    cash_floor: float = 0.1
