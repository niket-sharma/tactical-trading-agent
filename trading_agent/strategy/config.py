"""Strategy configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MODE_RULE = "rule"
MODE_ML = "ml"
MODE_HYBRID = "hybrid"


@dataclass
class StrategyConfig:
    ma_short: int = 20
    ma_long: int = 100
    bull_bias: float = 0.8
    bear_bias: float = 0.8
    neutral_bias: float = 0.5
    vol_cutoff: float = 0.03
    max_leveraged_exposure: float = 1.0
    mode: str = MODE_RULE


@dataclass
class MLConfig:
    train_start_date: str
    train_end_date: str
    val_start_date: str
    val_end_date: str
    model_type: str = "logistic_regression"
    threshold: float = 0.55


@dataclass
class HybridConfig:
    blend_alpha: float = 0.5
