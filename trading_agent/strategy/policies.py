"""Policy implementations for rule, ML, and hybrid strategies."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from trading_agent.strategy.config import HybridConfig, MLConfig, StrategyConfig
from trading_agent.strategy.models import predict_direction_proba
from trading_agent.strategy.rules import compute_rule_allocation


class BasePolicy:
    """Interface for allocation policies."""

    def get_allocation(self, row: pd.Series, date, state) -> Tuple[float, float]:
        raise NotImplementedError


class RuleBasedPolicy(BasePolicy):
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.prev_weights = (0.5, 0.5)

    def get_allocation(self, row: pd.Series, date, state) -> Tuple[float, float]:
        weights = compute_rule_allocation(row, self.config, prev_weights=self.prev_weights)
        self.prev_weights = weights
        return weights


class MLBasedPolicy(BasePolicy):
    def __init__(self, model, feature_cols, ml_config: MLConfig):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.ml_config = ml_config

    def get_allocation(self, row: pd.Series, date, state) -> Tuple[float, float]:
        try:
            x_row = row[self.feature_cols]
        except KeyError:
            return 0.5, 0.5

        if x_row.isna().any():
            return 0.5, 0.5

        x_row_df = x_row.to_frame().T
        p_up = predict_direction_proba(self.model, x_row_df)
        threshold = self.ml_config.threshold
        if p_up > threshold:
            w_t, w_s = 0.8, 0.2
        elif p_up < 1 - threshold:
            w_t, w_s = 0.2, 0.8
        else:
            w_t, w_s = 0.5, 0.5

        total = w_t + w_s
        return w_t / total, w_s / total


class HybridPolicy(BasePolicy):
    def __init__(
        self,
        rule_policy: RuleBasedPolicy,
        ml_policy: MLBasedPolicy,
        hybrid_config: HybridConfig,
    ):
        self.rule_policy = rule_policy
        self.ml_policy = ml_policy
        self.hybrid_config = hybrid_config

    def get_allocation(self, row: pd.Series, date, state) -> Tuple[float, float]:
        w_t_rule, w_s_rule = self.rule_policy.get_allocation(row, date, state)
        w_t_ml, w_s_ml = self.ml_policy.get_allocation(row, date, state)

        alpha = self.hybrid_config.blend_alpha
        w_t = (1 - alpha) * w_t_rule + alpha * w_t_ml
        w_s = (1 - alpha) * w_s_rule + alpha * w_s_ml

        total = w_t + w_s
        if total == 0:
            return 0.5, 0.5
        return w_t / total, w_s / total
