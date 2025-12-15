"""Simple portfolio policy for QQQ/TQQQ/SQQQ allocations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from trading_agent.strategy.portfolio_config import PortfolioConfig


def classify_macro_trend(row: pd.Series, config: PortfolioConfig) -> str:
    """Classify macro trend using short/long SMAs."""
    close = row.get("QQQ_Close")
    sma_short = row.get(f"QQQ_SMA_{config.ma_short}")
    sma_long = row.get(f"QQQ_SMA_{config.ma_long}")

    if any(pd.isna(val) for val in (close, sma_short, sma_long)):
        return "neutral"
    if close > sma_short > sma_long:
        return "bull"
    if close < sma_short < sma_long:
        return "bear"
    return "neutral"


def classify_swing_signal(row: pd.Series, config: PortfolioConfig) -> str:
    """Classify short-term swing using recent momentum."""
    mom_key = f"QQQ_mom_{config.swing_lookback}"
    momentum = row.get(mom_key) if mom_key in row else row.get("QQQ_mom_10")
    if pd.isna(momentum):
        return "neutral"
    if momentum > config.swing_threshold:
        return "upswing"
    if momentum < -config.swing_threshold:
        return "downswing"
    return "neutral"


class QQQPortfolioPolicy:
    """Lightweight policy to map regime -> QQQ/TQQQ/SQQQ/cash weights."""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.prev_weights: Dict[str, float] = {"TQQQ": 0.0, "SQQQ": 0.0, "QQQ": 0.0, "CASH": 1.0}
        self.last_context: Dict[str, object] = {}

    def get_allocation(
        self,
        row: pd.Series,
        prev_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        macro = classify_macro_trend(row, self.config)
        swing = classify_swing_signal(row, self.config)

        risk_flags = []
        qqq_close = row.get("QQQ_Close")
        sma_short = row.get(f"QQQ_SMA_{self.config.ma_short}")
        sma_long = row.get(f"QQQ_SMA_{self.config.ma_long}")
        sma50_slope = row.get("QQQ_SMA_50_slope_10")

        risk_off = False
        if pd.notna(sma50_slope) and sma50_slope < 0:
            risk_off = True
            risk_flags.append("Risk-off override: SMA50 slope is negative")
        if pd.notna(qqq_close) and pd.notna(sma_short) and qqq_close < sma_short:
            risk_off = True
            risk_flags.append("Risk-off override: QQQ below SMA50")

        direction_score = 0.0
        if macro == "bull":
            direction_score += 1.0
        elif macro == "bear":
            direction_score -= 1.0

        if swing == "upswing":
            direction_score += 0.5
        elif swing == "downswing":
            direction_score -= 0.5

        direction_score = float(np.clip(direction_score, -1.0, 1.0))

        # Start with core QQQ sleeve
        core_weight = max(0.0, min(1.0, self.config.core_qqq_weight))
        bucket = max(0.0, 1.0 - core_weight)
        weights: Dict[str, float] = {"TQQQ": 0.0, "SQQQ": 0.0, "QQQ": core_weight, "CASH": 0.0}

        risk_budget = min(self.config.max_leverage, self.config.base_risk + self.config.risk_step * abs(direction_score))
        risk_budget = max(0.0, min(bucket, risk_budget))

        if risk_off:
            t_weight = min(self.config.tqqq_min_weight, bucket)
            s_weight = min(self.config.sqqq_min_weight, bucket - t_weight)
            remaining = max(0.0, bucket - t_weight - s_weight)
            weights["TQQQ"] = t_weight
            weights["SQQQ"] = s_weight
            # Split remaining between extra QQQ and cash buffer
            weights["QQQ"] += remaining * 0.5
            weights["CASH"] = max(self.config.cash_floor, self.config.cash_risk_off_floor, remaining * 0.5)
            macro = "risk_off"
        else:
            if direction_score > 0:
                tqqq_share = 0.6 + 0.4 * abs(direction_score)
                t_alloc = risk_budget * tqqq_share
                q_alloc = risk_budget * (1 - tqqq_share)
                weights["TQQQ"] = t_alloc
                weights["QQQ"] += q_alloc
            elif direction_score < 0:
                weights["SQQQ"] = risk_budget
            else:
                neutral_alloc = min(self.config.neutral_exposure, risk_budget)
                weights["QQQ"] += neutral_alloc

            # If price is below SMA, damp TQQQ and lift cash
            if pd.notna(qqq_close) and pd.notna(sma_short) and qqq_close < sma_short and not risk_off:
                weights["TQQQ"] = min(weights["TQQQ"], risk_budget * 0.5)
                risk_flags.append("Risk-off override: QQQ below SMA50")

        # Enforce hard cap on TQQQ
        t_cap = max(0.0, min(1.0, self.config.tqqq_hard_cap))
        if weights["TQQQ"] > t_cap:
            overflow = weights["TQQQ"] - t_cap
            weights["TQQQ"] = t_cap
            weights["CASH"] += overflow

        min_cash = max(self.config.cash_floor, self.config.cash_risk_off_floor if risk_off else self.config.cash_floor)

        # Normalize to ensure total <= 1 and cash >= floor
        non_cash_total = weights["TQQQ"] + weights["SQQQ"] + weights["QQQ"]
        max_non_cash = max(0.0, 1.0 - min_cash)
        if non_cash_total > max_non_cash and non_cash_total > 0:
            scale = max_non_cash / non_cash_total
            weights["TQQQ"] *= scale
            weights["SQQQ"] *= scale
            weights["QQQ"] *= scale
            non_cash_total = weights["TQQQ"] + weights["SQQQ"] + weights["QQQ"]

        # Ensure QQQ never drops below core weight
        if weights["QQQ"] < core_weight:
            weights["QQQ"] = core_weight
            non_cash_total = weights["TQQQ"] + weights["SQQQ"] + weights["QQQ"]

        weights["CASH"] = max(min_cash, 1.0 - non_cash_total)

        self.prev_weights = weights
        self.last_context = {
            "macro": macro,
            "swing": swing,
            "direction_score": direction_score,
            "risk_flags": risk_flags,
            "sma50_slope": sma50_slope,
        }
        return weights
