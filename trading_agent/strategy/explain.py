"""Human-friendly explanations for portfolio allocations."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from trading_agent.strategy.portfolio_config import PortfolioConfig
from trading_agent.strategy.portfolio_policy import (
    QQQPortfolioPolicy,
    classify_macro_trend,
    classify_swing_signal,
)


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def describe_regime(row: pd.Series, config: PortfolioConfig) -> Tuple[str, str, str]:
    """Return macro, swing, and a short sentence describing the regime."""
    macro = classify_macro_trend(row, config)
    swing = classify_swing_signal(row, config)

    macro_txt = {
        "bull": "Bullish trend (price above key SMAs)",
        "bear": "Bearish trend (price below SMAs)",
        "risk_off": "Risk-off override (trend caution)",
        "neutral": "Neutral / mixed trend",
    }.get(macro, "Neutral / mixed trend")
    swing_txt = {"upswing": "short-term upswing", "downswing": "short-term downswing", "neutral": "range-bound"}.get(
        swing, "range-bound"
    )
    sentence = f"Macro: {macro_txt}. Short-term: {swing_txt}."
    return macro, swing, sentence


def format_allocation(weights: Dict[str, float]) -> str:
    parts = [f"{k}: {_format_pct(v)}" for k, v in weights.items()]
    return ", ".join(parts)


def describe_changes(current: Dict[str, float], previous: Optional[Dict[str, float]]) -> str:
    if not previous:
        return "Initiate positions per target weights."

    deltas = {}
    assets = ["TQQQ", "SQQQ", "QQQ", "CASH"]
    for asset in assets:
        deltas[asset] = current.get(asset, 0.0) - previous.get(asset, 0.0)

    moves = []
    for asset in assets:
        delta = deltas[asset]
        if abs(delta) < 0.0025:
            continue
        action = "Increase" if delta > 0 else "Reduce"
        moves.append(f"{action} {asset} by {_format_pct(abs(delta))}")

    return "; ".join(moves) + ("." if moves else "No allocation changes suggested.")


def summarize_next_day_plan(features: pd.DataFrame, policy: QQQPortfolioPolicy) -> Optional[Dict[str, object]]:
    """Generate a next-day plan using the latest features and policy."""
    if features is None or features.empty:
        return None

    clean = features.dropna(subset=["QQQ_Close"], how="any")
    if len(clean) < 2:
        return None

    prev_weights: Optional[Dict[str, float]] = None
    prior_day_weights: Optional[Dict[str, float]] = None
    prior_date: Optional[pd.Timestamp] = None
    current_weights: Optional[Dict[str, float]] = None
    current_date: Optional[pd.Timestamp] = None

    for date, row in clean.iterrows():
        current_weights = policy.get_allocation(row, prev_weights=prev_weights)
        prior_day_weights = prev_weights
        prior_date = current_date
        prev_weights = current_weights
        current_date = date

    if current_weights is None or current_date is None:
        return None

    macro, swing, regime_sentence = describe_regime(clean.loc[current_date], policy.config)
    context = getattr(policy, "last_context", {}) if hasattr(policy, "last_context") else {}
    ctx_macro = context.get("macro")
    ctx_swing = context.get("swing")
    if ctx_macro:
        macro = ctx_macro
    if ctx_swing:
        swing = ctx_swing
    if ctx_macro == "risk_off":
        regime_sentence = (
            "Macro: Risk-off override (trend caution). Short-term: "
            + {
                "upswing": "short-term upswing",
                "downswing": "short-term downswing",
            }.get(swing, "range-bound")
            + "."
        )
    changes_text = describe_changes(current_weights, prior_day_weights)

    risk_notes = policy.last_context.get("risk_flags") if hasattr(policy, "last_context") else None
    risk_notes_text = "; ".join(risk_notes) if risk_notes else ""

    return {
        "as_of": current_date,
        "regime": {
            "macro": macro,
            "swing": swing,
            "text": regime_sentence,
        },
        "weights": current_weights,
        "previous_weights": prior_day_weights,
        "changes_text": changes_text,
        "risk_notes": risk_notes_text,
        "headline": f"Next-day plan for {current_date.date()}",
        "allocation_text": f"Target allocation -> {format_allocation(current_weights)}",
    }
