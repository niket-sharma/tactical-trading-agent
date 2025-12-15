import numpy as np
import pandas as pd

from tests.test_core import make_price_df
from trading_agent.features.indicators import compute_indicators
from trading_agent.strategy.explain import summarize_next_day_plan
from trading_agent.strategy.portfolio_config import PortfolioConfig
from trading_agent.strategy.portfolio_policy import QQQPortfolioPolicy, classify_macro_trend, classify_swing_signal


def test_next_day_plan_returns_weights_and_text():
    prices = make_price_df(120)
    feats = compute_indicators(prices)
    policy = QQQPortfolioPolicy(PortfolioConfig())

    plan = summarize_next_day_plan(feats, policy)

    assert plan is not None
    weights = plan["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert plan["allocation_text"]
    assert plan["changes_text"]

    macro = classify_macro_trend(feats.iloc[-1], policy.config)
    swing = classify_swing_signal(feats.iloc[-1], policy.config)
    assert macro in {"bull", "bear", "neutral"}
    assert swing in {"upswing", "downswing", "neutral"}


def test_core_qqq_weight_floor():
    prices = make_price_df(150)
    feats = compute_indicators(prices)
    cfg = PortfolioConfig(core_qqq_weight=0.5)
    policy = QQQPortfolioPolicy(cfg)

    weights = policy.get_allocation(feats.iloc[-1])
    assert weights["QQQ"] >= cfg.core_qqq_weight - 1e-6


def test_negative_sma_slope_triggers_risk_off():
    # Create a downward trending price series to generate negative SMA50 slope
    dates = pd.date_range("2021-01-01", periods=120, freq="D")
    base = pd.Series(np.linspace(200, 150, num=120), index=dates)
    data = {
        ("QQQ", "Close"): base,
        ("QQQ", "Open"): base,
        ("QQQ", "High"): base,
        ("QQQ", "Low"): base,
        ("QQQ", "Adj Close"): base,
        ("QQQ", "Volume"): pd.Series(1_000_000, index=dates),
        ("TQQQ", "Close"): base * 3,
        ("TQQQ", "Open"): base * 3,
        ("TQQQ", "High"): base * 3,
        ("TQQQ", "Low"): base * 3,
        ("TQQQ", "Adj Close"): base * 3,
        ("TQQQ", "Volume"): pd.Series(2_000_000, index=dates),
        ("SQQQ", "Close"): base * 0.5,
        ("SQQQ", "Open"): base * 0.5,
        ("SQQQ", "High"): base * 0.5,
        ("SQQQ", "Low"): base * 0.5,
        ("SQQQ", "Adj Close"): base * 0.5,
        ("SQQQ", "Volume"): pd.Series(1_500_000, index=dates),
    }
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    feats = compute_indicators(df)
    cfg = PortfolioConfig()
    policy = QQQPortfolioPolicy(cfg)
    weights = policy.get_allocation(feats.iloc[-1])

    assert weights["CASH"] >= cfg.cash_risk_off_floor - 1e-6
    assert weights["TQQQ"] <= cfg.tqqq_hard_cap
