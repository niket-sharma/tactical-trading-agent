import pandas as pd

from trading_agent.features.indicators import compute_indicators
from trading_agent.features.targets import build_ml_dataset
from trading_agent.strategy.config import HybridConfig, MLConfig, StrategyConfig
from trading_agent.strategy.models import train_ml_model, time_based_split
from trading_agent.strategy.policies import HybridPolicy, MLBasedPolicy, RuleBasedPolicy
from trading_agent.strategy.backtest import backtest_strategy


def make_prices(rows=260) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    # Build an oscillating series to ensure both up and down moves
    steps = pd.Series(
        [1 if (i % 10) < 5 else -1 for i in range(rows)],
        index=dates,
        dtype=float,
    )
    base = 100 + steps.cumsum()
    data = {
        ("QQQ", "Close"): base,
        ("QQQ", "Open"): base - 0.5,
        ("QQQ", "High"): base + 1,
        ("QQQ", "Low"): base - 1,
        ("QQQ", "Adj Close"): base,
        ("QQQ", "Volume"): 1_000_000,
        ("TQQQ", "Close"): base * 3,
        ("TQQQ", "Open"): base * 3 - 0.5,
        ("TQQQ", "High"): base * 3 + 1,
        ("TQQQ", "Low"): base * 3 - 1,
        ("TQQQ", "Adj Close"): base * 3,
        ("TQQQ", "Volume"): 2_000_000,
        ("SQQQ", "Close"): base * 0.5,
        ("SQQQ", "Open"): base * 0.5 - 0.5,
        ("SQQQ", "High"): base * 0.5 + 1,
        ("SQQQ", "Low"): base * 0.5 - 1,
        ("SQQQ", "Adj Close"): base * 0.5,
        ("SQQQ", "Volume"): 1_500_000,
    }
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_walkthrough_rule_ml_hybrid():
    prices = make_prices()
    feats = compute_indicators(prices)

    # Rule allocation for a given day
    rule_policy = RuleBasedPolicy(StrategyConfig())
    rule_weights = rule_policy.get_allocation(feats.iloc[100], feats.index[100], {})
    assert sum(rule_weights) == 1

    # ML training and allocation
    X, y, dates = build_ml_dataset(feats, prices)
    ml_cfg = MLConfig(
        train_start_date="2020-01-01",
        train_end_date="2020-08-31",
        val_start_date="2020-09-01",
        val_end_date="2020-12-31",
    )
    X_train, y_train, X_val, y_val = time_based_split(X, y, dates, ml_cfg)
    model = train_ml_model(X_train, y_train)
    ml_policy = MLBasedPolicy(model, feature_cols=X.columns, ml_config=ml_cfg)
    ml_weights = ml_policy.get_allocation(feats.iloc[150], feats.index[150], {})
    assert sum(ml_weights) == 1

    # Hybrid blend
    hybrid_policy = HybridPolicy(rule_policy, ml_policy, HybridConfig(blend_alpha=0.5))
    hybrid_weights = hybrid_policy.get_allocation(feats.iloc[150], feats.index[150], {})
    assert sum(hybrid_weights) == 1

    # Backtest end-to-end
    results = backtest_strategy(prices, feats, hybrid_policy)
    assert not results.empty
    assert results["total_value"].iloc[-1] > 0
