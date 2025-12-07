import pandas as pd

from trading_agent.features.indicators import compute_indicators
from trading_agent.features.targets import build_ml_dataset
from trading_agent.strategy.backtest import backtest_strategy
from trading_agent.strategy.config import StrategyConfig
from trading_agent.strategy.policies import RuleBasedPolicy
from trading_agent.agent.daily_loop import recommendations_to_signals, AgentRecommendation


def make_price_df(rows: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    base = pd.Series(range(rows), index=dates, dtype=float) + 100
    data = {
        ("QQQ", "Close"): base,
        ("QQQ", "Open"): base - 0.5,
        ("QQQ", "High"): base + 1.0,
        ("QQQ", "Low"): base - 1.0,
        ("QQQ", "Adj Close"): base,
        ("QQQ", "Volume"): pd.Series(1_000_000, index=dates),
        ("TQQQ", "Close"): base * 3,
        ("TQQQ", "Open"): base * 3 - 0.5,
        ("TQQQ", "High"): base * 3 + 1.0,
        ("TQQQ", "Low"): base * 3 - 1.0,
        ("TQQQ", "Adj Close"): base * 3,
        ("TQQQ", "Volume"): pd.Series(2_000_000, index=dates),
        ("SQQQ", "Close"): base * 0.5,
        ("SQQQ", "Open"): base * 0.5 - 0.5,
        ("SQQQ", "High"): base * 0.5 + 1.0,
        ("SQQQ", "Low"): base * 0.5 - 1.0,
        ("SQQQ", "Adj Close"): base * 0.5,
        ("SQQQ", "Volume"): pd.Series(1_500_000, index=dates),
    }
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_indicators_and_targets_align():
    prices = make_price_df()
    feats = compute_indicators(prices)
    X, y, dates = build_ml_dataset(feats, prices)
    assert not X.empty
    assert len(X) == len(y) == len(dates)
    assert "QQQ_SMA_10" in X.columns
    assert "QQQ_vol_10" in X.columns


def test_backtest_runs_with_rule_policy():
    prices = make_price_df()
    feats = compute_indicators(prices)
    policy = RuleBasedPolicy(StrategyConfig())
    results = backtest_strategy(prices, feats, policy)
    assert not results.empty
    assert {"total_value", "QQQ_buyhold_value", "lev_capital"} <= set(results.columns)
    assert results["total_value"].iloc[-1] > 0


def test_recommendations_to_signals():
    recs = [
        AgentRecommendation(date=pd.Timestamp("2020-01-01"), weight_tqqq=0.6, weight_sqqq=0.4),
        AgentRecommendation(date=pd.Timestamp("2020-01-02"), weight_tqqq=0.4, weight_sqqq=0.6),
        AgentRecommendation(date=pd.Timestamp("2020-01-03"), weight_tqqq=0.7, weight_sqqq=0.3),
    ]
    signals = recommendations_to_signals(recs, portfolio_value=10000.0, flips_only=False)
    assert not signals.empty
    assert "delta_weight_T" in signals.columns
    assert signals["regime_flip"].sum() >= 1
