import pytest

from trading_agent.config import DATA_CACHE_FILE, TICKERS
from trading_agent.data.loader import adjust_and_align, get_price_history, load_from_cache, save_to_cache
from trading_agent.features.indicators import compute_indicators
from trading_agent.strategy.backtest import backtest_strategy
from trading_agent.strategy.config import StrategyConfig
from trading_agent.strategy.policies import RuleBasedPolicy


def _load_prices_or_skip():
    cached = load_from_cache(DATA_CACHE_FILE)
    if cached is not None:
        return adjust_and_align(cached)
    try:
        prices = get_price_history(TICKERS, start_date="2023-01-01")
        prices = adjust_and_align(prices)
        save_to_cache(prices, DATA_CACHE_FILE)
        return prices
    except Exception as exc:  # pragma: no cover - network/caching guard
        pytest.skip(f"Skipping live integration test (data unavailable): {exc}")


def test_live_rule_backtest_small_window():
    prices = _load_prices_or_skip()
    features = compute_indicators(prices)
    policy = RuleBasedPolicy(StrategyConfig())
    results = backtest_strategy(prices, features, policy)
    assert not results.empty
    assert results["total_value"].iloc[-1] > 0
