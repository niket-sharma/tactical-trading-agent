"""Entrypoint to run backtests for rule, ML, or hybrid modes."""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from trading_agent.config import DATA_CACHE_FILE, DEFAULT_END_DATE, DEFAULT_START_DATE, TICKERS
from trading_agent.data.loader import adjust_and_align, get_price_history, load_from_cache, save_to_cache
from trading_agent.features.indicators import compute_indicators
from trading_agent.features.targets import build_ml_dataset
from trading_agent.strategy.backtest import backtest_strategy
from trading_agent.strategy.config import HybridConfig, MLConfig, StrategyConfig, MODE_HYBRID, MODE_ML, MODE_RULE
from trading_agent.strategy.metrics import (
    compute_cagr,
    compute_daily_volatility,
    compute_max_drawdown,
    compute_sharpe,
)
from trading_agent.strategy.models import evaluate_model, time_based_split, train_ml_model
from trading_agent.strategy.policies import HybridPolicy, MLBasedPolicy, RuleBasedPolicy


def load_prices(start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    cached = load_from_cache(DATA_CACHE_FILE)
    if cached is None:
        prices = get_price_history(TICKERS, start_date=start_date, end_date=end_date)
        prices = adjust_and_align(prices)
        save_to_cache(prices, DATA_CACHE_FILE)
        cached = prices
    cached = adjust_and_align(cached)
    if start_date or end_date:
        cached = cached.loc[start_date:end_date]
    return cached


def build_policy(mode: str, features: pd.DataFrame, prices: pd.DataFrame):
    strategy_config = StrategyConfig(mode=mode)
    rule_policy = RuleBasedPolicy(strategy_config)

    if mode == MODE_RULE:
        return rule_policy, None

    ml_config = MLConfig(
        train_start_date="2014-01-01",
        train_end_date="2020-12-31",
        val_start_date="2021-01-01",
        val_end_date="2022-12-31",
        model_type="logistic_regression",
        threshold=0.55,
    )
    X, y, dates = build_ml_dataset(features, prices)
    X_train, y_train, X_val, y_val = time_based_split(X, y, dates, ml_config)
    if len(X_train) == 0:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_ml_model(X_train, y_train, model_type=ml_config.model_type)
    acc, auc = evaluate_model(model, X_val, y_val)
    print(f"[ML] Validation accuracy={acc:.3f}, AUC={auc:.3f}")
    ml_policy = MLBasedPolicy(model, feature_cols=X.columns, ml_config=ml_config)

    if mode == MODE_ML:
        return ml_policy, ml_config

    hybrid_config = HybridConfig()
    hybrid_policy = HybridPolicy(rule_policy, ml_policy, hybrid_config)
    return hybrid_policy, ml_config


def run_backtest(mode: str, start_date: str, end_date: Optional[str]):
    prices = load_prices(start_date, end_date)
    features = compute_indicators(prices)

    policy, ml_config = build_policy(mode, features, prices)
    results = backtest_strategy(prices, features, policy)
    if results.empty:
        print("No results - not enough data.")
        return

    cagr = compute_cagr(results["total_value"])
    sharpe = compute_sharpe(results["total_return"])
    max_dd = compute_max_drawdown(results["total_value"])
    vol = compute_daily_volatility(results["total_return"])

    print(f"Mode: {mode}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max drawdown: {max_dd:.2%}")
    print(f"Daily volatility: {vol:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest QQQ/TQQQ/SQQQ strategies.")
    parser.add_argument("--mode", choices=[MODE_RULE, MODE_ML, MODE_HYBRID], default=MODE_RULE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    return parser.parse_args()


def main():
    args = parse_args()
    run_backtest(args.mode, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
