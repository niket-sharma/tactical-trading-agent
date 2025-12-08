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


def build_policy(
    mode: str,
    features: pd.DataFrame,
    prices: pd.DataFrame,
    model_type: str = "logistic_regression",
    train_start: str = "2014-01-01",
    train_end: str = "2020-12-31",
    val_start: str = "2021-01-01",
    val_end: str = "2022-12-31",
):
    strategy_config = StrategyConfig(mode=mode)
    rule_policy = RuleBasedPolicy(strategy_config)

    if mode == MODE_RULE:
        return rule_policy, None

    ml_config = MLConfig(
        train_start_date=train_start,
        train_end_date=train_end,
        val_start_date=val_start,
        val_end_date=val_end,
        model_type=model_type,
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


def run_backtest(
    mode: str,
    start_date: str,
    end_date: Optional[str],
    model_type: str = "logistic_regression",
    train_start: str = "2014-01-01",
    train_end: str = "2020-12-31",
    val_start: str = "2021-01-01",
    val_end: str = "2022-12-31",
    exclude_train_from_eval: bool = False,
    transaction_cost_bps: float = 0.0,
):
    prices = load_prices(start_date, end_date)
    features = compute_indicators(prices)

    policy, ml_config = build_policy(
        mode,
        features,
        prices,
        model_type=model_type,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
    )
    results = backtest_strategy(
        prices,
        features,
        policy,
        transaction_cost_bps=transaction_cost_bps,
    )
    if results.empty:
        print("No results - not enough data.")
        return

    if exclude_train_from_eval and mode in {MODE_ML, MODE_HYBRID} and ml_config:
        eval_start = pd.to_datetime(ml_config.val_start_date)
        results = results.loc[results.index >= eval_start]

    cagr = compute_cagr(results["total_value"])
    sharpe = compute_sharpe(results["total_return"])
    max_dd = compute_max_drawdown(results["total_value"])
    vol = compute_daily_volatility(results["total_return"])

    print(f"Mode: {mode}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max drawdown: {max_dd:.2%}")
    print(f"Daily volatility: {vol:.2%}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest QQQ/TQQQ/SQQQ strategies.")
    parser.add_argument("--mode", choices=[MODE_RULE, MODE_ML, MODE_HYBRID], default=MODE_RULE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest", "xgboost"],
        default="logistic_regression",
        help="ML model type for ml/hybrid modes.",
    )
    parser.add_argument("--train-start", default="2014-01-01", help="ML train start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", default="2020-12-31", help="ML train end date (YYYY-MM-DD)")
    parser.add_argument("--val-start", default="2021-01-01", help="ML validation start date (YYYY-MM-DD)")
    parser.add_argument("--val-end", default="2022-12-31", help="ML validation end date (YYYY-MM-DD)")
    parser.add_argument(
        "--exclude-train-from-eval",
        action="store_true",
        help="If set, evaluation metrics are computed only after validation start.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="Per-trade transaction cost in basis points applied to traded value.",
    )
    parser.add_argument(
        "--save-signals",
        type=str,
        default=None,
        help="Optional path to save daily signals (allocations) as CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_backtest(
        args.mode,
        args.start_date,
        args.end_date,
        model_type=args.model_type,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        exclude_train_from_eval=args.exclude_train_from_eval,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    if args.save_signals and results is not None and not results.empty:
        cols = [
            "weight_tqqq",
            "weight_sqqq",
            "total_value",
            "total_return",
            "lev_capital",
            "QQQ_buyhold_value",
            "TQQQ_buyhold_value",
            "SQQQ_buyhold_value",
        ]
        export_df = results[[c for c in cols if c in results.columns]].copy()
        export_df.to_csv(args.save_signals, index=True, date_format="%Y-%m-%d")
        print(f"Saved daily signals to {args.save_signals}")


if __name__ == "__main__":
    main()
