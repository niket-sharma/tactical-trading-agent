"""Offline daily-loop simulation for the trading agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from trading_agent.config import (
    DATA_CACHE_FILE,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    TICKERS,
)
from trading_agent.data.loader import adjust_and_align, get_price_history, load_from_cache, save_to_cache
from trading_agent.features.indicators import compute_indicators
from trading_agent.features.targets import build_ml_dataset
from trading_agent.strategy.config import HybridConfig, MLConfig, StrategyConfig, MODE_HYBRID, MODE_ML, MODE_RULE
from trading_agent.strategy.models import evaluate_model, time_based_split, train_ml_model
from trading_agent.strategy.policies import HybridPolicy, MLBasedPolicy, RuleBasedPolicy


@dataclass
class AgentRecommendation:
    date: pd.Timestamp
    weight_tqqq: float
    weight_sqqq: float

    def __str__(self) -> str:
        wt = f"{self.weight_tqqq:.0%}"
        ws = f"{self.weight_sqqq:.0%}"
        return f"{self.date.date()}: allocate {wt} TQQQ / {ws} SQQQ"


def load_prices(start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:
    """Load cached prices or download and cache them."""
    cached = load_from_cache(DATA_CACHE_FILE)
    if cached is not None:
        return cached

    prices = get_price_history(TICKERS, start_date=start_date, end_date=end_date)
    prices = adjust_and_align(prices)
    save_to_cache(prices, DATA_CACHE_FILE)
    return prices


def simulate_daily_agent(mode: str = MODE_RULE) -> List[AgentRecommendation]:
    prices = load_prices()
    prices = adjust_and_align(prices)
    features = compute_indicators(prices)

    strategy_config = StrategyConfig(mode=mode)
    rule_policy = RuleBasedPolicy(strategy_config)

    ml_policy = None
    hybrid_config = HybridConfig()

    if mode in {MODE_ML, MODE_HYBRID}:
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
        print(f"Validation accuracy={acc:.3f}, AUC={auc:.3f}")
        ml_policy = MLBasedPolicy(model, feature_cols=X.columns, ml_config=ml_config)

    if mode == MODE_RULE:
        policy = rule_policy
    elif mode == MODE_ML:
        policy = ml_policy
    else:
        policy = HybridPolicy(rule_policy, ml_policy, hybrid_config)

    recommendations: List[AgentRecommendation] = []
    dates = features.index
    prev_weights = (0.5, 0.5)
    for idx in range(len(dates) - 1):
        date = dates[idx]
        row = features.loc[date]
        state = {"prev_weights": prev_weights}
        w_t, w_s = policy.get_allocation(row, date, state)
        prev_weights = (w_t, w_s)
        recommendations.append(AgentRecommendation(date=date, weight_tqqq=w_t, weight_sqqq=w_s))

    for rec in recommendations[:5]:
        print(rec)
    return recommendations


def recommendations_to_signals(
    recommendations: List[AgentRecommendation],
    portfolio_value: float = 10000.0,
    flips_only: bool = False,
) -> pd.DataFrame:
    """Convert allocations into daily trade deltas and action tags."""
    records = []
    prev_wt, prev_ws = 0.5, 0.5
    prev_dominant = "TQQQ" if prev_wt >= prev_ws else "SQQQ"

    for rec in recommendations:
        delta_wt = rec.weight_tqqq - prev_wt
        delta_ws = rec.weight_sqqq - prev_ws
        trade_t_val = delta_wt * portfolio_value
        trade_s_val = delta_ws * portfolio_value

        dominant = "TQQQ" if rec.weight_tqqq >= rec.weight_sqqq else "SQQQ"
        flipped = dominant != prev_dominant

        if not flips_only or flipped:
            records.append(
                {
                    "date": rec.date,
                    "weight_T": rec.weight_tqqq,
                    "weight_S": rec.weight_sqqq,
                    "delta_weight_T": delta_wt,
                    "delta_weight_S": delta_ws,
                    "trade_value_T": trade_t_val,
                    "trade_value_S": trade_s_val,
                    "dominant": dominant,
                    "regime_flip": flipped,
                }
            )

        prev_wt, prev_ws = rec.weight_tqqq, rec.weight_sqqq
        prev_dominant = dominant

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.set_index("date", inplace=True)
    return df


def export_daily_signals(
    mode: str = MODE_RULE,
    output_csv: Optional[str] = None,
    portfolio_value: float = 10000.0,
    flips_only: bool = False,
) -> pd.DataFrame:
    """Run the agent and export daily signals to a CSV."""
    recs = simulate_daily_agent(mode=mode)
    signals = recommendations_to_signals(
        recs, portfolio_value=portfolio_value, flips_only=flips_only
    )
    if output_csv:
        signals.to_csv(output_csv, index=True, date_format="%Y-%m-%d")
        print(f"Wrote {len(signals)} signal rows to {output_csv}")
    return signals


def parse_args():
    parser = argparse.ArgumentParser(description="Generate daily allocation signals.")
    parser.add_argument("--mode", choices=[MODE_RULE, MODE_ML, MODE_HYBRID], default=MODE_RULE)
    parser.add_argument("--output", help="Path to CSV for signals", default="signals.csv")
    parser.add_argument("--portfolio-value", type=float, default=10000.0)
    parser.add_argument("--flips-only", action="store_true", help="Only emit rows when regime flips")
    return parser.parse_args()


def main():
    args = parse_args()
    export_daily_signals(
        mode=args.mode,
        output_csv=args.output,
        portfolio_value=args.portfolio_value,
        flips_only=args.flips_only,
    )


if __name__ == "__main__":
    main()
