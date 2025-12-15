"""CLI to print a plain-English next-day trading plan (research-only)."""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

from trading_agent.config import DEFAULT_END_DATE, DEFAULT_START_DATE
from trading_agent.features.indicators import compute_indicators
from trading_agent.main_backtest import load_prices
from trading_agent.strategy.explain import summarize_next_day_plan
from trading_agent.strategy.portfolio_config import PortfolioConfig
from trading_agent.strategy.portfolio_policy import QQQPortfolioPolicy


def _parse_date_arg(label: Optional[str]) -> Optional[str]:
    if label in (None, "", "None"):
        return None
    try:
        return pd.to_datetime(label).date().isoformat()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Invalid date '{label}': {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain next-day allocations for QQQ/TQQQ/SQQQ.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Backtest end date (YYYY-MM-DD or None)")
    return parser.parse_args()


def main():
    args = parse_args()
    start_date = _parse_date_arg(args.start_date)
    end_date = _parse_date_arg(args.end_date)

    prices = load_prices(start_date, end_date)
    features = compute_indicators(prices)

    policy = QQQPortfolioPolicy(PortfolioConfig())
    plan = summarize_next_day_plan(features, policy)
    if not plan:
        print("Not enough data to build a next-day plan.")
        return

    print(plan["headline"])
    print(plan["regime"]["text"])
    print(plan["allocation_text"])
    print(f"Adjustments vs. prior day: {plan['changes_text']}")
    if plan.get("risk_notes"):
        print(plan["risk_notes"])
    print("\nNote: Research output only. Not financial advice.")


if __name__ == "__main__":
    main()
