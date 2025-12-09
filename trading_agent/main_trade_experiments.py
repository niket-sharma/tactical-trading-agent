"""CLI to sweep discrete trade system parameters and log results to CSV."""

from __future__ import annotations

import argparse
from typing import Optional

from trading_agent.config import DEFAULT_END_DATE, DEFAULT_START_DATE
from trading_agent.main_backtest import load_prices
from trading_agent.trade_engine import TradeSystemConfig
from trading_agent.trade_engine.experiments import run_param_sweep
from trading_agent.main_trade_system import _parse_date_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweeps for the discrete trade system.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Backtest end date (YYYY-MM-DD or None)")
    parser.add_argument("--output", default="trade_experiments.csv", help="Path to save CSV results.")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of parameter combinations (useful for quick tests).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_date = _parse_date_arg(args.start_date)
    end_date = _parse_date_arg(args.end_date)

    prices = load_prices(start_date, end_date)
    cfg = TradeSystemConfig()
    results = run_param_sweep(prices, cfg, param_grid=None, max_runs=args.max_runs)
    results.to_csv(args.output, index=False)

    print(f"Completed {len(results)} runs. Saved results to {args.output}")
    if not results.empty:
        print("Top 5 by CAGR/Sharpe:")
        print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()
