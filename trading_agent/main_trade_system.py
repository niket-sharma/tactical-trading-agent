"""CLI entrypoint for the discrete trend + mean reversion trade engine."""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

from trading_agent.config import DEFAULT_END_DATE, DEFAULT_START_DATE
from trading_agent.main_backtest import load_prices
from trading_agent.strategy.metrics import compute_cagr, compute_daily_volatility, compute_max_drawdown, compute_sharpe
from trading_agent.trade_engine import TradeSystemConfig, run_trade_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run discrete TQQQ/SQQQ trade system backtest.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Backtest end date (YYYY-MM-DD or None)")
    parser.add_argument("--initial-equity", type=float, default=10000.0, help="Starting equity for the account.")
    parser.add_argument("--max-position-frac", type=float, default=0.25, help="Maximum position fraction per trade.")
    parser.add_argument("--typical-position-frac", type=float, default=0.15, help="Typical position fraction per trade.")
    return parser.parse_args()


def _parse_date_arg(label: Optional[str]) -> Optional[str]:
    """Convert CLI date arg to ISO string or None, raising on invalid input."""
    if label in (None, "", "None"):
        return None
    try:
        return pd.to_datetime(label).date().isoformat()
    except Exception as exc:  # noqa: BLE001 - provide user-friendly message
        raise SystemExit(f"Invalid date '{label}': {exc}") from exc


def run(start_date: str, end_date: Optional[str], cfg: TradeSystemConfig):
    prices = load_prices(start_date, end_date)
    equity_df, trades = run_trade_backtest(prices, cfg)
    if equity_df.empty:
        print("No backtest results (not enough data).")
        return

    equity_df["daily_return"] = equity_df["total_equity"].pct_change().fillna(0.0)
    cagr = compute_cagr(equity_df["total_equity"])
    sharpe = compute_sharpe(equity_df["daily_return"])
    max_dd = compute_max_drawdown(equity_df["total_equity"])
    vol = compute_daily_volatility(equity_df["daily_return"])

    print(f"Trade system results {start_date} -> {end_date or 'latest'}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max drawdown: {max_dd:.2%}")
    print(f"Daily volatility: {vol:.2%}")
    print(f"Trades executed: {len(trades)}")
    if trades:
        last = trades[-1]
        print(
            f"Last trade: {last.symbol} {last.entry_date.date()} -> {last.exit_date.date()}, "
            f"pnl={last.pnl_pct:.2%}, reason={last.reason}"
        )
    return equity_df, trades


def main():
    args = parse_args()
    start_date = _parse_date_arg(args.start_date)
    end_date = _parse_date_arg(args.end_date)
    cfg = TradeSystemConfig(
        initial_equity=args.initial_equity,
        max_position_fraction=args.max_position_frac,
        typical_position_fraction=args.typical_position_frac,
    )
    run(start_date, end_date, cfg)


if __name__ == "__main__":
    main()
