"""Discrete trade backtester for the trend + mean reversion system."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from trading_agent.trade_engine.config import TradeSystemConfig
from trading_agent.trade_engine.indicators import compute_trade_indicators
from trading_agent.trade_engine.rules import compute_stops, determine_entry
from trading_agent.trade_engine.types import Position, TradeResult


def _position_value(position: Position, price_row: pd.Series) -> float:
    return position.shares * price_row[(position.symbol, "Close")] if position else 0.0


def run_trade_backtest(price_df: pd.DataFrame, config: Optional[TradeSystemConfig] = None) -> Tuple[pd.DataFrame, List[TradeResult]]:
    """Simulate discrete trades in TQQQ/SQQQ using rule-based entries/exits.

    Args:
        price_df: MultiIndex price DataFrame with OHLCV for QQQ/TQQQ/SQQQ.
        config: TradeSystemConfig overrides (optional).

    Returns:
        equity_curve: DataFrame indexed by date with equity, cash, and position info.
        trades: list of TradeResult objects for executed round trips.
    """
    cfg = config or TradeSystemConfig()
    indicators = compute_trade_indicators(price_df, cfg)
    dates = indicators.index
    if len(dates) < 2:
        return pd.DataFrame(), []

    cash = cfg.initial_equity
    position: Optional[Position] = None
    trades: List[TradeResult] = []
    records = []

    for idx, date in enumerate(dates):
        price_row = price_df.loc[date]
        indicator_row = indicators.loc[date]

        # Mark-to-market
        pos_value = _position_value(position, price_row) if position else 0.0
        total_equity = cash + pos_value

        # Handle open position exits (decisions at close, fills at next open)
        exit_reason = None
        if position and idx < len(dates) - 1:
            current_price = price_row[(position.symbol, "Close")]
            position.max_price = max(position.max_price, current_price)
            position.days_held += 1

            trail_stop_price = position.max_price * (1 - cfg.trailing_stop_pct) if cfg.trailing_stop_pct else None
            hit_stop = current_price <= position.stop_loss
            hit_take = current_price >= position.take_profit
            hit_trail = trail_stop_price is not None and current_price <= trail_stop_price
            hit_time = cfg.max_open_days and position.days_held >= cfg.max_open_days

            if hit_stop:
                exit_reason = "stop_loss"
            elif hit_take:
                exit_reason = "take_profit"
            elif hit_trail:
                exit_reason = "trailing_stop"
            elif hit_time:
                exit_reason = "time_exit"

            if exit_reason:
                exit_date = dates[idx + 1]
                exit_price = price_df.loc[exit_date][(position.symbol, "Open")]
                cash += position.shares * exit_price
                trades.append(
                    TradeResult(
                        symbol=position.symbol,
                        entry_date=position.entry_date,
                        exit_date=exit_date,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        pnl_pct=exit_price / position.entry_price - 1,
                        reason=exit_reason,
                    )
                )
                position = None
                pos_value = 0.0
                total_equity = cash

        # Flat or after exit: consider new entry using information up to today's close
        entry_reason = "flat"
        if position is None and idx < len(dates) - 1:
            symbol, entry_reason = determine_entry(indicator_row, cfg)
            if symbol:
                entry_date = dates[idx + 1]
                entry_price = price_df.loc[entry_date][(symbol, "Open")]
                target_fraction = min(cfg.typical_position_fraction, cfg.max_position_fraction)
                trade_value = cash * target_fraction
                if trade_value > 0:
                    shares = trade_value / entry_price
                    stop_loss, take_profit, trailing_stop = compute_stops(entry_price, cfg)
                    position = Position(
                        symbol=symbol,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        size_fraction=target_fraction,
                        shares=shares,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_stop=trailing_stop,
                        max_price=entry_price,
                        days_held=0,
                    )
                    cash -= trade_value
                    pos_value = shares * entry_price
                    total_equity = cash + pos_value

        records.append(
            {
                "date": date,
                "cash": cash,
                "position_symbol": position.symbol if position else "",
                "position_value": pos_value,
                "total_equity": total_equity,
                "entry_reason": entry_reason,
                "exit_reason": exit_reason or "",
            }
        )

    # Force-close any open trade on the final close
    if position:
        final_date = dates[-1]
        final_close = price_df.loc[final_date][(position.symbol, "Close")]
        cash += position.shares * final_close
        trades.append(
            TradeResult(
                symbol=position.symbol,
                entry_date=position.entry_date,
                exit_date=final_date,
                entry_price=position.entry_price,
                exit_price=final_close,
                pnl_pct=final_close / position.entry_price - 1,
                reason="final_close",
            )
        )
        position = None
        records[-1]["cash"] = cash
        records[-1]["position_symbol"] = ""
        records[-1]["position_value"] = 0.0
        records[-1]["total_equity"] = cash

    equity_df = pd.DataFrame.from_records(records).set_index("date")
    return equity_df, trades


def backtest_trade_system(price_df: pd.DataFrame, config: Optional[TradeSystemConfig] = None):
    """Alias for run_trade_backtest to keep public API ergonomic."""
    return run_trade_backtest(price_df, config)
