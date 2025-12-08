"""Backtesting engine for allocation policies."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from trading_agent.strategy.policies import BasePolicy


def backtest_strategy(
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    policy: BasePolicy,
    initial_long_QQQ: float = 10000.0,
    initial_lev_capital: float = 10000.0,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Simulate daily rebalancing between TQQQ and SQQQ."""
    dates = feature_df.index
    if len(dates) < 2:
        return pd.DataFrame()

    qqq_close = price_df[("QQQ", "Close")]
    tqqq_close = price_df[("TQQQ", "Close")]
    sqqq_close = price_df[("SQQQ", "Close")]

    qqq_shares = initial_long_QQQ / qqq_close.iloc[0]
    lev_capital = initial_lev_capital
    tqqq_bh_shares = initial_lev_capital / tqqq_close.iloc[0]
    sqqq_bh_shares = initial_lev_capital / sqqq_close.iloc[0]

    prev_total_value = qqq_shares * qqq_close.iloc[0] + lev_capital
    prev_weights = (0.5, 0.5)

    records = []
    for idx in range(len(dates) - 1):
        date = dates[idx]
        next_date = dates[idx + 1]

        feature_row = feature_df.loc[date].copy()
        price_row = price_df.loc[date]
        feature_row["QQQ_Close"] = price_row[("QQQ", "Close")]
        feature_row["QQQ_SMA_50"] = feature_row.get("QQQ_SMA_50")
        feature_row["QQQ_SMA_100"] = feature_row.get("QQQ_SMA_100")
        feature_row["QQQ_vol_20"] = feature_row.get("QQQ_vol_20")

        state: Dict[str, Tuple[float, float]] = {"prev_weights": prev_weights, "lev_capital": lev_capital}
        w_t, w_s = policy.get_allocation(feature_row, date, state)
        prev_weights = (w_t, w_s)

        price_t = price_df.loc[date]
        price_next = price_df.loc[next_date]

        target_t_val = lev_capital * w_t
        target_s_val = lev_capital * w_s

        # Transaction costs applied to traded value (both legs)
        traded_val = abs(target_t_val) + abs(target_s_val)
        cost = traded_val * (transaction_cost_bps / 10000.0)

        tqqq_shares = target_t_val / price_t[("TQQQ", "Close")]
        sqqq_shares = target_s_val / price_t[("SQQQ", "Close")]

        lev_capital_start = lev_capital
        lev_capital = (
            tqqq_shares * price_next[("TQQQ", "Close")] + sqqq_shares * price_next[("SQQQ", "Close")] - cost
        )
        lev_return = lev_capital / lev_capital_start - 1 if lev_capital_start else 0.0

        qqq_value = qqq_shares * price_next[("QQQ", "Close")]
        total_value = qqq_value + lev_capital
        total_return = total_value / prev_total_value - 1 if prev_total_value else 0.0
        prev_total_value = total_value

        tqqq_only_value = tqqq_bh_shares * price_next[("TQQQ", "Close")]
        sqqq_only_value = sqqq_bh_shares * price_next[("SQQQ", "Close")]

        records.append(
            {
                "date": next_date,
                "QQQ_buyhold_value": qqq_value,
                "lev_capital": lev_capital,
                "total_value": total_value,
                "weight_T": w_t,
                "weight_S": w_s,
                "weight_tqqq": w_t,
                "weight_sqqq": w_s,
                "lev_return": lev_return,
                "total_return": total_return,
                "TQQQ_buyhold_value": tqqq_only_value,
                "SQQQ_buyhold_value": sqqq_only_value,
            }
        )

    result_df = pd.DataFrame.from_records(records).set_index("date")
    return result_df
