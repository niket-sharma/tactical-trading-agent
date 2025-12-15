"""Streamlit UI for running QQQ/TQQQ/SQQQ backtests, signals, and next-day plans."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path when launched via `streamlit run`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_agent.api import backtest
from trading_agent.config import DEFAULT_START_DATE
from trading_agent.features.indicators import compute_indicators
from trading_agent.main_backtest import load_prices
from trading_agent.strategy.config import MODE_HYBRID, MODE_ML, MODE_RULE
from trading_agent.strategy.explain import summarize_next_day_plan
from trading_agent.strategy.portfolio_config import PortfolioConfig
from trading_agent.strategy.portfolio_policy import QQQPortfolioPolicy


@st.cache_data(show_spinner=False)
def run_backtest_cached(mode: str, model_type: str, start: str, end: str):
    return backtest(mode=mode, model_type=model_type, start_date=start, end_date=end)


@st.cache_data(show_spinner=False)
def load_prices_cached(start: str, end: str):
    return load_prices(start, end)


def main():
    st.title("QQQ / TQQQ / SQQQ Research Lab")
    st.write("Rule/ML/hybrid backtests plus a plain-English next-day plan (research-only).")

    tab_backtest, tab_plan = st.tabs(["Backtest", "Next-day Plan"])

    with tab_backtest:
        col1, col2 = st.columns(2)
        mode = col1.selectbox("Mode", [MODE_RULE, MODE_ML, MODE_HYBRID], format_func=str)
        model_type = col1.selectbox(
            "ML model",
            options=["logistic_regression", "random_forest", "xgboost"],
            index=0,
            help="Used in ML/Hybrid modes; ignored for Rule mode.",
        )
        start_date = col1.text_input("Start date (YYYY-MM-DD)", value=DEFAULT_START_DATE)
        end_date_default = date.today().isoformat()
        end_date = col2.text_input("End date (YYYY-MM-DD)", value=end_date_default)

        if st.button("Run backtest", type="primary"):
            with st.spinner("Running backtest..."):
                output = run_backtest_cached(mode, model_type, start_date, end_date or None)
            results = output["results"]
            metrics = output["metrics"]

            if results.empty:
                st.warning("No results (not enough data).")
                return

            st.subheader("Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CAGR", f"{metrics.get('cagr', float('nan')):.2%}")
            m2.metric("Sharpe", f"{metrics.get('sharpe', float('nan')):.2f}")
            m3.metric("Max DD", f"{metrics.get('max_dd', float('nan')):.2%}")
            m4.metric("Daily Vol", f"{metrics.get('vol', float('nan')):.2%}")

            st.subheader("Equity Curves")
            chart_df = results[["total_value", "QQQ_buyhold_value", "TQQQ_buyhold_value", "SQQQ_buyhold_value"]]
            st.line_chart(chart_df)

            st.subheader("Latest Signals")
            st.dataframe(results[["weight_tqqq", "weight_sqqq"]].tail())

            csv = results.to_csv(index=True, date_format="%Y-%m-%d").encode("utf-8")
            st.download_button("Download signals CSV", data=csv, file_name="signals.csv", mime="text/csv")

    with tab_plan:
        st.write(
            "Get a plain-English next-day plan: macro regime, swing signal, target allocations, and adjustments vs. prior day."
        )
        plan_date_default = date.today()
        plan_date = st.date_input("Plan date (as-of)", value=plan_date_default)
        lookback_days = st.number_input(
            "Lookback window (days of history to use)", min_value=100, max_value=800, value=400, step=50
        )
        ma_short = st.number_input("Short SMA length", min_value=20, max_value=200, value=50, step=5)
        ma_long = st.number_input("Long SMA length", min_value=50, max_value=600, value=200, step=10)

        if st.button("Generate next-day plan", type="primary"):
            end_str = plan_date.isoformat()
            start_dt = plan_date - timedelta(days=int(lookback_days))
            start_str = start_dt.isoformat()

            with st.spinner("Building plan..."):
                prices = load_prices_cached(start_str, end_str)
                if prices.empty:
                    st.warning("No price data for the selected window.")
                    return

                feats = compute_indicators(prices)
                qqq_close = prices[("QQQ", "Close")]
                for window in {int(ma_short), int(ma_long)}:
                    col = f"QQQ_SMA_{window}"
                    if col not in feats.columns:
                        feats[col] = qqq_close.rolling(window=window, min_periods=window).mean()

                policy = QQQPortfolioPolicy(PortfolioConfig(ma_short=int(ma_short), ma_long=int(ma_long)))
                plan = summarize_next_day_plan(feats, policy)

            if not plan:
                st.warning("Not enough data to build a plan for that date.")
                return

            st.subheader(plan["headline"])
            st.write(plan["regime"]["text"])

            m1, m2 = st.columns(2)
            m1.metric("Macro regime", plan["regime"]["macro"].capitalize())
            m2.metric("Swing signal", plan["regime"]["swing"].capitalize())

            st.write("Target allocation (QQQ / TQQQ / SQQQ / CASH):")
            st.write(plan["allocation_text"])
            st.write(f"Adjustments vs prior day: {plan['changes_text']}")
            if plan.get("risk_notes"):
                st.warning(plan["risk_notes"])
            st.caption("Research output only. Not financial advice.")

            st.subheader("QQQ with selected SMAs")
            chart_cols = {
                "QQQ Close": feats["QQQ_Close"],
                f"SMA {int(ma_short)}": feats[f"QQQ_SMA_{int(ma_short)}"],
                f"SMA {int(ma_long)}": feats[f"QQQ_SMA_{int(ma_long)}"],
            }
            st.line_chart(pd.DataFrame(chart_cols).dropna())


if __name__ == "__main__":
    main()
