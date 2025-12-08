"""Streamlit UI for running QQQ/TQQQ/SQQQ backtests and viewing signals."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from trading_agent.api import backtest
from trading_agent.config import DEFAULT_START_DATE
from trading_agent.strategy.config import MODE_RULE, MODE_ML, MODE_HYBRID


@st.cache_data(show_spinner=False)
def run_backtest_cached(mode: str, model_type: str, start: str, end: str):
    return backtest(mode=mode, model_type=model_type, start_date=start, end_date=end)


def main():
    st.title("QQQ / TQQQ / SQQQ Backtester")
    st.write("Rule-based, ML, and hybrid allocation between TQQQ (long) and SQQQ (inverse).")

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


if __name__ == "__main__":
    main()
