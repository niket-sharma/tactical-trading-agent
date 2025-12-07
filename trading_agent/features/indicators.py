"""Feature engineering for price-based indicators."""

from __future__ import annotations

import pandas as pd


def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute moving averages, returns, volatility, and momentum features."""
    features = pd.DataFrame(index=price_df.index)

    qqq_close = price_df[("QQQ", "Close")]
    tqqq_close = price_df[("TQQQ", "Close")]
    sqqq_close = price_df[("SQQQ", "Close")]

    features["QQQ_Close"] = qqq_close
    features["TQQQ_Close"] = tqqq_close
    features["SQQQ_Close"] = sqqq_close

    ma_windows = [10, 20, 50, 100, 200]
    for window in ma_windows:
        sma = qqq_close.rolling(window=window, min_periods=window).mean()
        features[f"QQQ_SMA_{window}"] = sma
        features[f"QQQ_over_SMA_{window}"] = (qqq_close - sma) / sma

    features["QQQ_ret"] = qqq_close.pct_change()
    features["TQQQ_ret"] = tqqq_close.pct_change()
    features["SQQQ_ret"] = sqqq_close.pct_change()

    features["QQQ_vol_10"] = features["QQQ_ret"].rolling(window=10, min_periods=10).std()
    features["QQQ_vol_20"] = features["QQQ_ret"].rolling(window=20, min_periods=20).std()

    features["QQQ_mom_10"] = qqq_close / qqq_close.shift(10) - 1

    return features
