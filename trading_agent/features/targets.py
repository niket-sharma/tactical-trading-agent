"""Target generation for ML training."""

from __future__ import annotations

import pandas as pd
from typing import Tuple


def compute_targets(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute next-day return targets for QQQ."""
    qqq_close = price_df[("QQQ", "Close")]
    qqq_ret_next = qqq_close.shift(-1) / qqq_close - 1
    targets = pd.DataFrame(index=price_df.index)
    targets["QQQ_ret_next"] = qqq_ret_next
    targets["y_direction"] = (qqq_ret_next > 0).astype(int)
    return targets


def build_ml_dataset(
    features_df: pd.DataFrame, price_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Align features with ML targets and drop rows with missing data."""
    targets = compute_targets(price_df)
    combined = features_df.join(targets, how="inner")
    combined = combined.dropna()

    feature_cols = [col for col in combined.columns if col not in {"QQQ_ret_next", "y_direction"}]
    X = combined[feature_cols]
    y = combined["y_direction"]
    return X, y, combined.index
