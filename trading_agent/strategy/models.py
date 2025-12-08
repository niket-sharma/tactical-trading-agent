"""ML utilities for training and inference."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from trading_agent.features.targets import build_ml_dataset as build_targets_dataset
from trading_agent.strategy.config import MLConfig


def build_ml_dataset(features_df: pd.DataFrame, price_df: pd.DataFrame):
    """Proxy to feature/target builder for convenience."""
    return build_targets_dataset(features_df, price_df)


def train_ml_model(
    X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "logistic_regression"
):
    """Train a simple classifier."""
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2, n_jobs=-1
        )
    elif model_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("xgboost is not installed. pip install xgboost to use this model.") from exc
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
    else:  # pragma: no cover - unexpected model
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def time_based_split(
    X: pd.DataFrame, y: pd.Series, dates: pd.Index, ml_config: MLConfig
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split dataset by date ranges."""
    train_mask = (dates >= ml_config.train_start_date) & (dates <= ml_config.train_end_date)
    val_mask = (dates >= ml_config.val_start_date) & (dates <= ml_config.val_end_date)

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    return X_train, y_train, X_val, y_val


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[float, float]:
    """Compute accuracy and ROC-AUC for validation set."""
    if len(y_val) == 0:
        return float("nan"), float("nan")
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_val, preds)
    try:
        auc = roc_auc_score(y_val, probs)
    except ValueError:
        auc = float("nan")
    return acc, auc


def predict_direction_proba(model, x_row: pd.Series) -> float:
    """Return probability that QQQ goes up next day."""
    if isinstance(x_row, pd.Series):
        x_row = x_row.to_frame().T
    proba = model.predict_proba(x_row)[0, 1]
    return float(proba)
