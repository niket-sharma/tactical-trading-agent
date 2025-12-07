"""Download and cache QQQ/TQQQ/SQQQ price history."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from trading_agent.config import ensure_cache_dir


def get_price_history(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download daily OHLCV for the given tickers using yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("yfinance is required to download data") from exc

    data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    if not isinstance(data.columns, pd.MultiIndex):
        data = data.stack(0).unstack().swaplevel(axis=1)
        data.sort_index(axis=1, inplace=True)

    data.index = pd.to_datetime(data.index)
    return data


def adjust_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Align all tickers on the same date index and drop missing rows."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    close_cols = [col for col in df.columns if col[1] == "Close"]
    df = df.dropna(subset=close_cols)
    common_dates = df.index
    aligned = df.loc[common_dates]
    return aligned


def save_to_cache(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Cache the price DataFrame to disk."""
    cache_dir = ensure_cache_dir()
    target = path or (cache_dir / "prices.pkl")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(target)
    return target


def load_from_cache(path: Path) -> Optional[pd.DataFrame]:
    """Load cached prices if the file exists."""
    if path.exists():
        return pd.read_pickle(path)
    return None
