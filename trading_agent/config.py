"""Global configuration for the trading agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
DATA_CACHE_DIR = ROOT / "cache"
DATA_CACHE_FILE = DATA_CACHE_DIR / "prices.pkl"

TICKERS: List[str] = ["QQQ", "TQQQ", "SQQQ"]
DEFAULT_START_DATE = "2014-01-01"
DEFAULT_END_DATE: Optional[str] = None  # inclusive end date; None means up to latest


@dataclass
class DateRange:
    start_date: str = DEFAULT_START_DATE
    end_date: Optional[str] = DEFAULT_END_DATE

    def tuple(self) -> tuple[str, Optional[str]]:
        return self.start_date, self.end_date


def ensure_cache_dir() -> Path:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE_DIR
