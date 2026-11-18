"""
News sentiment: the Federal Reserve Bank of San Francisco's Daily News Sentiment Index, and the
signals LevelEdge tests from it (benchmarks/news_sentiment.py, and the trust-score engine).

The index scores economics articles from 24 major US newspapers with a finance-tuned lexicon; the
daily value is a trailing average with geometrically declining weights (Buckman, Shapiro, Sudhof &
Wilson 2020). It is published weekly, so a value for day t is treated as known only from t + LAG_DAYS.
"""
from __future__ import annotations

import os
import urllib.request

import pandas as pd

URL = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
LAG_DAYS = 8
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks", "cache", "news")


def load_sentiment(cache_dir: str | None = None, refresh: bool = False) -> pd.Series:
    """Daily index values, indexed by the date the articles appeared (not yet lagged)."""
    cache_dir = cache_dir or CACHE
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "news_sentiment_data.xlsx")
    if refresh or not os.path.exists(path):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    d = pd.read_excel(path, sheet_name="Data")
    s = pd.Series(d.iloc[:, 1].to_numpy(float), index=pd.DatetimeIndex(d.iloc[:, 0]).normalize(), name="news_sentiment")
    return s.dropna().sort_index()


def available(sent: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Sentiment as known at each trading day's close: the latest value at least LAG_DAYS old."""
    known = sent.copy()
    known.index = known.index + pd.Timedelta(days=LAG_DAYS)
    return known.reindex(known.index.union(trading_days)).ffill().reindex(trading_days)


def sentiment_signals(sent: pd.Series, trading_days: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """1/0 holding signals and the raw features, all on information available at each close."""
    s = available(sent, trading_days)
    avg = s.rolling(252, min_periods=200).mean()
    chg = s - s.shift(21)
    return {
        "level": (s > avg).astype(float).where(avg.notna()),
        "momentum": (chg > 0).astype(float).where(chg.notna()),
        "level_raw": s - avg,
        "change_21d": chg,
    }


__all__ = ["load_sentiment", "available", "sentiment_signals", "LAG_DAYS"]
