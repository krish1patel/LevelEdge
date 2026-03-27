"""
Polygon.io data fetcher for LevelEdge.

Returns DataFrames with the same schema as yfinance.history() so they can be
passed directly to Predictor(data=...).

Free-tier limits
----------------
- 2 years of minute-level aggregate bars for US stocks
- 5 API calls / minute — pagination is handled with automatic back-off so
  callers don't need to worry about rate limits.
"""
from __future__ import annotations

import os
import time as _time
from datetime import datetime

import pandas as pd

from leveledge.constants import US_EASTERN

# LevelEdge interval string → Polygon (multiplier, timespan)
_INTERVAL_MAP: dict[str, tuple[int, str]] = {
    "1m":  (1,  "minute"),
    "2m":  (2,  "minute"),
    "5m":  (5,  "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "1h":  (1,  "hour"),
    "90m": (90, "minute"),
    "1d":  (1,  "day"),
}

_MARKET_OPEN  = pd.Timestamp("09:30:00").time()
_MARKET_CLOSE = pd.Timestamp("16:00:00").time()

# Seconds to wait between paginated requests (free tier: 5 req/min → 12 s apart)
# Using 15s for a comfortable margin
_PAGE_SLEEP = 15


def _to_polygon_ticker(ticker: str) -> str:
    """Convert yfinance-style ticker to Polygon format.
    ETH-USD → X:ETHUSD,  QQQ → QQQ
    """
    if '-' in ticker:
        base, quote = ticker.split('-', 1)
        return f"X:{base}{quote}"
    return ticker


def fetch_polygon(
    ticker: str,
    interval: str,
    start: datetime,
    end: datetime,
    api_key: str | None = None,
    market_hours_only: bool | None = None,
) -> pd.DataFrame:
    """
    Fetch adjusted aggregate bars from Polygon.io.

    Returns a DataFrame with a timezone-aware DatetimeIndex (US/Eastern) and
    columns Open, High, Low, Close, Volume — same schema as yfinance.history().

    Parameters
    ----------
    ticker           : yfinance-style ticker ("QQQ", "ETH-USD"). Crypto tickers
                       with a dash are auto-converted to Polygon X: format.
    interval         : LevelEdge interval string ("1m", "15m", "1h", …)
    start / end      : Datetime range (inclusive). May be tz-aware or naive ET.
    api_key          : Polygon API key; falls back to POLYGON_API_KEY env var.
    market_hours_only: Filter intraday bars to 09:30–16:00 ET. Defaults to
                       True for stocks, False for crypto (tickers containing '-').
    """
    is_crypto = '-' in ticker
    if market_hours_only is None:
        market_hours_only = not is_crypto
    try:
        from polygon import RESTClient
    except ImportError:
        raise ImportError(
            "polygon-api-client is not installed. Run: pip install polygon-api-client"
        )

    if interval not in _INTERVAL_MAP:
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported: {list(_INTERVAL_MAP)}"
        )

    if api_key is None:
        api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable is not set.")

    multiplier, timespan = _INTERVAL_MAP[interval]
    client = RESTClient(api_key)
    polygon_ticker = _to_polygon_ticker(ticker)

    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    # Manual pagination with rate-limit sleep between pages.
    bars = []
    page = 0
    for agg in client.list_aggs(
        ticker=polygon_ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=start_str,
        to=end_str,
        limit=50000,
        adjusted=True,
    ):
        bars.append(agg)
        page += 1
        # Sleep every 50000 bars (i.e., between pages) to respect 5 req/min
        if page % 50000 == 0:
            _time.sleep(_PAGE_SLEEP)

    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    records = [
        {
            "Datetime": pd.Timestamp(b.timestamp, unit="ms", tz="UTC").tz_convert(US_EASTERN),
            "Open":     b.open,
            "High":     b.high,
            "Low":      b.low,
            "Close":    b.close,
            "Volume":   b.volume,
        }
        for b in bars
    ]

    df = pd.DataFrame(records).set_index("Datetime")

    # Clip to [start, end]
    start_ts = (
        pd.Timestamp(start).tz_localize(US_EASTERN)
        if start.tzinfo is None
        else pd.Timestamp(start).tz_convert(US_EASTERN)
    )
    end_ts = (
        pd.Timestamp(end).tz_localize(US_EASTERN)
        if end.tzinfo is None
        else pd.Timestamp(end).tz_convert(US_EASTERN)
    )
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    # Filter to regular market hours for intraday intervals
    if market_hours_only and timespan in ("minute", "hour"):
        df = df[
            (df.index.time >= _MARKET_OPEN) &
            (df.index.time <= _MARKET_CLOSE)
        ]

    return df.sort_index()
