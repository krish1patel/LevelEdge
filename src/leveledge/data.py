"""
OHLCV loading and cleaning.

Every frame that reaches the model goes through :func:`clean_bars`, whether it
came from yfinance, Polygon, or a caller-supplied DataFrame.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from leveledge.constants import US_EASTERN
from leveledge.sessions import (
    DAILY_MIN,
    completed_bars,
    interval_minutes,
    is_crypto_ticker,
    regular_session_only,
    to_eastern,
)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# How far back yfinance serves each interval (days, measured from *now*).
YF_LOOKBACK_DAYS = {"1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "90m": 59, "1h": 729}


def clean_bars(
    df: pd.DataFrame,
    interval: str,
    ticker: str,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Normalise a raw OHLCV frame.

    * US/Eastern DatetimeIndex, sorted, de-duplicated
    * stocks: regular session only (pre/post bars are thin and the outcome is
      always a regular-session close)
    * drop bars that had not closed by ``as_of`` (default: now)
    * drop rows with missing or non-positive prices
    """
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {ticker} ({interval}).")
    imin = interval_minutes(interval)
    crypto = is_crypto_ticker(ticker)
    out = df[[c for c in OHLCV if c in df.columns]].copy()
    if "Volume" not in out:
        out["Volume"] = 0.0
    out.index = to_eastern(pd.DatetimeIndex(out.index))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    out["Volume"] = out["Volume"].fillna(0.0).astype(float)
    if not crypto:
        out = regular_session_only(out, imin)
    as_of = as_of or datetime.now(tz=US_EASTERN)
    out = completed_bars(out, imin, crypto, as_of)
    if out.empty:
        raise ValueError(f"No completed {interval} bars for {ticker} before {as_of:%Y-%m-%d %H:%M} ET.")
    return out


def fetch_history(ticker: str, interval: str, as_of: datetime | None = None) -> pd.DataFrame:
    """Download as much history as yfinance allows, ending at ``as_of``."""
    ticker = ticker.upper().strip()
    t = yf.Ticker(ticker)
    now = datetime.now(tz=US_EASTERN)
    if as_of is None or as_of >= now - timedelta(minutes=1):
        period = "max" if interval == "1d" else f"{YF_LOOKBACK_DAYS[interval]}d"
        raw = t.history(period=period, interval=interval, prepost=False, auto_adjust=True)
        as_of = None
    else:
        if interval == "1d":
            start = as_of - timedelta(days=365 * 25)
        else:
            start = now - timedelta(days=YF_LOOKBACK_DAYS[interval])
            if start >= as_of:
                raise ValueError(
                    f"yfinance keeps only ~{YF_LOOKBACK_DAYS[interval]} days of {interval} bars; "
                    f"{as_of:%Y-%m-%d} is too far back. Use a longer interval or Polygon data."
                )
        pad = timedelta(days=1) if interval == "1d" else timedelta(minutes=interval_minutes(interval))
        raw = t.history(start=start, end=as_of + pad, interval=interval, prepost=False, auto_adjust=True)
    return clean_bars(raw, interval, ticker, as_of)


def last_price(ticker: str) -> float | None:
    """Latest traded price (for display only; the model anchors on completed bars)."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m", prepost=False)
        return float(df["Close"].iloc[-1]) if not df.empty else None
    except Exception:
        return None


__all__ = ["clean_bars", "fetch_history", "last_price", "OHLCV", "DAILY_MIN"]
