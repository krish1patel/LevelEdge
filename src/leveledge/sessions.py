"""
Market-session arithmetic: bar end times, completed-bar filtering, and the
exact number of bars between "now" and a target datetime.

Conventions
-----------
* Bars are indexed by their START time (yfinance / Polygon convention), in
  US/Eastern.
* A bar ENDS at ``start + interval`` or at the session close, whichever is
  first (yfinance's last 1h bar of a stock session is 15:30-16:00, the last
  90m bar is 15:30-16:00).
* Stocks use the NYSE calendar with real open/close times, so early-close
  days (e.g. the day after Thanksgiving) are handled. Only regular-session
  bars are used.
* Crypto (ticker contains "-", e.g. BTC-USD) trades 24/7 on a fixed grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from leveledge.constants import US_EASTERN

DAILY_MIN = 1440


def interval_minutes(interval: str) -> int:
    """'15m' -> 15, '1h' -> 60, '1d' -> 1440."""
    unit, n = interval[-1], int(interval[:-1])
    return {"m": n, "h": 60 * n, "d": DAILY_MIN * n}[unit]


def is_crypto_ticker(ticker: str) -> bool:
    return "-" in ticker


@lru_cache(maxsize=16)
def _nyse_schedule(start: str, end: str) -> pd.DataFrame:
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    sched = sched[["market_open", "market_close"]].apply(lambda c: c.dt.tz_convert(US_EASTERN))
    sched.index = pd.DatetimeIndex(sched.index.date)
    return sched


def nyse_schedule(start, end) -> pd.DataFrame:
    """NYSE sessions between two dates (inclusive), open/close in US/Eastern."""
    s = pd.Timestamp(start).date().isoformat()
    e = pd.Timestamp(end).date().isoformat()
    return _nyse_schedule(s, e)


def to_eastern(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(US_EASTERN)
    return index.tz_convert(US_EASTERN)


def bar_ends(index: pd.DatetimeIndex, interval_min: int, is_crypto: bool) -> pd.DatetimeIndex:
    """End time of every bar in ``index`` (bar-start timestamps, US/Eastern)."""
    index = to_eastern(index)
    step = pd.Timedelta(minutes=interval_min)
    if is_crypto or len(index) == 0:
        return index + step

    closes = _session_field(index, "market_close")
    if interval_min >= DAILY_MIN:
        # Daily stock bar: ends at that session's close.
        return pd.DatetimeIndex(closes.fillna(pd.Series(index + pd.Timedelta(hours=16), index=closes.index)))
    raw_end = pd.Series(index + step, index=closes.index)
    # Bars on dates missing from the calendar (shouldn't happen) keep raw end.
    ends = raw_end.where(closes.isna() | (raw_end <= closes), closes)
    return pd.DatetimeIndex(ends)


def _session_field(index: pd.DatetimeIndex, field: str) -> pd.Series:
    """Per-bar session open/close (tz-aware Series aligned positionally to index)."""
    sched = nyse_schedule(index[0] - pd.Timedelta(days=1), index[-1] + pd.Timedelta(days=1))
    dates = pd.DatetimeIndex(index.tz_localize(None).normalize())
    vals = sched[field].reindex(dates)
    return pd.Series(vals.to_numpy(), index=range(len(index)), dtype=sched[field].dtype)


def session_ordinals(index: pd.DatetimeIndex, is_crypto: bool) -> np.ndarray:
    """Integer session id per bar (consecutive trading sessions differ by 1).

    Used to count how many session closes a forecast window crosses. Crypto
    has no sessions, so every bar gets 0.
    """
    if is_crypto or len(index) == 0:
        return np.zeros(len(index), dtype=int)
    index = to_eastern(index)
    sched = nyse_schedule(index[0], index[-1])
    ordinal = {d: i for i, d in enumerate(sched.index)}
    dates = index.tz_localize(None).normalize()
    return np.array([ordinal.get(d, -1) for d in dates], dtype=int)


def regular_session_only(df: pd.DataFrame, interval_min: int) -> pd.DataFrame:
    """Drop pre/post-market bars from intraday stock data."""
    if interval_min >= DAILY_MIN or df.empty:
        return df
    idx = to_eastern(df.index)
    opens = _session_field(idx, "market_open")
    closes = _session_field(idx, "market_close")
    starts = pd.Series(idx, index=opens.index)
    keep = opens.notna() & (starts >= opens) & (starts < closes)
    return df.loc[keep.to_numpy()]


def completed_bars(df: pd.DataFrame, interval_min: int, is_crypto: bool, as_of: datetime) -> pd.DataFrame:
    """Keep only bars that had fully closed by ``as_of``.

    yfinance returns the in-progress bar; its volume and candle shape are
    partial, which corrupts the features of the very row we predict from.
    """
    if df.empty:
        return df
    ends = bar_ends(df.index, interval_min, is_crypto)
    return df.loc[np.asarray(ends <= pd.Timestamp(as_of))]


@dataclass(frozen=True)
class Horizon:
    """How far ahead a forecast reaches, measured in bars of the model interval."""

    bars: int
    anchor_time: pd.Timestamp          # close time of the last completed bar
    effective_target: pd.Timestamp     # close time of the bar the forecast resolves on
    sessions_crossed: int              # session closes between anchor and target (stocks)

    @property
    def description(self) -> str:
        return f"{self.bars} bars, resolves at {self.effective_target:%a %b %d %I:%M %p} ET"


def _future_stock_bar_ends(start: pd.Timestamp, end: pd.Timestamp, interval_min: int):
    """(bar_end, session_ordinal_offset) for every RTH bar ending in (start, end]."""
    sched = nyse_schedule(start - pd.Timedelta(days=1), end)
    out = []
    step = pd.Timedelta(minutes=interval_min)
    for i, (_, row) in enumerate(sched.iterrows()):
        o, c = row["market_open"], row["market_close"]
        if c <= start:
            continue
        if interval_min >= DAILY_MIN:
            if c <= end:
                out.append((c, i))
            continue
        t = o
        while t < c:
            e = min(t + step, c)
            if start < e <= end and t >= start - pd.Timedelta(seconds=1):
                out.append((e, i))
            t += step
    return out, sched


def compute_horizon(
    last_bar_start: pd.Timestamp,
    target: datetime,
    interval_min: int,
    is_crypto: bool,
) -> Horizon:
    """Count bars from the last completed bar to ``target``.

    Only bars that END at or before ``target`` count, so the forecast resolves
    on a real bar close (reported as ``effective_target``).
    """
    last_bar_start = pd.Timestamp(last_bar_start).tz_convert(US_EASTERN)
    target = pd.Timestamp(target).tz_convert(US_EASTERN)
    anchor = bar_ends(pd.DatetimeIndex([last_bar_start]), interval_min, is_crypto)[0]

    if target <= anchor:
        raise ValueError(
            f"Target {target:%Y-%m-%d %H:%M} ET is not after the last completed "
            f"bar close ({anchor:%Y-%m-%d %H:%M} ET)."
        )

    step = pd.Timedelta(minutes=interval_min)
    if is_crypto:
        bars = int((target - anchor) // step)
        eff = anchor + bars * step
        crossed = 0
    else:
        ends, sched = _future_stock_bar_ends(anchor, target, interval_min)
        bars = len(ends)
        eff = ends[-1][0] if ends else anchor
        anchor_date = pd.Timestamp(last_bar_start.date())
        dates = list(sched.index)
        anchor_ord = dates.index(anchor_date) if anchor_date in dates else 0
        crossed = (ends[-1][1] - anchor_ord) if ends else 0

    if bars <= 0:
        raise ValueError(
            f"No complete {interval_min}-minute bar closes between "
            f"{anchor:%Y-%m-%d %H:%M} and {target:%Y-%m-%d %H:%M} ET. "
            "Pick a later target or a shorter interval."
        )
    return Horizon(bars=bars, anchor_time=anchor, effective_target=eff, sessions_crossed=int(crossed))
