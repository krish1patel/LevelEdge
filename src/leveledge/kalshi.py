"""
Kalshi public market data: the market price LevelEdge is measured against.

Kalshi lists binary contracts that ask LevelEdge's exact question ("will the
S&P 500 be above K at 4pm?"). A contract pays $1 if YES, so its price is the
market's probability. Edge = model probability - price - fee.

Only the public, unauthenticated market-data endpoints are used. No orders are
ever sent from this module.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from leveledge.constants import US_EASTERN

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
USER_AGENT = "leveledge-research"
MAX_BATCH = 100          # tickers per batch-candlestick request (API cap)
MIN_INTERVAL_S = 0.12    # client-side pacing: stay well under the public rate limit


@dataclass(frozen=True)
class Series:
    """A Kalshi above/below series and the price it settles on."""

    series: str
    underlying: str          # yfinance symbol LevelEdge forecasts
    name: str
    settles_on: str
    fee_rate: float = 0.07   # Kalshi general taker fee: rate * P * (1 - P) per contract


# Hourly above/below ladders. Settlement sources are from the series metadata.
SERIES: dict[str, Series] = {
    "KXINXU": Series("KXINXU", "^GSPC", "S&P 500", "S&P 500 index value at the hour"),
    "KXNASDAQ100U": Series("KXNASDAQ100U", "^NDX", "Nasdaq-100", "Nasdaq-100 index value at the hour"),
    "KXBTCD": Series("KXBTCD", "BTC-USD", "Bitcoin", "60-second average of CF Benchmarks BRTI before the hour"),
    "KXETHD": Series("KXETHD", "ETH-USD", "Ethereum", "60-second average of CF Benchmarks ETH RTI before the hour"),
}


def taker_fee(price, rate: float = 0.07):
    """Per-contract taker fee in dollars, before Kalshi's round-up to the cent.

    Kalshi charges ``ceil(rate * C * P * (1 - P))`` cents on an order of C
    contracts at price P. Per contract on a sizeable order that is
    ``rate * P * (1 - P)``: at most 1.75c at P = 0.50 with the 0.07 rate.
    Some index series carry a reduced rate, so 0.07 is the conservative default.
    """
    return rate * price * (1.0 - price)


class KalshiClient:
    """Small, paced, retrying client for Kalshi's public market data."""

    def __init__(self, base_url: str = BASE_URL, min_interval: float = MIN_INTERVAL_S, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.timeout = timeout
        self._lock = threading.Lock()
        self._last = 0.0

    # ---- transport -------------------------------------------------------------
    def get(self, path: str, params: dict | None = None) -> dict:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v not in (None, "")})
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        delay = 1.0
        for attempt in range(6):
            with self._lock:
                wait = self._last + self.min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    time.sleep(delay)
                    delay = min(delay * 2, 20)
                    continue
                raise RuntimeError(f"Kalshi {e.code} for {path}: {e.read()[:200]!r}") from e
            except (urllib.error.URLError, TimeoutError):
                time.sleep(delay)
                delay = min(delay * 2, 20)
        raise RuntimeError(f"Kalshi request kept failing: {path}")

    def _paginate(self, path: str, key: str, params: dict, max_pages: int = 200) -> list[dict]:
        out, cursor = [], None
        for _ in range(max_pages):
            d = self.get(path, {**params, "cursor": cursor})
            out += d.get(key, []) or []
            cursor = d.get("cursor")
            if not cursor:
                break
        return out

    # ---- markets ---------------------------------------------------------------
    def markets(self, series: str, status: str | None = None, min_close: datetime | None = None,
                max_close: datetime | None = None, event_ticker: str | None = None) -> list[dict]:
        params = {"series_ticker": series, "status": status, "limit": 1000, "event_ticker": event_ticker,
                  "min_close_ts": _ts(min_close), "max_close_ts": _ts(max_close)}
        return self._paginate("/markets", "markets", params)

    def historical_markets(self, series: str, min_close: datetime | None = None,
                           max_close: datetime | None = None) -> list[dict]:
        """Older settled markets (before roughly the last two months).

        This endpoint ignores date filters and returns newest first, so paging
        stops once a page is entirely older than ``min_close``.
        """
        out, cursor = [], None
        floor = min_close.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if min_close else None
        for _ in range(500):
            d = self.get("/historical/markets", {"series_ticker": series, "limit": 1000, "cursor": cursor})
            page = d.get("markets", []) or []
            out += page
            cursor = d.get("cursor")
            if not cursor or (floor and page and max(m.get("close_time", "") for m in page) < floor):
                break
        if max_close is not None:
            ceil = max_close.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            out = [m for m in out if m.get("close_time", "") <= ceil]
        if floor:
            out = [m for m in out if m.get("close_time", "") >= floor]
        return out

    def settled_markets(self, series: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Every settled market in ``series`` closing in [start, end], both endpoints."""
        rows = self.markets(series, status="settled", min_close=start, max_close=end)
        rows += self.historical_markets(series, min_close=start, max_close=end)
        df = markets_frame(rows)
        if df.empty:
            return df
        df = df[(df["close_time"] >= pd.Timestamp(start)) & (df["close_time"] <= pd.Timestamp(end))]
        return df.drop_duplicates("ticker").reset_index(drop=True)

    def open_markets(self, series: str) -> pd.DataFrame:
        return markets_frame(self.markets(series, status="open"))

    # ---- prices ----------------------------------------------------------------
    def candles(self, tickers: list[str], start: datetime, end: datetime, period_min: int = 1) -> pd.DataFrame:
        """Minute (or hourly) bid/ask/trade candles for many markets.

        One row per (ticker, candle). ``ts`` is the candle's END time, so the
        quote at time t is the close of the candle ending at or before t.
        """
        frames = []
        for i in range(0, len(tickers), MAX_BATCH):
            chunk = tickers[i:i + MAX_BATCH]
            d = self.get("/markets/candlesticks", {
                "market_tickers": ",".join(chunk), "start_ts": _ts(start), "end_ts": _ts(end),
                "period_interval": period_min})
            for m in d.get("markets", []) or []:
                for c in m.get("candlesticks", []) or []:
                    frames.append(_candle_row(m.get("market_ticker"), c))
        df = pd.DataFrame(frames)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(US_EASTERN)
        return df.sort_values(["ticker", "ts"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def _ts(t: datetime | None) -> int | None:
    if t is None:
        return None
    if t.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return int(t.timestamp())


def _f(x) -> float | None:
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def _candle_row(ticker: str, c: dict) -> dict:
    bid, ask, px = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
    return {
        "ticker": ticker,
        "ts": c.get("end_period_ts"),
        "bid": _f(bid.get("close_dollars")),
        "ask": _f(ask.get("close_dollars")),
        "last": _f(px.get("close_dollars")),
        "volume": _f(c.get("volume_fp")) or 0.0,
        "open_interest": _f(c.get("open_interest_fp")) or 0.0,
    }


def markets_frame(rows: list[dict]) -> pd.DataFrame:
    """Normalise raw market dicts to the columns LevelEdge uses."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "ticker": r.get("ticker"),
        "event_ticker": r.get("event_ticker"),
        "title": r.get("title"),
        "strike_type": r.get("strike_type"),
        "strike": _f(r.get("floor_strike")),
        "cap_strike": _f(r.get("cap_strike")),
        "open_time": r.get("open_time"),
        "close_time": r.get("close_time"),
        "status": r.get("status"),
        "result": r.get("result"),
        "expiration_value": _f(r.get("expiration_value")),
        "bid": _f(r.get("yes_bid_dollars")),
        "ask": _f(r.get("yes_ask_dollars")),
        "bid_size": _f(r.get("yes_bid_size_fp")),
        "ask_size": _f(r.get("yes_ask_size_fp")),
        "last": _f(r.get("last_price_dollars")),
        "volume": _f(r.get("volume_fp")) or 0.0,
        "volume_24h": _f(r.get("volume_24h_fp")) or 0.0,
        "open_interest": _f(r.get("open_interest_fp")) or 0.0,
    } for r in rows])
    for col in ("open_time", "close_time"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(US_EASTERN)
    # LevelEdge forecasts P(price > K): keep the single-strike "above" contracts
    df = df[df["strike_type"].isin(["greater", "greater_or_equal"]) & df["cap_strike"].isna()
            & df["strike"].notna()]
    df["y"] = df["result"].map({"yes": 1, "no": 0})
    return df.reset_index(drop=True)


def quote_at(candles: pd.DataFrame, t: pd.Timestamp, max_stale_min: int = 5) -> pd.DataFrame:
    """Latest two-sided quote per ticker as of ``t`` (candle end <= t)."""
    if candles.empty:
        return candles
    c = candles[(candles["ts"] <= t) & (candles["ts"] > t - pd.Timedelta(minutes=max_stale_min))]
    c = c.dropna(subset=["bid", "ask"])
    c = c[(c["ask"] > c["bid"]) & (c["ask"] > 0) & (c["bid"] < 1)]
    return c.groupby("ticker", as_index=False).last()


def now_et() -> datetime:
    return datetime.now(tz=timezone.utc).astimezone(US_EASTERN)


__all__ = ["KalshiClient", "SERIES", "Series", "taker_fee", "markets_frame", "quote_at", "now_et"]
