#!/usr/bin/env python3
"""
Longer-horizon edge test: forecasts made in the morning for the 4 PM settle.

Same protocol and scoring as kalshi_edge.py, with three changes:

* hourly bars (yfinance keeps ~2 years), so the sample spans Jan 2024 to now;
* only 4 PM S&P 500 and Nasdaq-100 above/below events (including the older
  INXDU/NASDAQ100U-named daily series);
* decisions at 10:30, 12:30 and 14:30 ET (330/210/90 minutes before the
  close), the first times a completed hourly bar exists for each.

Usage: python benchmarks/kalshi_horizon.py [--score-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["LEVELEDGE_DISABLE_LOGGING"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import kalshi_edge as K  # noqa: E402
from leveledge.constants import US_EASTERN as ET  # noqa: E402
from leveledge.data import fetch_history  # noqa: E402
from leveledge.kalshi import SERIES, KalshiClient, markets_frame  # noqa: E402

INTERVAL = "1h"
OFFSETS = [330, 210, 90]
START = pd.Timestamp("2024-01-08", tz=ET)        # after hourly-bar warm-up
BAND = {"^GSPC": 0.02, "^NDX": 0.025}
OUT = os.path.join(os.path.dirname(HERE), "benchmarks", "results", "kalshi_horizon")
CACHE = os.path.join(K.CACHE, "horizon")


def four_pm_events(client: KalshiClient, series: str) -> list[str]:
    evs = client._paginate("/events", "events", {"series_ticker": series, "status": "settled", "limit": 200},
                           max_pages=60)
    out = []
    for e in evs:
        t = e["event_ticker"]
        head, _, tail = t.partition("-")
        if (head.startswith("KX") and tail.endswith("H1600")) or (not head.startswith("KX") and "H" not in tail):
            out.append(t)
    return sorted(set(out))


def event_markets(client: KalshiClient, ev: str) -> pd.DataFrame:
    rows = client.get("/historical/markets", {"event_ticker": ev, "limit": 1000}).get("markets", [])
    if not rows:
        rows = client.get("/markets", {"event_ticker": ev, "limit": 1000}).get("markets", [])
    return markets_frame(rows)


def collect(series: str, client: KalshiClient):
    os.makedirs(CACHE, exist_ok=True)
    spec = SERIES[series]
    bp = os.path.join(CACHE, f"bars_{spec.underlying.replace('^', '')}_1h.parquet")
    bars = pd.read_parquet(bp) if os.path.exists(bp) else fetch_history(spec.underlying, INTERVAL)
    bars.to_parquet(bp)

    mp, cp = os.path.join(CACHE, f"markets_{series}.parquet"), os.path.join(CACHE, f"candles_{series}.parquet")
    markets = pd.read_parquet(mp) if os.path.exists(mp) else pd.DataFrame()
    candles = pd.read_parquet(cp) if os.path.exists(cp) else pd.DataFrame()
    done = set(markets["event_ticker"].unique()) if not markets.empty else set()
    evs = [e for e in four_pm_events(client, series) if e not in done]
    print(f"[{series}] {len(evs)} 4 PM events to fetch", flush=True)
    new_m, new_c = [], []
    for i, ev in enumerate(evs):
        m = event_markets(client, ev)
        if m.empty:
            new_m.append(pd.DataFrame({"event_ticker": [ev]}))   # remember empties
            continue
        m = m[m["y"].notna()]
        T = m["close_time"].iloc[0] if len(m) else None
        new_m.append(m if len(m) else pd.DataFrame({"event_ticker": [ev]}))
        if T is None or T < START or T.hour != 16 or T.minute != 0 or T > bars.index[-1] + pd.Timedelta(hours=1):
            continue
        first = T - pd.Timedelta(minutes=max(OFFSETS))
        seen = bars[bars.index <= first - pd.Timedelta(hours=1)]
        if len(seen) < 300:
            continue
        spot = float(seen["Close"].iloc[-1])
        b = BAND[spec.underlying]
        sel = m[(m["strike"] >= spot * (1 - b)) & (m["strike"] <= spot * (1 + b))]
        if sel.empty:
            continue
        for off in OFFSETS:
            d = T - pd.Timedelta(minutes=off)
            c = client.candles(sel["ticker"].tolist(), d - pd.Timedelta(minutes=6), d)
            if not c.empty:
                c["event_ticker"] = ev
                new_c.append(c)
        if (i + 1) % 50 == 0:
            print(f"  [{series}] {i + 1}/{len(evs)}", flush=True)
            markets, candles = _save(markets, new_m, mp), _save(candles, new_c, cp)
            new_m, new_c = [], []
    markets, candles = _save(markets, new_m, mp), _save(candles, new_c, cp)
    markets = markets.dropna(subset=["ticker"])
    markets = markets[markets["y"].notna()].drop_duplicates("ticker")
    candles = backfill(series, client, bars, markets, candles, cp)
    return bars, markets, candles


BACKFILL_BAND = {"^GSPC": 0.01, "^NDX": 0.012}


def backfill(series, client, bars, markets, candles, cp):
    """Archived events: the batch endpoint returns nothing, so fetch each
    near-the-money market's minute candles from the per-market historical
    endpoint, one window covering all decision times."""
    spec = SERIES[series]
    have = set(candles["event_ticker"].unique()) if not candles.empty else set()
    if os.path.exists(cp + ".tried"):
        have |= set(open(cp + ".tried").read().split())
    evs = markets[(markets["close_time"] >= START) & (markets["close_time"].dt.hour == 16)
                  & (markets["close_time"].dt.minute == 0)
                  & (markets["close_time"] <= bars.index[-1] + pd.Timedelta(hours=1))]
    todo = sorted(set(evs["event_ticker"]) - have)
    print(f"[{series}] backfilling {len(todo)} archived events", flush=True)
    new, tried = [], []
    for i, ev in enumerate(todo):
        m = evs[evs["event_ticker"] == ev]
        T = m["close_time"].iloc[0]
        first = T - pd.Timedelta(minutes=max(OFFSETS))
        seen = bars[bars.index <= first - pd.Timedelta(hours=1)]
        tried.append(ev)
        if len(seen) < 300:
            continue
        spot = float(seen["Close"].iloc[-1])
        b = BACKFILL_BAND[spec.underlying]
        sel = m[(m["strike"] >= spot * (1 - b)) & (m["strike"] <= spot * (1 + b))]
        for t in sel["ticker"]:
            c = client.historical_candles(t, first - pd.Timedelta(minutes=6),
                                          T - pd.Timedelta(minutes=min(OFFSETS)))
            if not c.empty:
                c["event_ticker"] = ev
                new.append(c)
        if (i + 1) % 25 == 0:
            print(f"  [{series}] backfill {i + 1}/{len(todo)}", flush=True)
            candles = _save(candles, new, cp)
            new = []
            with open(cp + ".tried", "a") as f:
                f.write("\n".join(tried) + "\n")
            tried = []
    candles = _save(candles, new, cp)
    with open(cp + ".tried", "a") as f:
        f.write("\n".join(tried) + "\n")
    return candles


def _save(have, new, path):
    out = pd.concat([have] + new, ignore_index=True) if new else have
    if not out.empty:
        out.to_parquet(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "predictions.csv")
    if not args.score_only:
        client = KalshiClient()
        frames = []
        for s in ["KXINXU", "KXNASDAQ100U"]:
            bars, markets, candles = collect(s, client)
            if candles.empty:
                continue
            frames.append(K.build_rows(s, bars, markets, candles, args.procs, offsets=OFFSETS, interval=INTERVAL))
        pd.concat(frames, ignore_index=True).to_csv(rows_path, index=False)
    df = pd.read_csv(rows_path)
    for c in ("target", "decision"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(ET)
    res = K.score(df)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    L = res["liquid"]
    print(json.dumps({"rows": res["rows"], "events": res["events"], "period": res["period"],
                      "calibration": res["calibration"], "liquid_calibration": L["calibration"],
                      "liquid_v2_minus_market": L["brier_v2_minus_market"],
                      "liquid_primary": L["strategies"]["p_v2|theta=0.02|fee=0.07"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
