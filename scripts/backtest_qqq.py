#!/usr/bin/env python3
"""
QQQ same-day backtest: predict 4 PM ET market close from various intraday times.

- Fetches up to 2 years of training data from Polygon.io (falls back to yfinance
  if POLYGON_API_KEY is not set, which limits history to ~60 days)
- Logs every prediction to Supabase backtest_logs (via Predictor.predict_xgb)
- Cross-checks candles_ahead against the expected value derived from the last
  candle timestamp and interval width
- Prints calibration and accuracy summary at the end

Usage:
    python scripts/backtest_qqq.py
    python scripts/backtest_qqq.py --days 10 --intervals 15m 1h --strikes -3 -2 -1 1 2 3
    python scripts/backtest_qqq.py --history-days 365  # training window size
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from leveledge import Predictor, fetch_polygon
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TICKER           = "QQQ"
TARGET_TIME      = time(16, 0)
DEF_INTERVALS    = ["15m", "30m", "1h"]
DEF_STRIKES      = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
DEF_END_TIMES    = [time(10, 30), time(12, 0), time(13, 30), time(15, 0)]
DEF_N_DAYS       = 5
DEF_HISTORY_DAYS = 730   # ~2 years — Polygon free tier limit for intraday

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prefetch_polygon_data(
    ticker: str, intervals: list[str], history_days: int
) -> dict[str, pd.DataFrame]:
    """
    Pre-fetch Polygon aggregate bars for every interval in one go.
    Returns {interval: DataFrame} keyed by interval string.
    Falls back to an empty dict if POLYGON_API_KEY is not set (callers then
    use yfinance's 60-day window via Predictor's default fetch).
    """
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("  [info] POLYGON_API_KEY not set — using yfinance (60-day window).")
        return {}

    end   = datetime.now(tz=US_EASTERN)
    start = end - timedelta(days=history_days)

    import time as _time

    cache: dict[str, pd.DataFrame] = {}
    for i, interval in enumerate(intervals):
        if i > 0:
            _time.sleep(13)  # respect 5 req/min between interval fetches
        print(f"  Fetching {ticker} {interval} from Polygon ({history_days}d)...", end=" ", flush=True)
        try:
            df = fetch_polygon(ticker, interval, start, end, api_key=api_key)
            cache[interval] = df
            print(f"{len(df)} bars")
        except Exception as e:
            print(f"ERROR: {e}")
    return cache


def get_trading_days(n: int, start_date: date | None = None, end_date: date | None = None) -> list[date]:
    """
    Return Mon-Fri trading days (holidays not filtered).

    - start_date + end_date: return all weekdays in [start_date, end_date]
    - start_date only: return n weekdays starting from start_date
    - neither: return the last n completed weekdays ending yesterday
    """
    if start_date and end_date:
        days = []
        d = start_date
        while d <= end_date:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        return days

    if start_date:
        days = []
        d = start_date
        today = datetime.now(tz=US_EASTERN).date()
        while len(days) < n and d < today:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        return days

    # default: last n completed weekdays
    today = datetime.now(tz=US_EASTERN).date()
    days = []
    d = today - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


def fetch_actual_closes(ticker: str, days: list[date]) -> dict[date, float]:
    """
    Fetch the actual close price at 4 PM ET for each trading day.
    Uses the last 15m candle at or before 4 PM (opens at 15:45, closes at 16:00).
    """
    actual: dict[date, float] = {}
    try:
        start = datetime.combine(min(days), time(15, 30), tzinfo=US_EASTERN)
        end   = datetime.combine(max(days), time(16, 30), tzinfo=US_EASTERN)
        hist  = yf.Ticker(ticker).history(start=start, end=end, interval="15m")
        if hist.empty:
            return actual
        hist.index = hist.index.tz_convert(US_EASTERN)
        for d in days:
            tgt    = datetime.combine(d, TARGET_TIME, tzinfo=US_EASTERN)
            subset = hist[(hist.index.date == d) & (hist.index <= tgt)]
            if not subset.empty:
                actual[d] = float(subset["Close"].iloc[-1])
    except Exception as e:
        print(f"  [warn] Could not fetch actual close prices: {e}")
    return actual


def expected_candles_same_day(last_candle_ts, target_dt: datetime, interval_min: int) -> int:
    """
    Expected candles_ahead for a same-day prediction.
    Mirrors the predictor formula: int((market_minutes - 1) / interval_min)
    where market_minutes is the raw time delta (no overnight subtraction needed
    since both times are within the same trading session).
    """
    delta_min = (target_dt - last_candle_ts).total_seconds() / 60
    return int((delta_min - 1) / interval_min)


# ---------------------------------------------------------------------------
# Polygon slice helper
# ---------------------------------------------------------------------------

def _slice_polygon(
    cache: dict[str, pd.DataFrame], interval: str, end_dt: datetime
) -> pd.DataFrame | None:
    """
    Return a copy of the cached Polygon DataFrame sliced to rows <= end_dt.
    Returns None if the interval is not in the cache (caller falls back to yfinance).
    """
    if interval not in cache or cache[interval].empty:
        return None
    df = cache[interval]
    end_ts = pd.Timestamp(end_dt).tz_convert(df.index.tzinfo)
    sliced = df[df.index <= end_ts]
    return sliced if not sliced.empty else None


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    n_days: int = DEF_N_DAYS,
    intervals: list[str] = DEF_INTERVALS,
    strike_pct: list[float] = DEF_STRIKES,
    end_times: list[time] = DEF_END_TIMES,
    history_days: int = DEF_HISTORY_DAYS,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    trading_days = get_trading_days(n_days, start_date=start_date, end_date=end_date)

    print("=" * 72)
    print(f"QQQ Backtest  |  target: 4 PM ET  |  {trading_days[0]} → {trading_days[-1]}")
    print("=" * 72)
    print(f"Days      : {[str(d) for d in trading_days]}")
    print(f"End times : {[str(t) for t in end_times]}")
    print(f"Intervals : {intervals}")
    print(f"Strikes   : {strike_pct}")

    print("\nPre-fetching Polygon training data ...")
    polygon_cache = prefetch_polygon_data(TICKER, intervals, history_days)

    print("\nFetching actual 4 PM close prices ...")
    actual_prices = fetch_actual_closes(TICKER, trading_days)
    for d, p in actual_prices.items():
        print(f"  {d}  ${p:.2f}")

    total = (
        sum(1 for d in trading_days if d in actual_prices)
        * len(end_times)
        * len(intervals)
        * len(strike_pct)
    )
    print(f"\nTotal predictions to run: {total}\n")

    results: list[dict] = []
    errors:  list[tuple] = []
    done = 0

    for day in trading_days:
        actual_4pm = actual_prices.get(day)
        if actual_4pm is None:
            print(f"[{day}] No actual price — skipping")
            continue

        target_dt = datetime.combine(day, TARGET_TIME, tzinfo=US_EASTERN)

        for end_time in end_times:
            end_dt = datetime.combine(day, end_time, tzinfo=US_EASTERN)

            # Resolve current price once per (day, end_time) slot.
            current_price: float | None = None
            for probe_intv in intervals:
                try:
                    probe_data = _slice_polygon(polygon_cache, probe_intv, end_dt)
                    probe = Predictor(
                        TICKER, target_dt, probe_intv, 999.0,
                        end_datetime=end_dt, data=probe_data,
                    )
                    current_price = probe.current_price
                    break
                except Exception:
                    continue

            if current_price is None:
                print(f"  [{day} {end_time}] Could not resolve current price — skipping slot")
                errors.append((day, end_time, None, None, "no current price"))
                done += len(intervals) * len(strike_pct)
                continue

            price_levels = [round(current_price * (1 + pct / 100), 2) for pct in strike_pct]

            for interval in intervals:
                for price, pct in zip(price_levels, strike_pct):
                    done += 1
                    label = f"[{done:>3}/{total}] {day} {end_time} {interval:>3s} {pct:+.0f}%"
                    print(f"  {label}  ", end="", flush=True)
                    try:
                        intv_data = _slice_polygon(polygon_cache, interval, end_dt)
                        p = Predictor(
                            TICKER, target_dt, interval, price,
                            end_datetime=end_dt, data=intv_data,
                        )

                        # ── candles_ahead cross-check ──────────────────────────
                        # p.data has RangeIndex after _create_target_variable resets it;
                        # p.data_withna retains the original DatetimeIndex.
                        last_ts   = p.data_withna.index[-1]
                        exp_ca    = expected_candles_same_day(last_ts, target_dt, p.interval_min)
                        ca        = p.candles_ahead
                        ca_ok     = abs(ca - exp_ca) <= 2   # ±2 candle tolerance

                        p.train_xgb()
                        pred = p.predict_xgb()   # also logs to Supabase backtest_logs

                        actual_hit = int(actual_4pm > price)

                        results.append({
                            "day":           day,
                            "end_time":      str(end_time),
                            "interval":      interval,
                            "strike_pct":    pct,
                            "current":       round(current_price, 2),
                            "price_level":   price,
                            "actual_4pm":    round(actual_4pm, 2),
                            "actual_hit":    actual_hit,
                            "prediction":    round(pred, 3),
                            "candles_ahead": ca,
                            "exp_candles":   exp_ca,
                            "ca_ok":         ca_ok,
                            "last_candle":   str(last_ts),
                        })

                        status  = "HIT " if actual_hit else "MISS"
                        ca_flag = "" if ca_ok else "  !! CA MISMATCH !!"
                        print(f"pred={pred:.3f}  actual={status}  ca={ca}/{exp_ca}{ca_flag}")

                    except Exception as e:
                        print(f"ERROR: {e}")
                        errors.append((day, end_time, interval, pct, str(e)))

    return pd.DataFrame(results), errors


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, errors: list[tuple]) -> None:
    if df.empty:
        print("\nNo results.")
        return

    df = df.copy()
    df["pred_hit"] = (df["prediction"] >= 0.6).astype(int)

    # ── candles validation ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CANDLES-AHEAD VALIDATION")
    print("=" * 72)
    n_bad = int((~df["ca_ok"]).sum())
    print(f"  {len(df)} predictions — {df['ca_ok'].sum()} OK,  {n_bad} mismatches (tolerance ±2 candles)")
    if n_bad:
        bad = df[~df["ca_ok"]][
            ["day", "end_time", "interval", "candles_ahead", "exp_candles", "last_candle"]
        ]
        print(bad.to_string(index=False))

    # ── sample candles check (one row per interval) ────────────────────────
    print("\n  Sample candles_ahead by interval (first occurrence each):")
    sample = (
        df.groupby("interval", group_keys=False)
        .apply(lambda g: g.iloc[0])
        [["interval", "end_time", "candles_ahead", "exp_candles", "last_candle"]]
    )
    print(sample.to_string(index=False))

    # ── calibration ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CALIBRATION — actual hit rate vs model prediction bucket")
    print("  (well-calibrated: hit rate in each bucket ≈ bucket midpoint)")
    print("=" * 72)
    bins   = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    labels = ["0-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
    df["bucket"] = pd.cut(df["prediction"], bins=bins, labels=labels, right=False)
    cal = (
        df.groupby("bucket", observed=True)
        .agg(n=("actual_hit", "count"), hit_rate=("actual_hit", "mean"), avg_pred=("prediction", "mean"))
        .round(3)
    )
    print(cal.to_string())

    # ── accuracy by interval ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ACCURACY BY INTERVAL  (threshold ≥ 0.6 → predict HIT)")
    print("=" * 72)
    rows = []
    for intv, g in df.groupby("interval"):
        called  = g[g["pred_hit"] == 1]
        bull_ok = int((called["actual_hit"] == 1).sum())
        rows.append({
            "interval":     intv,
            "n":            len(g),
            "base_rate":    round(g["actual_hit"].mean(), 3),
            "acc":          round((g["pred_hit"] == g["actual_hit"]).mean(), 3),
            "precision":    round(called["actual_hit"].mean(), 3) if len(called) else float("nan"),
            "preds_called": len(called),
        })
    print(pd.DataFrame(rows).set_index("interval").to_string())

    # ── accuracy by strike ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ACCURACY BY STRIKE %")
    print("=" * 72)
    rows = []
    for pct, g in df.groupby("strike_pct"):
        called = g[g["pred_hit"] == 1]
        rows.append({
            "strike_pct":   pct,
            "n":            len(g),
            "base_rate":    round(g["actual_hit"].mean(), 3),
            "acc":          round((g["pred_hit"] == g["actual_hit"]).mean(), 3),
            "avg_pred":     round(g["prediction"].mean(), 3),
            "precision":    round(called["actual_hit"].mean(), 3) if len(called) else float("nan"),
        })
    print(pd.DataFrame(rows).set_index("strike_pct").to_string())

    # ── accuracy by prediction time ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ACCURACY BY PREDICTION TIME")
    print("=" * 72)
    rows = []
    for et, g in df.groupby("end_time"):
        called = g[g["pred_hit"] == 1]
        rows.append({
            "end_time":  et,
            "n":         len(g),
            "acc":       round((g["pred_hit"] == g["actual_hit"]).mean(), 3),
            "precision": round(called["actual_hit"].mean(), 3) if len(called) else float("nan"),
        })
    print(pd.DataFrame(rows).set_index("end_time").to_string())

    # ── raw results ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RAW RESULTS")
    print("=" * 72)
    display_cols = [
        "day", "end_time", "interval", "strike_pct",
        "current", "price_level", "actual_4pm",
        "actual_hit", "prediction", "candles_ahead", "exp_candles", "ca_ok",
    ]
    print(df[display_cols].to_string(index=False))

    if errors:
        print(f"\n{len(errors)} errors during backtest:")
        for e in errors[:15]:
            print(f"  {e}")

    print(f"\nAll predictions logged to Supabase → backtest_logs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QQQ same-day backtest → 4 PM ET")
    parser.add_argument("--days",         type=int,   default=DEF_N_DAYS,
                        help="Number of recent trading days (default 5); ignored if --start-date set")
    parser.add_argument("--intervals",    nargs="+",  default=DEF_INTERVALS,
                        help="Intervals to test (default: 15m 30m 1h)")
    parser.add_argument("--strikes",      nargs="+",  type=float, default=DEF_STRIKES,
                        help="Strike %% offsets (default: -3 -2 -1 1 2 3)")
    parser.add_argument("--history-days", type=int,   default=DEF_HISTORY_DAYS,
                        help="Training history window in days via Polygon (default 730 = ~2yr)")
    parser.add_argument("--start-date",   type=date.fromisoformat,
                        help="Start date for backtest range (YYYY-MM-DD)")
    parser.add_argument("--end-date",     type=date.fromisoformat,
                        help="End date for backtest range (YYYY-MM-DD); requires --start-date")
    args = parser.parse_args()

    df, errors = run_backtest(
        n_days=args.days,
        intervals=args.intervals,
        strike_pct=args.strikes,
        history_days=args.history_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print_summary(df, errors)


if __name__ == "__main__":
    main()
