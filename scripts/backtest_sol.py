#!/usr/bin/env python3
"""
SOL-USD crypto backtest: predict the next 15-minute close from every quarter hour.

- Predictions made every 15 min (x:00, x:15, x:30, x:45 UTC)
- Target: prediction_time + 15 min
- Intervals: 2m, 5m, 15m
- 3-day backtest period (default)

Usage:
    python scripts/backtest_sol.py
    python scripts/backtest_sol.py --days 5 --strikes 0.0
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
import warnings
from datetime import date, datetime, time, timedelta, timezone

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

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TICKER           = "SOL-USD"
DEF_INTERVALS    = ["2m", "5m", "15m"]
DEF_STRIKES      = [0.0]
DEF_PRED_MINUTES = [0, 15, 30, 45]     # minutes past each hour
DEF_N_DAYS       = 3
DEF_HISTORY_DAYS = 365


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_days(
    n: int,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[date]:
    """Return calendar days (crypto trades 24/7)."""
    today = datetime.now(tz=UTC).date()

    if start_date and end_date:
        days = []
        d = start_date
        while d <= end_date:
            if d < today:
                days.append(d)
            d += timedelta(days=1)
        return days

    if start_date:
        days = []
        d = start_date
        while len(days) < n and d < today:
            days.append(d)
            d += timedelta(days=1)
        return days

    # Default: last n completed days
    days = []
    d = today - timedelta(days=1)
    while len(days) < n:
        days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


def fetch_actual_15m_prices(
    days: list[date], pred_minutes: list[int],
) -> dict[tuple[date, int, int], float]:
    """
    Fetch SOL-USD close at (pred_time + 15 min) for each (day, hour, minute) tuple.
    Returns {(day, hour, minute): actual_close_price}.
    """
    actual: dict[tuple[date, int, int], float] = {}
    try:
        start_dt = datetime.combine(min(days), time(0, 0), tzinfo=UTC)
        end_dt   = datetime.combine(max(days) + timedelta(days=2), time(0, 0), tzinfo=UTC)
        hist = yf.Ticker(TICKER).history(start=start_dt, end=end_dt, interval="15m")
        if hist.empty:
            return actual
        hist.index = hist.index.tz_convert(UTC)
        for d in days:
            for h in range(24):
                for m in pred_minutes:
                    pred_dt   = datetime.combine(d, time(h, m), tzinfo=UTC)
                    target_dt = pred_dt + timedelta(minutes=15)
                    subset = hist[hist.index <= target_dt]
                    if not subset.empty:
                        actual[(d, h, m)] = float(subset["Close"].iloc[-1])
    except Exception as e:
        print(f"  [warn] Could not fetch actual 15m prices: {e}")
    return actual


def expected_candles_crypto(last_candle_ts: pd.Timestamp, target_dt: datetime, interval_min: int) -> int:
    delta = target_dt - last_candle_ts - timedelta(minutes=1)
    return int(delta.total_seconds() / 60 / interval_min)


def prefetch_polygon_data(intervals: list[str], history_days: int) -> dict[str, pd.DataFrame]:
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("  [info] POLYGON_API_KEY not set -- using yfinance (60-day window).")
        return {}

    end   = datetime.now(tz=UTC)
    start = end - timedelta(days=history_days)
    cache: dict[str, pd.DataFrame] = {}

    for i, interval in enumerate(intervals):
        if i > 0:
            _time.sleep(15)
        print(f"  Fetching {TICKER} {interval} from Polygon ({history_days}d)...", end=" ", flush=True)
        try:
            df = fetch_polygon(TICKER, interval, start, end, api_key=api_key)
            cache[interval] = df
            print(f"{len(df)} bars")
        except Exception as e:
            print(f"ERROR: {e}")

    return cache


def _slice_polygon(
    cache: dict[str, pd.DataFrame], interval: str, end_dt: datetime,
) -> pd.DataFrame | None:
    if interval not in cache or cache[interval].empty:
        return None
    df = cache[interval]
    end_ts = pd.Timestamp(end_dt).tz_convert(UTC)
    sliced = df[df.index <= end_ts]
    return sliced if not sliced.empty else None


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    n_days: int = DEF_N_DAYS,
    intervals: list[str] = DEF_INTERVALS,
    strike_pct: list[float] = DEF_STRIKES,
    pred_minutes: list[int] = DEF_PRED_MINUTES,
    history_days: int = DEF_HISTORY_DAYS,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, list]:

    days = get_days(n_days, start_date=start_date, end_date=end_date)

    print("=" * 72)
    print(f"SOL-USD Backtest  |  target: +15m  |  {days[0]} -> {days[-1]}")
    print("=" * 72)
    print(f"Days         : {[str(d) for d in days]}")
    print(f"Pred times   : every 15 min (:{pred_minutes})")
    print(f"Intervals    : {intervals}")
    print(f"Strikes      : {strike_pct}")

    print("\nPre-fetching Polygon training data ...")
    polygon_cache = prefetch_polygon_data(intervals, history_days)

    print("\nFetching actual +15m prices ...")
    actual_prices = fetch_actual_15m_prices(days, pred_minutes)
    print(f"  Fetched {len(actual_prices)} (day, hour, minute) price points")

    total = len(actual_prices) * len(intervals) * len(strike_pct)
    print(f"\nTotal predictions to run: {total}\n")

    results: list[dict] = []
    errors:  list[tuple] = []
    done = 0

    for day in days:
        for hour in range(24):
            for minute in pred_minutes:
                actual_close = actual_prices.get((day, hour, minute))
                if actual_close is None:
                    done += len(intervals) * len(strike_pct)
                    continue

                pred_dt   = datetime.combine(day, time(hour, minute), tzinfo=UTC)
                target_dt = pred_dt + timedelta(minutes=15)
                end_dt    = pred_dt  # data cutoff = prediction time

                # Resolve current price once per time slot
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
                    print(f"  [{day} {hour:02d}:{minute:02d} UTC] Could not resolve price -- skipping")
                    errors.append((day, hour, minute, None, None, "no current price"))
                    done += len(intervals) * len(strike_pct)
                    continue

                price_levels = [round(current_price * (1 + pct / 100), 2) for pct in strike_pct]

                for interval in intervals:
                    for price, pct in zip(price_levels, strike_pct):
                        done += 1
                        label = (
                            f"[{done:>4}/{total}] {day} "
                            f"{hour:02d}:{minute:02d}->{(hour + (minute+15)//60) % 24:02d}:{(minute+15)%60:02d} UTC "
                            f"{interval:>3s} {pct:+.0f}%"
                        )
                        print(f"  {label}  ", end="", flush=True)
                        try:
                            intv_data = _slice_polygon(polygon_cache, interval, end_dt)
                            p = Predictor(
                                TICKER, target_dt, interval, price,
                                end_datetime=end_dt, data=intv_data,
                            )

                            last_ts  = p.data_withna.index[-1]
                            exp_ca   = expected_candles_crypto(last_ts, target_dt, p.interval_min)
                            ca       = p.candles_ahead
                            ca_ok    = abs(ca - exp_ca) <= 2

                            p.train_xgb()
                            pred = p.predict_xgb()

                            actual_hit = int(actual_close > price)

                            results.append({
                                "day":           day,
                                "pred_hour_utc": hour,
                                "pred_min_utc":  minute,
                                "interval":      interval,
                                "strike_pct":    pct,
                                "current":       round(current_price, 2),
                                "price_level":   price,
                                "actual_close":  round(actual_close, 2),
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
                            errors.append((day, hour, minute, interval, pct, str(e)))

    return pd.DataFrame(results), errors


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, errors: list) -> None:
    if df.empty:
        print("\nNo results.")
        return

    df = df.copy()
    df["pred_hit"] = (df["prediction"] >= 0.6).astype(int)

    print("\n" + "=" * 72)
    print("CANDLES-AHEAD VALIDATION")
    print("=" * 72)
    n_bad = int((~df["ca_ok"]).sum())
    print(f"  {len(df)} predictions -- {df['ca_ok'].sum()} OK,  {n_bad} mismatches (+/-2 candle tolerance)")
    if n_bad:
        bad = df[~df["ca_ok"]][["day", "pred_hour_utc", "pred_min_utc", "interval", "candles_ahead", "exp_candles", "last_candle"]]
        print(bad.to_string(index=False))

    print("\n" + "=" * 72)
    print("CALIBRATION -- actual hit rate vs model prediction bucket")
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

    print("\n" + "=" * 72)
    print("ACCURACY BY INTERVAL  (threshold >= 0.6 -> predict HIT)")
    print("=" * 72)
    rows = []
    for intv, g in df.groupby("interval"):
        called = g[g["pred_hit"] == 1]
        rows.append({
            "interval":     intv,
            "n":            len(g),
            "base_rate":    round(g["actual_hit"].mean(), 3),
            "acc":          round((g["pred_hit"] == g["actual_hit"]).mean(), 3),
            "precision":    round(called["actual_hit"].mean(), 3) if len(called) else float("nan"),
            "preds_called": len(called),
        })
    print(pd.DataFrame(rows).set_index("interval").to_string())

    print("\n" + "=" * 72)
    print("ACCURACY BY STRIKE %")
    print("=" * 72)
    rows = []
    for pct, g in df.groupby("strike_pct"):
        called = g[g["pred_hit"] == 1]
        rows.append({
            "strike_pct": pct,
            "n":          len(g),
            "base_rate":  round(g["actual_hit"].mean(), 3),
            "acc":        round((g["pred_hit"] == g["actual_hit"]).mean(), 3),
            "avg_pred":   round(g["prediction"].mean(), 3),
            "precision":  round(called["actual_hit"].mean(), 3) if len(called) else float("nan"),
        })
    print(pd.DataFrame(rows).set_index("strike_pct").to_string())

    print("\n" + "=" * 72)
    print("ACCURACY BY PREDICTION HOUR (UTC)")
    print("=" * 72)
    rows = []
    for hr, g in df.groupby("pred_hour_utc"):
        called = g[g["pred_hit"] == 1]
        rows.append({
            "pred_hour_utc": hr,
            "n":             len(g),
            "base_rate":     round(g["actual_hit"].mean(), 3),
            "acc":           round((g["pred_hit"] == g["actual_hit"]).mean(), 3),
            "precision":     round(called["actual_hit"].mean(), 3) if len(called) else float("nan"),
            "preds_called":  len(called),
        })
    print(pd.DataFrame(rows).set_index("pred_hour_utc").to_string())

    print("\n" + "=" * 72)
    print("OVERALL TOTALS")
    print("=" * 72)
    total_n = len(df)
    overall_acc = round((df["pred_hit"] == df["actual_hit"]).mean(), 3)
    overall_base = round(df["actual_hit"].mean(), 3)
    called = df[df["pred_hit"] == 1]
    overall_prec = round(called["actual_hit"].mean(), 3) if len(called) else float("nan")
    print(f"  Total predictions : {total_n}")
    print(f"  Base rate         : {overall_base}")
    print(f"  Overall accuracy  : {overall_acc}")
    print(f"  Overall precision : {overall_prec}  (n={len(called)} calls at >=0.6)")
    print(f"  Errors            : {len(errors)}")

    print("\n" + "=" * 72)
    print("RAW RESULTS")
    print("=" * 72)
    display_cols = [
        "day", "pred_hour_utc", "pred_min_utc", "interval", "strike_pct",
        "current", "price_level", "actual_close",
        "actual_hit", "prediction", "candles_ahead", "exp_candles", "ca_ok",
    ]
    print(df[display_cols].to_string(index=False))

    if errors:
        print(f"\n{len(errors)} errors during backtest:")
        for e in errors[:15]:
            print(f"  {e}")

    print(f"\nAll predictions logged to Supabase -> backtest_logs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SOL-USD crypto backtest -> +15m target")
    parser.add_argument("--days",         type=int,   default=DEF_N_DAYS)
    parser.add_argument("--intervals",    nargs="+",  default=DEF_INTERVALS)
    parser.add_argument("--strikes",      nargs="+",  type=float, default=DEF_STRIKES)
    parser.add_argument("--history-days", type=int,   default=DEF_HISTORY_DAYS)
    parser.add_argument("--start-date",   type=date.fromisoformat)
    parser.add_argument("--end-date",     type=date.fromisoformat)
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
