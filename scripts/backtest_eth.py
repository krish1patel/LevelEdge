#!/usr/bin/env python3
"""
ETH-USD crypto backtest: predict the next full hour's close from the current hour.

- Predictions made every UTC hour (default: 0–23)
- Target: current_hour + 1 (e.g. at 01:00 UTC, predict 02:00 UTC close)
- Intervals: 5m, 15m, 30m, 1h (all tested by default)
- Logs every prediction to Supabase backtest_logs (via Predictor.predict_xgb)
- Uses Polygon.io for 2-year training window if POLYGON_API_KEY is set

Usage:
    python scripts/backtest_eth.py
    python scripts/backtest_eth.py --days 3 --strikes 0.0
    python scripts/backtest_eth.py --start-date 2026-02-01 --end-date 2026-03-20 --strikes 0.0
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
TICKER           = "ETH-USD"
DEF_INTERVALS    = ["5m", "15m", "30m", "1h"]
DEF_STRIKES      = [0.0]
DEF_PRED_HOURS   = list(range(24))          # UTC hours to make predictions (every hour)
DEF_N_DAYS       = 5
DEF_HISTORY_DAYS = 365   # 1yr keeps 5m bars to ~1 Polygon page; use 730 for 15m+


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_days(
    n: int,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[date]:
    """Return calendar days (crypto trades 24/7, no weekday filter)."""
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


def fetch_actual_hourly_prices(
    days: list[date], pred_hours: list[int]
) -> dict[tuple[date, int], float]:
    """
    Fetch ETH-USD close at (pred_hour + 1) UTC for each (day, pred_hour) pair.
    E.g. for pred_hour=1 on 2026-03-23, returns the close at 02:00 UTC.
    """
    actual: dict[tuple[date, int], float] = {}
    try:
        start_dt = datetime.combine(min(days), time(0, 0), tzinfo=UTC)
        # +2 days covers the 23:00→00:00 wrap on the last day
        end_dt   = datetime.combine(max(days) + timedelta(days=2), time(0, 0), tzinfo=UTC)
        hist = yf.Ticker(TICKER).history(start=start_dt, end=end_dt, interval="1h")
        if hist.empty:
            return actual
        hist.index = hist.index.tz_convert(UTC)
        for d in days:
            for h in pred_hours:
                # target = 1 hour after prediction time
                pred_dt   = datetime.combine(d, time(h, 0), tzinfo=UTC)
                target_dt = pred_dt + timedelta(hours=1)
                subset = hist[hist.index <= target_dt]
                if not subset.empty:
                    actual[(d, h)] = float(subset["Close"].iloc[-1])
    except Exception as e:
        print(f"  [warn] Could not fetch actual hourly prices: {e}")
    return actual


def expected_candles_crypto(last_candle_ts: pd.Timestamp, target_dt: datetime, interval_min: int) -> int:
    """Expected candles_ahead for a crypto (24/7) prediction."""
    delta = target_dt - last_candle_ts - timedelta(minutes=1)
    return int(delta.total_seconds() / 60 / interval_min)


def prefetch_polygon_data(intervals: list[str], history_days: int) -> dict[str, pd.DataFrame]:
    """Pre-fetch Polygon bars for all intervals. Returns {} if no API key."""
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("  [info] POLYGON_API_KEY not set — using yfinance (60-day window).")
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
    cache: dict[str, pd.DataFrame], interval: str, end_dt: datetime
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
    pred_hours: list[int] = DEF_PRED_HOURS,
    history_days: int = DEF_HISTORY_DAYS,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, list]:

    days = get_days(n_days, start_date=start_date, end_date=end_date)

    print("=" * 72)
    print(f"ETH-USD Backtest  |  target: +1h  |  {days[0]} → {days[-1]}")
    print("=" * 72)
    print(f"Days         : {[str(d) for d in days]}")
    print(f"Pred hours   : {pred_hours} UTC")
    print(f"Intervals    : {intervals}")
    print(f"Strikes      : {strike_pct}")

    print("\nPre-fetching Polygon training data ...")
    polygon_cache = prefetch_polygon_data(intervals, history_days)

    print("\nFetching actual +1h prices ...")
    actual_prices = fetch_actual_hourly_prices(days, pred_hours)
    print(f"  Fetched {len(actual_prices)} (day, hour) price points")

    total = len(actual_prices) * len(intervals) * len(strike_pct)
    print(f"\nTotal predictions to run: {total}\n")

    results: list[dict] = []
    errors:  list[tuple] = []
    done = 0

    for day in days:
        for pred_hour in pred_hours:
            actual_next_hour = actual_prices.get((day, pred_hour))
            if actual_next_hour is None:
                done += len(intervals) * len(strike_pct)
                continue

            pred_dt   = datetime.combine(day, time(pred_hour, 0), tzinfo=UTC)
            target_dt = pred_dt + timedelta(hours=1)
            end_dt    = pred_dt  # data cutoff = prediction time

            # Resolve current price once per (day, hour)
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
                print(f"  [{day} {pred_hour:02d}:00 UTC] Could not resolve price — skipping")
                errors.append((day, pred_hour, None, None, "no current price"))
                done += len(intervals) * len(strike_pct)
                continue

            price_levels = [round(current_price * (1 + pct / 100), 2) for pct in strike_pct]

            for interval in intervals:
                for price, pct in zip(price_levels, strike_pct):
                    done += 1
                    label = f"[{done:>4}/{total}] {day} {pred_hour:02d}:00→{(pred_hour+1)%24:02d}:00 UTC {interval:>3s} {pct:+.0f}%"
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

                        actual_hit = int(actual_next_hour > price)

                        results.append({
                            "day":           day,
                            "pred_hour_utc": pred_hour,
                            "interval":      interval,
                            "strike_pct":    pct,
                            "current":       round(current_price, 2),
                            "price_level":   price,
                            "actual_close":  round(actual_next_hour, 2),
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
                        errors.append((day, pred_hour, interval, pct, str(e)))

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
    print(f"  {len(df)} predictions — {df['ca_ok'].sum()} OK,  {n_bad} mismatches (±2 candle tolerance)")
    if n_bad:
        bad = df[~df["ca_ok"]][["day", "pred_hour_utc", "interval", "candles_ahead", "exp_candles", "last_candle"]]
        print(bad.to_string(index=False))

    print("\n  Sample candles_ahead by interval (first occurrence each):")
    sample = (
        df.groupby("interval", group_keys=False)
        .apply(lambda g: g.iloc[0])
        [["interval", "pred_hour_utc", "candles_ahead", "exp_candles", "last_candle"]]
    )
    print(sample.to_string(index=False))

    print("\n" + "=" * 72)
    print("CALIBRATION — actual hit rate vs model prediction bucket")
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
    print("ACCURACY BY INTERVAL  (threshold ≥ 0.6 → predict HIT)")
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
    print("RAW RESULTS")
    print("=" * 72)
    display_cols = [
        "day", "pred_hour_utc", "interval", "strike_pct",
        "current", "price_level", "actual_close",
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
    parser = argparse.ArgumentParser(description="ETH-USD crypto backtest → +1h target")
    parser.add_argument("--days",         type=int,   default=DEF_N_DAYS,
                        help="Number of recent days (default 5); ignored if --start-date set")
    parser.add_argument("--intervals",    nargs="+",  default=DEF_INTERVALS,
                        help="Intervals to test (default: 5m 15m 30m 1h)")
    parser.add_argument("--strikes",      nargs="+",  type=float, default=DEF_STRIKES,
                        help="Strike %% offsets (default: -3 -2 -1 1 2 3)")
    parser.add_argument("--pred-hours",   nargs="+",  type=int,   default=DEF_PRED_HOURS,
                        help="UTC hours to make predictions (default: 0 4 8 12 16 20)")
    parser.add_argument("--history-days", type=int,   default=DEF_HISTORY_DAYS,
                        help="Polygon training window in days (default 730)")
    parser.add_argument("--start-date",   type=date.fromisoformat,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date",     type=date.fromisoformat,
                        help="End date (YYYY-MM-DD); requires --start-date")
    args = parser.parse_args()

    df, errors = run_backtest(
        n_days=args.days,
        intervals=args.intervals,
        strike_pct=args.strikes,
        pred_hours=args.pred_hours,
        history_days=args.history_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print_summary(df, errors)


if __name__ == "__main__":
    main()
