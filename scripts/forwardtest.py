"""
Forward Test: QQQ, SPY, NVDA, TSLA
====================================
Runs predictions 3x per day (10am, 12pm, 2pm ET) on every supported interval
across 8 strikes per ticker (±1%, ±1.5%, ±2%, ±3% of current price at
prediction time). Target is always 4pm ET same day.

All predictions are logged to Supabase → forwardtest_logs.

Schedule via Task Scheduler or GitHub Actions cron:
  10:00 AM EDT  →  python forward_test.py   (cron: 0 14 * * 1-5)
  12:00 PM EDT  →  python forward_test.py   (cron: 0 16 * * 1-5)
   2:00 PM EDT  →  python forward_test.py   (cron: 0 18 * * 1-5)
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
os.environ.setdefault("LEVELEDGE_DISABLE_LOGGING", "1")  # this script logs to forwardtest_logs itself

from dotenv import load_dotenv
from supabase import create_client

from leveledge import Predictor, forecast
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKERS    = ["QQQ", "SPY", "NVDA", "TSLA"]
STRIKE_PCT = (-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0)
INTERVALS  = [i for i in ALLOWED_INTERVALS if i not in ("1m", "1d", "90m")]
TABLE_NAME = "forwardtest_logs"

# ---------------------------------------------------------------------------
# Supabase logging
# ---------------------------------------------------------------------------

def log_prediction(sb, p: Predictor, prediction: float, run_time: datetime) -> None:
    """Log one prediction to forwardtest_logs (same columns as v1, plus model_type)."""
    try:
        record = p.log_record(prediction)
        record["logged_at_utc"] = run_time.isoformat()
        record["outcome_price"] = None
        sb.table(TABLE_NAME).insert(record).execute()
    except Exception as e:
        print(f"    [warn] Supabase log failed ({p.ticker_str} {p.interval} ${p.price:.2f}): {e}")


# ---------------------------------------------------------------------------
# Core: run predictions for one ticker
# ---------------------------------------------------------------------------

def run_ticker(ticker: str, target_dt: datetime, end_dt: datetime, sb, run_time: datetime) -> None:
    """One model fit per interval; every strike is a query against that fit."""
    total = errors = 0
    for interval in INTERVALS:
        try:
            model = forecast(ticker, target_dt, interval)
        except Exception as e:
            print(f"    [{ticker}] {interval}: {e}")
            errors += len(STRIKE_PCT)
            continue
        spot = model.current_price
        levels = [round(spot * (1 + pct / 100.0), 2) for pct in STRIKE_PCT]
        if interval == INTERVALS[0]:
            print(f"  [{ticker}] Price: ${spot:.2f} | Levels: {levels}")
        for price in levels:
            try:
                p = Predictor(ticker, target_dt, interval, price)
                p.train_xgb()            # cache hit: reuses the fit above
                prediction = p.predict_xgb()
                log_prediction(sb, p, prediction, run_time)
                total += 1
            except Exception as e:
                print(f"    [{ticker}] {interval} ${price:.2f}: {e}")
                errors += 1
    print(f"  [{ticker}] Done — {total} predictions logged, {errors} errors")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    now_et    = datetime.now(tz=US_EASTERN)
    run_time  = now_et
    today     = now_et.date()
    target_dt = datetime(today.year, today.month, today.day, 16, 0, tzinfo=US_EASTERN)
    end_dt    = now_et

    from leveledge.sessions import nyse_schedule
    sched = nyse_schedule(today, today)
    if sched.empty:
        print(f"{today} is not an NYSE trading day. Exiting.")
        return
    close = sched["market_close"].iloc[0]
    if close < target_dt:
        print(f"Early close today ({close:%H:%M} ET); targeting the close instead of 4 PM.")
        target_dt = close.to_pydatetime()
    if end_dt >= target_dt:
        print(f"Today's close ({target_dt:%H:%M} ET) has already passed. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"Forward Test — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"Target: {target_dt:%H:%M} ET | Tickers: {TICKERS}")
    print(f"Intervals: {INTERVALS}")
    print(f"Strikes: {STRIKE_PCT}%")
    print(f"{'='*60}\n")

    for ticker in TICKERS:
        print(f"Running {ticker}...")
        run_ticker(ticker, target_dt, end_dt, sb, run_time)
        print()

    print("All predictions complete.")


if __name__ == "__main__":
    main()