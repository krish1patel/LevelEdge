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

from dotenv import load_dotenv
from supabase import create_client

from leveledge import Predictor
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
    """Log a single prediction to forwardtest_logs matching the table schema exactly."""
    try:
        metrics = p.xgb_expected_model_metrics if hasattr(p, "xgb_expected_model_metrics") else (None, None, None)
        sb.table(TABLE_NAME).insert({
            "logged_at_utc":      run_time.isoformat(),
            "ticker":             p.ticker_str,
            "is_crypto":          p.isCrypto,
            "interval":           p.interval,
            "interval_minutes":   p.interval_min,
            "target_datetime":    p.target_datetime.isoformat(),
            "price_level":        p.price,
            "current_price":      float(p.current_price),
            "target_price_ratio": float(p.target_price_ratio),
            "candles_ahead":      p.candles_ahead,
            "prediction":         float(prediction),
            "model_type":         "xgboost_classifier",
            "model_auc":          float(metrics[0]) if metrics[0] is not None else None,
            "model_ps":           float(metrics[1]) if metrics[1] is not None else None,
            "model_pr":           float(metrics[2]) if metrics[2] is not None else None,
            "outcome_price":      None,
        }).execute()
    except Exception as e:
        print(f"    [warn] Supabase log failed ({p.ticker_str} {p.interval} ${p.price:.2f}): {e}")


# ---------------------------------------------------------------------------
# Core: run predictions for one ticker
# ---------------------------------------------------------------------------

def run_ticker(
    ticker: str,
    target_dt: datetime,
    end_dt: datetime,
    sb,
    run_time: datetime,
) -> None:
    print(f"  [{ticker}] Resolving current price...")

    # Resolve current price once via a throwaway Predictor
    current_price: float | None = None
    for probe_interval in INTERVALS:
        try:
            probe = Predictor(ticker, target_dt, probe_interval, 999.0, end_datetime=end_dt)
            current_price = probe.current_price
            break
        except Exception:
            continue

    if current_price is None:
        print(f"  [{ticker}] Could not resolve current price — skipping ticker")
        return

    price_levels = [round(current_price * (1 + pct / 100.0), 2) for pct in STRIKE_PCT]
    print(f"  [{ticker}] Price: ${current_price:.2f} | Levels: {price_levels}")

    total  = 0
    errors = 0

    for interval in INTERVALS:
        for price in price_levels:
            try:
                p = Predictor(ticker, target_dt, interval, price, end_datetime=end_dt)
                p.train_xgb()
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

    if end_dt >= target_dt:
        print("Target time (4pm ET) has already passed for today. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"Forward Test — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"Target: 4:00 PM ET | Tickers: {TICKERS}")
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