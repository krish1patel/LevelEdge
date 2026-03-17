"""
Backtest QQQ: for each trading day over the last 2 completed weeks, run predictions
for 4pm target from market open to close every 60 minutes. Uses all intervals except
1m and 1d. Results are logged to Supabase (backtest_logs).

Strike levels are computed dynamically per prediction time as percent offsets from
the current price at that moment (default: -1%, -0.5%, +0.5%, +1%).
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, time, timedelta

# Reduce noise from sklearn/xgboost during batch backtest
warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import yfinance as yf
from dotenv import load_dotenv
from tqdm import tqdm

from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN

# Intervals to use: all except 1m and 1d
BACKTEST_INTERVALS = [i for i in ALLOWED_INTERVALS if i not in ("1m", "1d")]

# Market hours: predictions are initiated at these end times each day
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
HOURLY_END_TIMES = [
    time(9, 30),
    time(10, 30),
    time(11, 30),
    time(12, 30),
    time(13, 30),
    time(14, 30),
    time(15, 30),
]

# Strike percent offsets from current price at prediction time
DEFAULT_STRIKE_PCT = (-1.0, -0.5, 0.5, 1.0)


def get_trading_days_last_two_weeks() -> list[datetime]:
    """Return trading-day dates (midnight Eastern) for the last 2 completed weeks (Mon-Fri)."""
    today = datetime.now(tz=US_EASTERN).date()
    monday_this_week = today - timedelta(days=today.weekday())
    monday_two_weeks_ago = monday_this_week - timedelta(weeks=2)
    all_days = [
        datetime(
            monday_two_weeks_ago.year,
            monday_two_weeks_ago.month,
            monday_two_weeks_ago.day,
            tzinfo=US_EASTERN,
        )
        + timedelta(days=i)
        for i in range(10)  # 2 weeks = 10 weekdays
    ]
    # Filter out weekends (Saturday=5, Sunday=6)
    return [d for d in all_days if d.weekday() < 5]


def run_backtest(
    ticker: str = "QQQ",
    intervals: list[str] | None = None,
    strike_pct: tuple[float, ...] = DEFAULT_STRIKE_PCT,
) -> None:
    """
    Run a 2-week backtest for the given ticker.

    For each trading day, for each hourly end time, a Predictor is initialised
    once to obtain the current price at that moment. Price levels are then derived
    dynamically as percent offsets from that price. Predictions are logged directly
    to Supabase backtest_logs via Predictor._log_prediction.

    Parameters
    ----------
    ticker      : Stock ticker to backtest (default: "QQQ")
    intervals   : List of intervals to use (default: all except 1m and 1d)
    strike_pct  : Tuple of percent offsets from current price for strike levels
                  (default: -1.0, -0.5, +0.5, +1.0)
    """
    load_dotenv()
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        raise SystemExit("SUPABASE_URL and SUPABASE_KEY must be set (e.g. in .env)")

    intervals = intervals or BACKTEST_INTERVALS
    trading_days = get_trading_days_last_two_weeks()

    if not trading_days:
        raise SystemExit("No trading days found for the last two weeks.")

    print(
        f"Backtest {ticker}: {len(trading_days)} days, {len(HOURLY_END_TIMES)} end-times/day, "
        f"{len(intervals)} intervals, {len(strike_pct)} dynamic price levels → Supabase backtest_logs"
    )
    print(
        f"Days: {[d.date() for d in trading_days]}; "
        f"intervals: {intervals}; strike_pct: {strike_pct}"
    )
    print()

    target_time = time(16, 0)
    total = 0
    errors = []

    # Pre-compute all valid (day, end_time) combos for the outer progress bar
    valid_slots = [
        (day_dt, end_time)
        for day_dt in trading_days
        for end_time in HOURLY_END_TIMES
        if datetime.combine(day_dt.date(), end_time, tzinfo=US_EASTERN)
        < datetime.combine(day_dt.date(), target_time, tzinfo=US_EASTERN)
    ]

    total_predictions = len(valid_slots) * len(intervals) * len(strike_pct)
    print(f"Total predictions to run: {total_predictions}\n")

    with tqdm(total=total_predictions, unit="pred", desc="Backtest", ncols=100) as pbar:
        for day_dt, end_time in valid_slots:
            target_dt = datetime.combine(day_dt.date(), target_time, tzinfo=US_EASTERN)
            end_dt = datetime.combine(day_dt.date(), end_time, tzinfo=US_EASTERN)

            # ------------------------------------------------------------------
            # Resolve current price once per end_time by initialising a throwaway
            # Predictor on the first available interval. This guarantees the price
            # is consistent with the data the model actually sees, and avoids an
            # extra yfinance API call.
            # ------------------------------------------------------------------
            current_price: float | None = None
            for probe_interval in intervals:
                try:
                    probe = Predictor(
                        ticker,
                        target_dt,
                        probe_interval,
                        999.0,  # placeholder price — not used for anything here
                        end_datetime=end_dt,
                    )
                    current_price = probe.current_price
                    break
                except Exception:
                    continue

            if current_price is None:
                errors.append(
                    (day_dt.date(), end_time, None, None, "Could not resolve current price")
                )
                pbar.update(len(intervals) * len(strike_pct))
                continue

            # Dynamic price levels relative to current price at this moment
            price_levels = [
                round(current_price * (1 + pct / 100.0), 2) for pct in strike_pct
            ]

            for interval in intervals:
                for price in price_levels:
                    pbar.set_postfix({
                        "date": str(day_dt.date()),
                        "end": str(end_time),
                        "interval": interval,
                        "price": f"${price:.2f}",
                        "errors": len(errors),
                    })
                    try:
                        p = Predictor(
                            ticker,
                            target_dt,
                            interval,
                            price,
                            end_datetime=end_dt,
                        )
                        p.train_xgb()
                        p.predict_xgb()
                        total += 1
                    except Exception as e:
                        errors.append(
                            (day_dt.date(), end_time, interval, price, str(e))
                        )
                    finally:
                        pbar.update(1)

    print(f"\nDone. Total backtest predictions logged to Supabase: {total}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for day, et, intv, pr, msg in errors[:20]:
            print(f"  {day} {et} {intv} ${pr}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtest QQQ 4pm target, hourly end times, dynamic strikes → Supabase backtest_logs"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="QQQ",
        help="Ticker to backtest (default: QQQ)",
    )
    parser.add_argument(
        "--pct",
        type=float,
        nargs="+",
        default=list(DEFAULT_STRIKE_PCT),
        help="Strike %% offsets from current price (default: -1 -0.5 0.5 1)",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        nargs="+",
        default=None,
        help="Intervals to use (default: all except 1m and 1d)",
    )
    args = parser.parse_args()

    run_backtest(
        ticker=args.ticker,
        intervals=args.intervals,
        strike_pct=tuple(args.pct),
    )