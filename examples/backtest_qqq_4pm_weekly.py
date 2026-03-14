"""
Backtest QQQ: for each trading day this week, run predictions for 4pm target
from market open to close every 60 minutes. Uses all intervals except 1m and 1d.
Results are logged to Supabase (backtest_logs). Uses 4 price levels near QQQ.
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

from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN

# Intervals to use: all except 1m and 1d
BACKTEST_INTERVALS = [i for i in ALLOWED_INTERVALS if i not in ("1m", "1d")]

# Market hours: 9:30 AM to 4:00 PM Eastern. End times every 60 min from open
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


def get_trading_days_this_week() -> list[datetime]:
    """Return list of trading-day dates (midnight Eastern) for this week (Mon–Fri)."""
    today = datetime.now(tz=US_EASTERN).date()
    # Monday of this week (weekday(): Mon=0, Fri=4)
    monday = today - timedelta(days=today.weekday())
    return [
        datetime(monday.year, monday.month, monday.day, tzinfo=US_EASTERN)
        + timedelta(days=i)
        for i in range(5)
    ]


def get_qqq_price_levels(count: int = 4) -> list[float]:
    """Fetch QQQ recent close and return 4 levels near it: base-2, base-1, base, base+1."""
    load_dotenv()
    ticker = yf.Ticker("QQQ")
    hist = ticker.history(period="5d", interval="1d")
    if hist.empty:
        return [478.0, 479.0, 480.0, 481.0]
    close = float(hist["Close"].iloc[-1])
    base = round(close)
    return [float(base - 2), float(base - 1), float(base), float(base + 1)]


def run_backtest(
    ticker: str = "QQQ",
    intervals: list[str] | None = None,
    price_levels: list[float] | None = None,
) -> None:
    load_dotenv()
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        raise SystemExit("SUPABASE_URL and SUPABASE_KEY must be set (e.g. in .env)")

    intervals = intervals or BACKTEST_INTERVALS
    if price_levels is None:
        price_levels = get_qqq_price_levels()

    trading_days = get_trading_days_this_week()
    if not trading_days:
        raise SystemExit("No NYSE trading days found for this week.")

    print(
        f"Backtest QQQ: {len(trading_days)} days, {len(HOURLY_END_TIMES)} end-times/day, "
        f"{len(intervals)} intervals, {len(price_levels)} price levels → Supabase backtest_logs"
    )
    print(f"Days: {[d.date() for d in trading_days]}; intervals: {intervals}; prices: {price_levels}")
    print()

    # 4pm target on each day
    target_time = time(16, 0)
    total = 0
    errors = []

    for day_dt in trading_days:
        target_dt = datetime.combine(
            day_dt.date(), target_time, tzinfo=US_EASTERN
        )
        for end_time in HOURLY_END_TIMES:
            end_dt = datetime.combine(
                day_dt.date(), end_time, tzinfo=US_EASTERN
            )
            if end_dt >= target_dt:
                continue
            for interval in intervals:
                for price in price_levels:
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
                        if total % 50 == 0:
                            print(f"  ... {total} predictions logged")
                    except Exception as e:
                        errors.append((day_dt.date(), end_time, interval, price, str(e)))

    print(f"Done. Total backtest predictions logged to Supabase: {total}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for day, et, intv, pr, msg in errors[:20]:
            print(f"  {day} {et} {intv} ${pr}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    run_backtest()
