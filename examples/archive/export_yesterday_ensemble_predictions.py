"""
Export yesterday's ensemble predictions for SPY and QQQ.

Reads the postprocessed predictions CSV produced by
`examples/postprocess_prediction_logs.py` and filters to rows where:
  - ticker is one of: SPY, QQQ
  - the prediction was logged yesterday (US/Eastern date, based on
    `logged_at_utc_start`)

For each matching row it writes a compact CSV with:
  - ticker
  - price_level
  - target_datetime
  - logged_at_utc_start
  - ensemble_prediction (0–1)
  - ensemble_prediction_pct (0–100)

Usage (from project root):
    python examples/export_yesterday_ensemble_predictions.py \
        --input prediction_logs_postprocessed.csv \
        --output yesterday_ensemble_predictions.csv

Optionally you can override the "yesterday" date:
    python examples/export_yesterday_ensemble_predictions.py \
        --date 2026-03-11
"""

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from leveledge.constants import US_EASTERN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export yesterday's ensemble predictions (SPY & QQQ) from the "
            "postprocessed predictions CSV."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default="prediction_logs_postprocessed.csv",
        help="Path to postprocessed predictions CSV (default: prediction_logs_postprocessed.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="yesterday_ensemble_predictions.csv",
        help="Path to output CSV (default: yesterday_ensemble_predictions.csv)",
    )
    parser.add_argument(
        "--date",
        "-d",
        default=None,
        help=(
            "Date to treat as 'yesterday' in YYYY-MM-DD (US/Eastern). "
            "If omitted, uses the calendar day before today in US/Eastern."
        ),
    )
    return parser.parse_args()


def get_yesterday_date_eastern(override_date: str | None = None) -> datetime:
    """
    Return the 'yesterday' date (midnight) in US/Eastern.

    If `override_date` is provided, it must be in YYYY-MM-DD format and will
    be interpreted as that date (not yesterday).
    """
    if override_date:
        dt = datetime.strptime(override_date, "%Y-%m-%d")
        # Localize to US/Eastern at midnight
        return dt.replace(tzinfo=US_EASTERN, hour=0, minute=0, second=0, microsecond=0)

    now_eastern = datetime.now(tz=timezone.utc).astimezone(US_EASTERN)
    yesterday = now_eastern.date() - timedelta(days=1)
    return datetime(
        year=yesterday.year,
        month=yesterday.month,
        day=yesterday.day,
        tzinfo=US_EASTERN,
    )


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    # Ensure we have the expected columns
    required_cols = {"ticker", "price_level", "target_datetime", "logged_at_utc_start", "prediction_avg"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Parse timestamps as UTC
    df["logged_at_utc_start"] = pd.to_datetime(df["logged_at_utc_start"], utc=True)

    # Filter to SPY and QQQ only
    df = df[df["ticker"].isin(["SPY", "QQQ"])].copy()
    if df.empty:
        print("No SPY/QQQ rows found in input CSV.")
        return

    # Determine yesterday date range in US/Eastern
    yesterday_start_eastern = get_yesterday_date_eastern(args.date)
    yesterday_end_eastern = yesterday_start_eastern + timedelta(days=1)

    yesterday_start_utc = yesterday_start_eastern.astimezone(timezone.utc)
    yesterday_end_utc = yesterday_end_eastern.astimezone(timezone.utc)

    # Filter rows where the prediction was logged yesterday (by start time)
    mask = (df["logged_at_utc_start"] >= yesterday_start_utc) & (df["logged_at_utc_start"] < yesterday_end_utc)
    df_yest = df[mask].copy()

    if df_yest.empty:
        print(f"No SPY/QQQ predictions found for date {yesterday_start_eastern.date()}.")
        return

    # Build output frame with compact, analysis-friendly columns
    out = pd.DataFrame(
        {
            "ticker": df_yest["ticker"],
            "price_level": df_yest["price_level"],
            "target_datetime": df_yest["target_datetime"],
            "logged_at_utc_start": df_yest["logged_at_utc_start"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ensemble_prediction": df_yest["prediction_avg"],
        }
    )
    out["ensemble_prediction_pct"] = out["ensemble_prediction"] * 100.0

    # Optional: sort for readability (by ticker then logged time then price level)
    out = out.sort_values(["ticker", "logged_at_utc_start", "price_level"]).reset_index(drop=True)

    out.to_csv(args.output, index=False)
    print(
        f"Wrote {len(out)} rows of SPY/QQQ ensemble predictions for "
        f"{yesterday_start_eastern.date()} to {args.output}"
    )


if __name__ == "__main__":
    main()

