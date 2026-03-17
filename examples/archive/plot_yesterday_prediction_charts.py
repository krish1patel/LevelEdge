"""
Visualize yesterday's SPY/QQQ predictions on candlestick charts.

Reads the postprocessed predictions CSV produced by
`examples/postprocess_prediction_logs.py` and, for SPY and QQQ:

- Filters to predictions whose `logged_at_utc_start` falls on yesterday's
  US/Eastern calendar day.
- Fetches intraday OHLC data for that day via yfinance.
- Plots a candlestick chart of yesterday's price action.
- Overlays one marker per prediction **at the time it was made** and at the
  corresponding `price_level`, annotated with ensemble prediction % and
  target time.

This avoids markers overwriting each other vertically because different
price levels are plotted at different y-values (the price levels themselves),
not all at the same close price.

Usage (from project root):
    python examples/plot_yesterday_prediction_charts.py \
        --input prediction_logs_postprocessed.csv \
        --interval 2m \
        --output-dir charts_yesterday
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from leveledge.constants import US_EASTERN


def get_yesterday_date_eastern(override_date: str | None = None) -> datetime:
    """
    Return the 'yesterday' date (midnight) in US/Eastern.

    If `override_date` is provided, it must be in YYYY-MM-DD format and will
    be interpreted as that date (not yesterday).
    """
    if override_date:
        dt = datetime.strptime(override_date, "%Y-%m-%d")
        return dt.replace(tzinfo=US_EASTERN, hour=0, minute=0, second=0, microsecond=0)

    now_eastern = datetime.now(tz=timezone.utc).astimezone(US_EASTERN)
    yesterday = now_eastern.date() - timedelta(days=1)
    return datetime(
        year=yesterday.year,
        month=yesterday.month,
        day=yesterday.day,
        tzinfo=US_EASTERN,
    )


def load_predictions(input_path: str) -> pd.DataFrame:
    """Load the postprocessed predictions CSV and normalize timestamps."""
    df = pd.read_csv(input_path)

    # Required columns
    required_cols = {
        "ticker",
        "price_level",
        "target_datetime",
        "logged_at_utc_start",
        "prediction_avg",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["logged_at_utc_start"] = pd.to_datetime(df["logged_at_utc_start"], utc=True)
    return df


def fetch_yesterday_candles(ticker: str, interval: str, day_start_eastern: datetime) -> pd.DataFrame:
    """
    Fetch yesterday's intraday OHLC data for the given ticker and interval using yfinance.
    """
    day_end_eastern = day_start_eastern + timedelta(days=1)
    start_utc = day_start_eastern.astimezone(timezone.utc)
    end_utc = day_end_eastern.astimezone(timezone.utc)

    yf_ticker = yf.Ticker(ticker)
    hist = yf_ticker.history(start=start_utc, end=end_utc, interval=interval)
    if hist.empty:
        raise ValueError(f"No data returned for {ticker} at interval {interval} for {day_start_eastern.date()}")

    # Ensure index is tz-aware and in US/Eastern
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize(timezone.utc).tz_convert(US_EASTERN)
    else:
        hist.index = hist.index.tz_convert(US_EASTERN)

    return hist


def plot_ticker_for_yesterday(
    ticker: str,
    df_preds: pd.DataFrame,
    interval: str,
    day_start_eastern: datetime,
    output_dir: str,
) -> None:
    """
    Create a candlestick chart for yesterday's price action for `ticker`
    and overlay prediction markers at (logged time, price_level).
    """
    candles = fetch_yesterday_candles(ticker, interval, day_start_eastern)

    # Filter predictions to yesterday (US/Eastern) based on logged_at_utc_start
    day_end_eastern = day_start_eastern + timedelta(days=1)
    day_start_utc = day_start_eastern.astimezone(timezone.utc)
    day_end_utc = day_end_eastern.astimezone(timezone.utc)

    preds = df_preds[
        (df_preds["logged_at_utc_start"] >= day_start_utc)
        & (df_preds["logged_at_utc_start"] < day_end_utc)
    ].copy()
    if preds.empty:
        return

    preds["logged_at_eastern"] = preds["logged_at_utc_start"].dt.tz_convert(US_EASTERN)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Candlestick-style plot using rectangles + wicks
    idx = mdates.date2num(candles.index.to_pydatetime())
    o = candles["Open"].values
    h = candles["High"].values
    l = candles["Low"].values
    c = candles["Close"].values

    width = 0.0008  # width in days; adjust as needed

    for x, open_, high_, low_, close_ in zip(idx, o, h, l, c):
        color = "green" if close_ >= open_ else "red"
        # Wick
        ax.vlines(x, low_, high_, color=color, linewidth=1)
        # Body
        ax.add_patch(
            plt.Rectangle(
                (x - width / 2, min(open_, close_)),
                width,
                abs(close_ - open_),
                facecolor=color,
                edgecolor=color,
                linewidth=1,
            )
        )

    # Overlay prediction markers at (time of prediction, price_level)
    y_min, y_max = candles["Low"].min(), candles["High"].max()
    y_range = y_max - y_min if y_max > y_min else 1.0

    for i, row in preds.iterrows():
        ts_eastern = row["logged_at_eastern"]
        price_level = row["price_level"]
        ensemble = float(row["prediction_avg"])
        ensemble_pct = ensemble * 100.0

        # Map ensemble 0–1 -> red->green color
        marker_color = (1 - ensemble, ensemble, 0.0)

        ax.scatter(
            ts_eastern,
            price_level,
            color=marker_color,
            s=60,
            edgecolor="black",
            zorder=5,
        )

        # Annotate slightly above the price level to avoid overlapping marker
        target_str = ""
        if isinstance(row.get("target_datetime"), str):
            try:
                # parse and show only time component in Eastern
                tgt_dt = pd.to_datetime(row["target_datetime"])
                if tgt_dt.tzinfo is None:
                    tgt_dt = tgt_dt.tz_localize(US_EASTERN)
                else:
                    tgt_dt = tgt_dt.tz_convert(US_EASTERN)
                target_str = tgt_dt.strftime("%H:%M")
            except Exception:
                target_str = row["target_datetime"]

        text_y = price_level + 0.01 * y_range
        label = f"{price_level:.2f} @ {ensemble_pct:.1f}%"
        if target_str:
            label += f" → {target_str}"

        ax.text(
            ts_eastern,
            text_y,
            label,
            fontsize=7,
            va="bottom",
            ha="left",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
        )

    ax.set_title(f"{ticker.upper()} — {day_start_eastern.date()} ({interval} candles, yesterday's predictions)")
    ax.set_xlabel("Time predictions made (US/Eastern)")
    ax.set_ylabel("Price / Price levels")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=US_EASTERN))
    fig.autofmt_xdate()
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add a simple colorbar-like explanation via text
    ax.text(
        0.01,
        0.98,
        "Marker color ~ ensemble probability (red=low, green=high)",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.5),
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"{ticker.upper()}_{day_start_eastern.date()}_{interval}_yesterday.png",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create candlestick charts for yesterday's price action for SPY & QQQ, "
            "overlaying ensemble predictions by price level and target time."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default="prediction_logs_postprocessed.csv",
        help="Path to postprocessed predictions CSV (default: prediction_logs_postprocessed.csv)",
    )
    parser.add_argument(
        "--interval",
        "-n",
        default="2m",
        help="YFinance interval to use for intraday candles (default: 2m)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="charts_yesterday",
        help="Directory to write chart PNGs into (default: charts_yesterday)",
    )
    parser.add_argument(
        "--date",
        "-d",
        default=None,
        help=(
            "Date to plot as 'yesterday' in YYYY-MM-DD (US/Eastern). "
            "If omitted, uses the calendar day before today in US/Eastern."
        ),
    )

    args = parser.parse_args()

    df = load_predictions(args.input)

    # Only SPY and QQQ
    df = df[df["ticker"].isin(["SPY", "QQQ"])].copy()
    if df.empty:
        print("No SPY/QQQ predictions found in input CSV.")
        return

    day_start_eastern = get_yesterday_date_eastern(args.date)

    for ticker, df_tk in df.groupby("ticker"):
        try:
            plot_ticker_for_yesterday(
                ticker=ticker,
                df_preds=df_tk,
                interval=args.interval,
                day_start_eastern=day_start_eastern,
                output_dir=args.output_dir,
            )
            print(f"Plotted yesterday's predictions for {ticker}")
        except Exception as exc:
            print(f"Skipping {ticker} due to error: {exc}")


if __name__ == "__main__":
    main()

