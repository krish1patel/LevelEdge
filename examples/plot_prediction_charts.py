"""
Create candlestick charts with prediction markers for each ticker.

Reads the postprocessed predictions CSV produced by
`examples/postprocess_prediction_logs.py` and, for each ticker,
fetches today's intraday OHLC data using yfinance, then overlays
prediction points at the time they were made.

Usage (from project root):
    python examples/plot_prediction_charts.py \
        --input prediction_logs_postprocessed.csv \
        --interval 2m \
        --output-dir charts

This will create one PNG per ticker (e.g. charts/SPY_2026-03-11.png).
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from leveledge.constants import US_EASTERN


def today_eastern() -> datetime:
    """Return 'today' date in US/Eastern timezone."""
    now = datetime.now(tz=timezone.utc).astimezone(US_EASTERN)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def load_predictions(input_path: str) -> pd.DataFrame:
    """
    Load the postprocessed predictions CSV and normalize timestamps.

    Expected columns (at minimum):
      - ticker
      - logged_at_utc_start
      - prediction_avg
      - price_level
    """
    df = pd.read_csv(input_path)

    # Parse timestamps
    if "logged_at_utc_start" in df.columns:
        df["logged_at_utc_start"] = pd.to_datetime(df["logged_at_utc_start"], utc=True)
    else:
        raise ValueError("Column 'logged_at_utc_start' is required in the input CSV.")

    return df


def fetch_today_candles(ticker: str, interval: str) -> pd.DataFrame:
    """
    Fetch today's intraday OHLC data for the given ticker and interval using yfinance.
    """
    yf_ticker = yf.Ticker(ticker)

    # For intraday, "1d" period gives today's session (and possibly some extended).
    hist = yf_ticker.history(period="1d", interval=interval)
    if hist.empty:
        raise ValueError(f"No data returned for {ticker} at interval {interval}")

    return hist


def plot_ticker(
    ticker: str,
    df_preds: pd.DataFrame,
    interval: str,
    output_dir: str,
) -> None:
    """
    Create a candlestick chart for today's price action for `ticker` and overlay prediction markers.
    """
    candles = fetch_today_candles(ticker, interval)

    # Ensure index is timezone-aware and in US/Eastern for nicer labeling
    if candles.index.tz is None:
        candles.index = candles.index.tz_localize(timezone.utc).tz_convert(US_EASTERN)
    else:
        candles.index = candles.index.tz_convert(US_EASTERN)

    # Filter predictions to today's date (in US/Eastern)
    start_of_day = today_eastern()
    end_of_day = start_of_day + timedelta(days=1)

    preds_today = df_preds[
        (df_preds["logged_at_utc_start"] >= start_of_day.astimezone(timezone.utc))
        & (df_preds["logged_at_utc_start"] < end_of_day.astimezone(timezone.utc))
    ].copy()
    if preds_today.empty:
        return

    # Convert prediction timestamps to US/Eastern for alignment with candles index
    preds_today["logged_at_eastern"] = preds_today["logged_at_utc_start"].dt.tz_convert(
        US_EASTERN
    )

    # Start plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare candlestick-like visualization using rectangles/lines
    idx = mdates.date2num(candles.index.to_pydatetime())
    o = candles["Open"].values
    h = candles["High"].values
    l = candles["Low"].values
    c = candles["Close"].values

    width = 0.0008  # width in days; adjust for aesthetics

    for i, (x, open_, high_, low_, close_) in enumerate(zip(idx, o, h, l, c)):
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

    # Overlay prediction markers
    # Use close price at the nearest candle time for vertical placement.
    for _, row in preds_today.iterrows():
        ts = row["logged_at_eastern"]
        # Find nearest candle time
        nearest_idx = candles.index.get_indexer([ts], method="nearest")[0]
        ts_candle = candles.index[nearest_idx]
        price_at_ts = candles["Close"].iloc[nearest_idx]

        # Visual encoding: color by prediction strength
        pred_val = row.get("prediction_avg", None)
        if pred_val is None:
            marker_color = "blue"
        else:
            # Map [0,1] to red->green gradient
            marker_color = (1 - pred_val, pred_val, 0.0)

        ax.scatter(
            ts_candle,
            price_at_ts,
            color=marker_color,
            s=50,
            edgecolor="black",
            zorder=5,
        )

        # Optionally annotate with price level if present
        if "price_level" in row and not pd.isna(row["price_level"]):
            ax.text(
                ts_candle,
                price_at_ts,
                f" {row['price_level']:.2f}",
                fontsize=7,
                va="bottom",
                ha="left",
                color="black",
            )

    ax.set_title(f"{ticker.upper()} — {start_of_day.date()} ({interval} candles)")
    ax.set_xlabel("Time (US/Eastern)")
    ax.set_ylabel("Price")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=US_EASTERN))
    fig.autofmt_xdate()

    ax.grid(True, linestyle="--", alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"{ticker.upper()}_{start_of_day.date()}_{interval}.png",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create candlestick charts for today's price action per ticker, "
            "overlaying prediction points from a postprocessed predictions CSV."
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
        default="charts",
        help="Directory to write chart PNGs into (default: charts)",
    )

    args = parser.parse_args()

    df = load_predictions(args.input)

    # Group by ticker and plot one chart per ticker
    for ticker, df_tk in df.groupby("ticker"):
        try:
            plot_ticker(ticker, df_tk, args.interval, args.output_dir)
            print(f"Plotted {ticker}")
        except Exception as exc:
            print(f"Skipping {ticker} due to error: {exc}")


if __name__ == "__main__":
    main()

