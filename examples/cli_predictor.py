from __future__ import annotations

import argparse
from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from leveledge import Predictor
from leveledge.predictor import ALLOWED_INTERVALS

try:
    TIMEZONE = ZoneInfo("US/Eastern")
except ZoneInfoNotFoundError:
    TIMEZONE = timezone(timedelta(hours=-5))
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ALLOWED_INTERVALS_STR = ", ".join(ALLOWED_INTERVALS)


def parse_cli_args() -> argparse.Namespace:
    parser: ArgumentParser = argparse.ArgumentParser(
        description="Prompt for inputs or accept CLI args before training the LevelEdge predictor."
    )
    parser.add_argument("--ticker", "-t", type=str, help="Ticker symbol to score (e.g. SPY)")
    parser.add_argument(
        "--datetime",
        "-d",
        type=str,
        help=f"Target datetime in {DATETIME_FORMAT} (US/Eastern), e.g. '2026-03-01 14:30:00'",
    )
    parser.add_argument(
        "--interval",
        "-i",
        choices=ALLOWED_INTERVALS,
        help="yfinance-compatible interval (one of the allowed intervals)",
    )
    parser.add_argument("--price", "-p", type=float, help="Price level to score against (numbers only)")
    return parser.parse_args()


def parse_datetime_arg(value: str) -> datetime:
    try:
        naive = datetime.strptime(value, DATETIME_FORMAT)
    except ValueError as exc:
        raise ArgumentTypeError(
            f"Datetime must be in the format {DATETIME_FORMAT}: {exc}"
        ) from exc
    return naive.replace(tzinfo=TIMEZONE)


def prompt_interval() -> str:
    prompt = f"Enter the interval you would like to analyze ({ALLOWED_INTERVALS_STR}): "
    while True:
        value = input(prompt).strip()
        if value in ALLOWED_INTERVALS:
            return value
        print(f"Invalid interval. Please choose one of: {ALLOWED_INTERVALS_STR}")


def prompt_datetime() -> tuple[datetime, str]:
    prompt = f"Enter the datetime you wish to predict for in EST ({DATETIME_FORMAT}): "
    while True:
        value = input(prompt).strip()
        try:
            naive = datetime.strptime(value, DATETIME_FORMAT)
        except ValueError:
            print("Datetime must be in the format YYYY-MM-DD HH:MM:SS. Try again.")
            continue
        return naive.replace(tzinfo=TIMEZONE), value


def prompt_price_level() -> float:
    prompt = "Enter the price level you wish to predict for (numbers only): $"
    while True:
        raw_value = input(prompt).strip().lstrip("$")
        try:
            return float(raw_value)
        except ValueError:
            print("Enter a valid numeric price (e.g. 42.50).")


def get_ticker(args: argparse.Namespace) -> str:
    if args.ticker:
        return args.ticker.strip()
    return input("Enter the ticker you wish to predict for (e.g. SPY, ETH-USD): ").strip()


def get_datetime(args: argparse.Namespace) -> tuple[datetime, str]:
    if args.datetime:
        return parse_datetime_arg(args.datetime), args.datetime
    return prompt_datetime()


def get_interval(args: argparse.Namespace) -> str:
    if args.interval:
        return args.interval
    return prompt_interval()


def get_price_level(args: argparse.Namespace) -> float:
    if args.price is not None:
        return args.price
    return prompt_price_level()


def main() -> None:
    args = parse_cli_args()

    ticker = get_ticker(args)
    tgt_datetime, tgt_datetime_str = get_datetime(args)
    intvl = get_interval(args)
    price_level = get_price_level(args)

    predictor = Predictor(ticker, tgt_datetime, intvl, price_level)

    predictor.train_xgb()

    prediction = predictor.predict_xgb()

    print("\n\n")
    print("=" * 30)
    print("Model Metrics")
    print("=" * 30)
    print()
    predictor.print_xgb_model_metrics()

    predictor.print_candles_ahead()

    print("\n\n")
    print("=" * 30)
    print("Prediction Results")
    print("=" * 30)
    print()
    print(
        f"{ticker.upper()} has a {prediction:.2%} chance of being above {price_level} at {tgt_datetime_str}"
    )


if __name__ == "__main__":
    main()
