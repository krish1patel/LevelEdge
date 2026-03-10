from __future__ import annotations

import argparse
from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from leveledge import Predictor
from leveledge.predictor import ALLOWED_INTERVALS

DEFAULT_TIMEZONE = "US/Eastern"
TZ_FALLBACK_OFFSETS: dict[str, int] = {
    "US/Eastern": -5,
    "US/Central": -6,
    "US/Mountain": -7,
    "US/Pacific": -8,
    "UTC": 0,
}
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ALLOWED_INTERVALS_STR = ", ".join(ALLOWED_INTERVALS)


def resolve_timezone(name: str) -> timezone:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        if name in TZ_FALLBACK_OFFSETS:
            return timezone(timedelta(hours=TZ_FALLBACK_OFFSETS[name]))
        raise ArgumentTypeError(
            "Timezone '{0}' is not available. Install tzdata or choose one of: {1}".format(
                name, ", ".join(sorted(TZ_FALLBACK_OFFSETS))
            )
        ) from exc


def build_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(
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
    parser.add_argument(
        "--timezone",
        "-z",
        type=str,
        default=DEFAULT_TIMEZONE,
        help=f"Timezone name to interpret the target datetime (default: {DEFAULT_TIMEZONE})",
    )
    return parser


def parse_cli_args() -> tuple[argparse.Namespace, ArgumentParser]:
    parser = build_parser()
    return parser.parse_args(), parser


def parse_datetime_arg(value: str, tzinfo: timezone) -> datetime:
    try:
        naive = datetime.strptime(value, DATETIME_FORMAT)
    except ValueError as exc:
        raise ArgumentTypeError(f"Datetime must be in the format {DATETIME_FORMAT}: {exc}") from exc
    return naive.replace(tzinfo=tzinfo)


def prompt_interval() -> str:
    prompt = f"Enter the interval you would like to analyze ({ALLOWED_INTERVALS_STR}): "
    while True:
        value = input(prompt).strip()
        if value in ALLOWED_INTERVALS:
            return value
        print(f"Invalid interval. Please choose one of: {ALLOWED_INTERVALS_STR}")


def prompt_datetime(tzinfo: timezone, tz_name: str) -> tuple[datetime, str]:
    prompt = f"Enter the datetime you wish to predict for in {tz_name} ({DATETIME_FORMAT}): "
    while True:
        value = input(prompt).strip()
        try:
            naive = datetime.strptime(value, DATETIME_FORMAT)
        except ValueError:
            print("Datetime must be in the format YYYY-MM-DD HH:MM:SS. Try again.")
            continue
        return naive.replace(tzinfo=tzinfo), value


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


def get_datetime(args: argparse.Namespace, tzinfo: timezone, tz_name: str) -> tuple[datetime, str]:
    if args.datetime:
        return parse_datetime_arg(args.datetime, tzinfo), args.datetime
    return prompt_datetime(tzinfo, tz_name)


def get_interval(args: argparse.Namespace) -> str:
    if args.interval:
        return args.interval
    return prompt_interval()


def get_price_level(args: argparse.Namespace) -> float:
    if args.price is not None:
        return args.price
    return prompt_price_level()


def main() -> None:
    args, parser = parse_cli_args()

    try:
        target_tz = resolve_timezone(args.timezone)
    except ArgumentTypeError as exc:
        parser.error(str(exc))

    ticker = get_ticker(args)
    tgt_datetime, tgt_datetime_str = get_datetime(args, target_tz, args.timezone)
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
