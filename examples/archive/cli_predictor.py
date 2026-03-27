import argparse
from datetime import datetime

from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
INTERVAL_PROMPT = ", ".join(ALLOWED_INTERVALS)


def parse_datetime_input(raw: str) -> datetime:
    naive = datetime.strptime(raw, DATETIME_FORMAT)
    return naive.replace(tzinfo=US_EASTERN)


def prompt_ticker() -> str:
    prompt = "Enter the ticker you wish to predict for (e.g. SPY, ETH-USD): "
    while True:
        value = input(prompt).strip()
        if value:
            return value.upper()
        print("Please provide a ticker symbol before continuing.")


def prompt_datetime() -> tuple[datetime, str]:
    prompt = f"Enter the datetime you wish to predict for in US/Eastern ({DATETIME_FORMAT}): "
    while True:
        value = input(prompt).strip()
        try:
            parsed = parse_datetime_input(value)
        except ValueError:
            print("Datetime must be in the format YYYY-MM-DD HH:MM:SS. Try again.")
            continue
        return parsed, value


def prompt_interval() -> str:
    prompt = f"Enter the interval you would like to analyze ({INTERVAL_PROMPT}): "
    while True:
        value = input(prompt).strip()
        if value in ALLOWED_INTERVALS:
            return value
        print(f"Interval must be one of: {INTERVAL_PROMPT}.")


def prompt_price_level() -> float:
    prompt = "Enter the price level you wish to predict for (numbers only): $"
    while True:
        raw = input(prompt).strip().lstrip("$")
        try:
            return float(raw)
        except ValueError:
            print("Enter a valid numeric price (e.g. 42.50).")


def parse_datetime_arg(value: str) -> datetime:
    try:
        return parse_datetime_input(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Datetime must use the format {DATETIME_FORMAT}."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively train LevelEdge on a ticker while optionally skipping prompts."
    )
    parser.add_argument(
        "-t",
        "--ticker",
        type=str,
        help="Ticker symbol to analyze (e.g. SPY, ETH-USD).",
    )
    parser.add_argument(
        "-d",
        "--datetime",
        type=parse_datetime_arg,
        metavar="YYYY-MM-DD HH:MM:SS",
        help="Datetime for the prediction (US/Eastern).",
    )
    parser.add_argument(
        "-i",
        "--interval",
        choices=ALLOWED_INTERVALS,
        help="Prediction interval (see allowed intervals).",
    )
    parser.add_argument(
        "-p",
        "--price",
        type=float,
        help="Target price level (numeric).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ticker:
        ticker = args.ticker.strip().upper()
    else:
        ticker = prompt_ticker()

    if args.datetime:
        tgt_datetime = args.datetime
        tgt_datetime_str = tgt_datetime.strftime(DATETIME_FORMAT)
    else:
        tgt_datetime, tgt_datetime_str = prompt_datetime()

    intvl = args.interval if args.interval else prompt_interval()

    price_level = args.price if args.price is not None else prompt_price_level()

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
