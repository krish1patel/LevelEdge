from leveledge import Predictor
from leveledge.predictor import ALLOWED_INTERVALS
from dateutil import parser
from zoneinfo import ZoneInfo

DEFAULT_TZ = ZoneInfo("US/Eastern")
DATETIME_PROMPT = (
    "Enter the datetime you wish to predict for "
    "(examples: '2026-02-18 15:00', '2026-02-18T15:00-05:00', "
    "'Feb 18 2026 3pm'). Timezone is optional and defaults to US/Eastern: "
)
INTERVAL_PROMPT = (
    "Enter the interval you would like to analyze "
    f"(options: {', '.join(ALLOWED_INTERVALS)}): "
)


def parse_target_datetime(raw: str):
    try:
        parsed = parser.parse(raw)
    except (parser.ParserError, ValueError) as exc:
        raise ValueError(
            "Could not parse the datetime. Try ISO 8601, 'YYYY-MM-DD HH:MM', "
            "or include a timezone offset like '-05:00'."
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=DEFAULT_TZ)
    else:
        parsed = parsed.astimezone(DEFAULT_TZ)

    return parsed


def prompt_for_datetime():
    while True:
        raw = input(DATETIME_PROMPT).strip()
        if not raw:
            print("Datetime input cannot be empty.")
            continue
        try:
            return parse_target_datetime(raw)
        except ValueError as exc:
            print(f"  >> {exc}")


def main():
    ticker = input('Enter the ticker you wish to predict for (e.g. SPY, ETH-USD): ').strip()
    tgt_datetime = prompt_for_datetime()
    interval = input(INTERVAL_PROMPT).strip()
    price_level = float(input('Enter the price level you wish to predict for: $'))

    predictor = Predictor(ticker, tgt_datetime, interval, price_level)

    predictor.train_xgb()
    prediction = predictor.predict_xgb()

    print('\n\n')
    print('=' * 30)
    print('Model Metrics')
    print('=' * 30)
    print()
    predictor.print_xgb_model_metrics()

    predictor.print_candles_ahead()

    print('\n\n')
    print('=' * 30)
    print('Prediction Results')
    print('=' * 30)
    print()
    print(
        f"{ticker.upper()} has a {prediction:.2%} chance of being above {price_level} at "
        f"{tgt_datetime.isoformat()} (US/Eastern)."
    )


if __name__ == "__main__":
    main()
