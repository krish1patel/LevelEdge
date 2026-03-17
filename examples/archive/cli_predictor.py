from datetime import datetime

from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
INTERVAL_PROMPT = ", ".join(ALLOWED_INTERVALS)

ticker = input('Enter the ticker you wish to predict for (e.g. SPY, ETH-USD): ').strip()
tgt_datetime_str = input(
    f'Enter the datetime you wish to predict for in US/Eastern ({DATETIME_FORMAT}): '
).strip()
intvl = input(f'Enter the interval you would like to analyze ({INTERVAL_PROMPT}): ').strip()
price_level = float(input('Enter the price level you wish to predict for: $').strip())

tgt_datetime = datetime.strptime(tgt_datetime_str, DATETIME_FORMAT).replace(tzinfo=US_EASTERN)

if intvl not in ALLOWED_INTERVALS:
    raise ValueError(
        f"Interval must be one of: {INTERVAL_PROMPT}. Received '{intvl}'."
    )

predictor = Predictor(ticker, tgt_datetime, intvl, price_level)

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
    f"{ticker.upper()} has a {prediction:.2%} chance of being above {price_level} at {tgt_datetime_str}"
)
