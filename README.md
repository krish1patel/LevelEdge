# What is LevelEdge

A simple stock prediction system that takes in a ticker, price level, interval, and a datetime to predict for, returning the probability of the price being above the inputted level.

## How the model works

1. Take in inputs
2. Fetch data from yahoo finance
3. Build a DataFrame of technical indicators
4. Calculate the number of candles ahead to predict for and create a target variable
5. Train on several train test splits, recording metrics for performance estimation
6. Retrain using all available data
7. Return prediction on latest data as a probability

# Install

1. Clone the repo
2. `cd LevelEdge`
3. (optional) Create a virtual environment and activate it. `python -m venv venv` `source venv/bin/activate`
4. `pip install -e .`

# Docs

- Import package with `from leveledge import Predictor`
- Initialize an object with `Predictor(ticker, datetime, interval, price)`
  - ticker - string
  - datetime - datetime object, must be timezone aware
  - interval - string, same as yfinance intervals; e.g. '15m', '1h'
  - price - float
- train_xgb()
  - Trains several models using xgboost
- predict_xgb()
  - Generates predictions and returns probability of price being above the inputted level at the inputted datetime

## Helper utilities

- `from leveledge import ensure_future_market_datetime`
  - Converts naive datetimes into US/Eastern-aware instants and validates they lie in the future for your market.
  - Pass the result to the `Predictor` constructor to avoid the "timezone unaware" error.

# Example usage

```python
from datetime import datetime

from leveledge import Predictor, ensure_future_market_datetime

prediction_time = ensure_future_market_datetime(
    datetime(2026, 3, 2, 15, 30),
    tz_name="US/Eastern",
)
predictor = Predictor(
    ticker_str="AAPL",
    target_datetime=prediction_time,
    interval="15m",
    price=175.0,
)

predictor.train_xgb()
probability = predictor.predict_xgb()
print(f"Probability price is above 175.0 at {prediction_time}: {probability:.2%}")
```

> Tip: `target_datetime` must be timezone aware and in the future, or the constructor will raise a `ValueError`. Match the timezone to the market (e.g., `US/Eastern` for NYSE) or use `ensure_future_market_datetime` so it is handled automatically.

# Limitations/Room for improvement

- Doesn't work well and often fails with prices well above or below the current price of a stock
- xgboost model could probably use some tuning
