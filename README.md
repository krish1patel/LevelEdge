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

## Allowed intervals

Both the CLI (`examples/cli_predictor.py`) and the Streamlit app (`examples/streamlit_predictor.py`) source their interval choices from `leveledge.constants.ALLOWED_INTERVALS`, and the core `Predictor` class in `leveledge.predictor` enforces the same set before downloading data from Yahoo Finance. The Streamlit dashboard additionally defaults to `"15m"` (falling back to the first entry if that constant ever changes), and the CLI raises a clear error if you type an unsupported value.

The current set of allowed intervals is:

- `1m`
- `2m`
- `5m`
- `10m`
- `15m`
- `30m`
- `1h`
- `90m`
- `1d`

# Limitations/Room for improvement

- Doesn't work well and often fails with prices well above or below the current price of a stock
- xgboost model could probably use some tuning
