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

## Command-line helper

`examples/cli_predictor.py` is a quick helper that walks through a terminal session and reports metrics before printing the probability of the target price being reached. It
- prompts for a ticker (e.g. SPY, ETH-USD)
- asks for a timestamp in `YYYY-MM-DD HH:MM:SS` format and assigns it the `US/Eastern` timezone via `leveledge.constants.US_EASTERN`
- requires one of the `leveledge.constants.ALLOWED_INTERVALS` so invalid intervals are caught before training
- accepts a numeric price level, trains the predictor, and reuses `predictor.print_xgb_model_metrics()` + `predictor.print_candles_ahead()` to summarize the fitted model before showing the probability

Just run `python examples/cli_predictor.py` and follow the prompts. When you need to automate the flow for testing, pipe the answers directly:

```bash
printf 'AAPL\n2026-03-25 15:30:00\n15m\n184.0\n' | python examples/cli_predictor.py
```

The script reuses the same `leveledge` configuration as the library so once it finishes you already have the trained model metrics and probability output ready for follow-up.


# Limitations/Room for improvement

- Doesn't work well and often fails with prices well above or below the current price of a stock
- xgboost model could probably use some tuning
