# What is LevelEdge

A simple stock prediction system that takes in a ticker, price level, interval, and a datetime to predict for, returning the probability of the price being above the inputted level.

## How the model works

1. Take in inputs
2. Fetch data from yahoo finance
3. Build a DataFrame of technical indicators
4. Calculate the number of candles ahead to predict for and create a target variable
5. Train on several train test splits, recording model performance
6. Choose the best model based on metrics and predict for the latest candle
7. Return probability of latest prediction

# Install

1. Clone the repo
2. `cd LevelEdge`
3. (optional) Create a virtual environment and activate it. `python -m venv venv` `source venv/bin/activate`
4. `pip install -e .`

# Usage

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

# Limitations/Room for improvement

- Doesn't work well and often fails with prices well above or below the current price of a stock
- Models perform reasonably well, but better metrics could be used to narrow down during selection
- xgboost model could probably use some tuning
