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

# Example scripts

- `python examples/cli_predictor.py` – Prompts for ticker, target time, interval, and level, then trains and predicts without leaving the terminal.
- `python examples/streamlit_predictor.py` – Launches a Streamlit dashboard so you can tweak the same inputs in a GUI.
- `python examples/batch_option_predictions.py` – Runs predictions for a batch of tickers/levels defined in the script and prints a summary table (handy for comparing setups).

# Limitations/Room for improvement

- Doesn't work well and often fails with prices well above or below the current price of a stock
- xgboost model could probably use some tuning

# CLI predictor example (timezone aware)

The `examples/cli_predictor.py` helper keeps the experience simple while protecting you from timezone bugs.

- Run it from the repo root with `python examples/cli_predictor.py`.
- Enter the ticker, target datetime (`YYYY-MM-DD HH:MM:SS`), interval (e.g. `5m`, `1h`, `1d`), and support level.
- The script automatically attaches `ZoneInfo('US/Eastern')` to whatever datetime you type, so you can paste naive local times and still work with the timezone-aware `Predictor` class.
- Because predictions use future candles, only enter datetimes strictly ahead of the current Eastern-time candle; past datetimes will raise a validation error inside `Predictor`.

After the run you will see the training metrics that `train_xgb()` produced plus the probability returned by `predict_xgb()`. Use this prompt-driven CLI when you want a quick manual check without wiring up the streamlit dashboard.
