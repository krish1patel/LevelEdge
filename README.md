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

## Example scripts

- `python examples/cli_predictor.py` – a terminal helper that prompts for ticker, target datetime, interval, and price level, enforces the allowed interval set from `leveledge.constants`, applies the required US/Eastern timezone, trains the XGBoost model, and prints the prediction plus the evaluation metrics and candles-ahead information.
- `streamlit run examples/streamlit_predictor.py` – launches a Streamlit dashboard where you can adjust the inputs with richer widgets and see the probability, model metrics, and candles-ahead output inside a modal dialog once the prediction finishes.
- `python examples/batch_option_predictions.py` – iterates a handful of sample tickers, fetches the next Friday option chains, picks call/put strikes below a $0.30 ask, runs the predictor for each strike, and writes the summarized results to `examples/option_predictions.csv` (also printed to the console as a markdown table).

# Limitations/Room for improvement

- Doesn't work well and often fails with prices well above or below the current price of a stock
- xgboost model could probably use some tuning
