---
title: LevelEdge
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


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

## Quick CLI helper

`examples/archive/cli_predictor.py` is a prompt-first entry point that trains a small LevelEdge ensemble for the ticker/interval/price/datetime combination you provide.

- All inputs can still be typed interactively, but you can skip the prompts by passing `--ticker`, `--datetime`, `--interval`, and `--price` if you already know the values.
- Partial flag sets are supported: `--ticker` plus `--interval` will prompt for the datetime and price level, so the script works in both automated and exploratory flows.
- The datetime flag expects `YYYY-MM-DD HH:MM:SS` in US/Eastern (matching the legacy prompt) and `--interval` must match one of the ALLOWED_INTERVALS defined by the package.

Example:
```
python examples/archive/cli_predictor.py --ticker AAPL --datetime "2026-03-27 15:30:00" --interval 15m --price 185.0
```
