# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

LevelEdge is a Python library that predicts the probability of a stock/crypto price being above a given level at a future datetime. It fetches OHLCV data from yfinance, engineers technical features, and trains an XGBoost classifier using walk-forward cross-validation — then returns a probability from the final model trained on all available data.

## Setup

```bash
pip install -e .
```

Requires a `.env` file with `SUPABASE_URL` and `SUPABASE_KEY` for prediction logging (failures are silently swallowed so it's optional during development).

## Running things

**Streamlit demo UI:**
```bash
streamlit run examples/streamlit_batch_predictor.py
```

**Forward test** (logs predictions for QQQ/SPY/NVDA/TSLA to Supabase):
```bash
python scripts/forwardtest.py
```

**Backfill outcome prices** (fills `outcome_price` in Supabase for expired predictions):
```bash
python scripts/update_outcomes.py
```

## Architecture

All library code lives in `src/leveledge/`. The package exposes a single class:

```python
from leveledge import Predictor

p = Predictor(ticker, target_datetime, interval, price)
# target_datetime must be timezone-aware and in the future
# interval: "1m", "2m", "5m", "15m", "30m", "1h", "90m", "1d"
# price: the level to predict above/below

p.train_xgb()        # walk-forward CV + final model on all data
prob = p.predict_xgb()  # float 0–1

# Optional regression model (predicts future_close/current_close ratio):
p.train_regression()
ratio = p.predict_regression()
```

**`Predictor.__init__`** does heavy lifting immediately: fetches data, computes `candles_ahead`, and calls `prepare_features()` which runs all three feature-engineering methods in sequence.

**Feature pipeline** (all three methods mutate `self.data`):
1. `_calculate_technical_indicators()` — SMAs, EMAs, RSI, MACD, Bollinger Bands, ATR, volume ratios, prev-day/prev-hour highs/lows
2. `_calculate_candlestick_patterns()` — body/shadow ratios, Doji, Hammer, Shooting Star, Engulfing patterns
3. `_calculate_residual_features()` — realized volatility, VWAP, z-scores, lagged log returns, OBV, candle streak

**Target variable:** binary — whether `future_close / current_close > price / current_close` (i.e., the relative move exceeds the requested level, not just the absolute price). This is set by `_create_target_variable()`.

**`candles_ahead` calculation** differs for stocks vs crypto:
- Stocks: uses `pandas-market-calendars` (NYSE) to count only market hours, with pre/post market awareness
- Crypto (detected by `-` in ticker string, e.g. `BTC-USD`): simple 24/7 calculation

**Walk-forward splits** in `train_xgb`: uses 4 windows of `len(data)/4` train + 100 test rows, stepping by `len(data)/4`. Skips splits with only one class in train or test.

**Prediction logging** (`_log_prediction`): fires on every `predict_xgb()` call, writing to Supabase `logs` or `backtest_logs` table. Errors are caught and suppressed.

**Backtest mode**: pass `end_datetime` to `Predictor.__init__` to simulate a prediction as of a past time. yfinance data fetch is limited to a 59-day window ending at `end_datetime`.

## Key files

| File | Purpose |
|------|---------|
| `src/leveledge/predictor.py` | Core `Predictor` class — everything |
| `src/leveledge/constants.py` | `ALLOWED_INTERVALS`, `US_EASTERN` timezone |
| `scripts/forwardtest.py` | Batch forward-test runner (meant to be cron'd) |
| `scripts/update_outcomes.py` | Fills `outcome_price` in Supabase post-expiry |
| `examples/streamlit_batch_predictor.py` | Interactive Streamlit UI |
| `examples/testing/` | One-off analysis scripts (backtests, calibration, threshold sweep) |
| `examples/archive/` | Older scripts, not actively maintained |
