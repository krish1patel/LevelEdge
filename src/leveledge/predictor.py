"""
Public entry points.

* :func:`forecast` — fit one model for (ticker, interval, target) and query any
  number of price levels from it. Use this for new code.
* :class:`Predictor` — the original one-level API, kept backward compatible
  for ``scripts/forwardtest.py``, the backtest scripts and the examples.
  It is now a thin wrapper around :func:`forecast`, so building eight
  Predictors for eight strikes reuses one fitted model instead of training
  eight.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN
from leveledge.data import clean_bars, fetch_history
from leveledge.model import ForecastConfig, LevelForecaster
from leveledge.sessions import compute_horizon, interval_minutes, is_crypto_ticker

MODEL_TYPE = "leveledge_v2_stacked"
_CACHE: "OrderedDict[tuple, LevelForecaster]" = OrderedDict()
_CACHE_SIZE = 32
_LOCKS: dict[tuple, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _cache_key(ticker, interval, bars: pd.DataFrame, horizon, evaluate, config):
    return (ticker, interval, len(bars), bars.index[-1], float(bars["Close"].iloc[-1]),
            horizon.bars, horizon.sessions_crossed, evaluate, repr(config))


def forecast(
    ticker: str,
    target_datetime: datetime,
    interval: str = "15m",
    as_of: datetime | None = None,
    data: pd.DataFrame | None = None,
    evaluate: bool = True,
    config: ForecastConfig | None = None,
) -> LevelForecaster:
    """Fit (or fetch from cache) a forecaster for ``ticker`` at ``target_datetime``.

    Parameters
    ----------
    as_of : simulate a forecast made at this past time (backtest). Default: now.
    data  : optional pre-fetched OHLCV frame (any source); it is cleaned and
            cut to bars completed by ``as_of``.
    """
    ticker = ticker.upper().strip()
    if interval not in ALLOWED_INTERVALS:
        raise ValueError(f"Invalid interval {interval!r}; choose one of {ALLOWED_INTERVALS}.")
    if target_datetime.tzinfo is None:
        raise ValueError("target_datetime must be timezone-aware.")
    now = datetime.now(tz=US_EASTERN)
    as_of_eff = as_of or now
    if target_datetime <= as_of_eff:
        raise ValueError("Target datetime must be after the forecast time.")

    imin = interval_minutes(interval)
    crypto = is_crypto_ticker(ticker)
    bars = (clean_bars(data, interval, ticker, as_of_eff) if data is not None
            else fetch_history(ticker, interval, as_of))
    horizon = compute_horizon(bars.index[-1], target_datetime, imin, crypto)

    key = _cache_key(ticker, interval, bars, horizon, evaluate, config)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:  # concurrent requests for the same forecast share one fit
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        model = LevelForecaster(bars, imin, crypto, horizon, config).fit(evaluate=evaluate)
        model.ticker = ticker
        model.interval = interval
        _CACHE[key] = model
        while len(_CACHE) > _CACHE_SIZE:
            old, _ = _CACHE.popitem(last=False)
            _LOCKS.pop(old, None)
        return model


class Predictor:
    """P(price > ``price`` at ``target_datetime``) — backward-compatible API.

    >>> p = Predictor("SPY", target, "15m", 600.0)
    >>> p.train_xgb()
    >>> p.predict_xgb()          # probability in [0, 1]
    """

    def __init__(self, ticker_str: str, target_datetime: datetime, interval: str, price: float,
                 end_datetime: datetime | None = None, data: pd.DataFrame | None = None,
                 config: ForecastConfig | None = None) -> None:
        if interval not in ALLOWED_INTERVALS:
            raise ValueError("Invalid interval input.")
        self.price = float(price)
        self.ticker_str = ticker_str.upper().strip()
        self.target_datetime = target_datetime
        self.interval = interval
        self.interval_min = interval_minutes(interval)
        self.isCrypto = is_crypto_ticker(self.ticker_str)
        self.hasPrePost = False  # v2 uses regular-session bars only
        self.is_backtest = end_datetime is not None
        self.end_datetime = end_datetime if end_datetime is not None else datetime.now(tz=US_EASTERN)
        self.config = config
        if self.target_datetime < self.end_datetime:
            raise ValueError("Target Datetime Must be after end datetime.")

        imin, crypto = self.interval_min, self.isCrypto
        self._bars = (clean_bars(data, interval, self.ticker_str, self.end_datetime) if data is not None
                      else fetch_history(self.ticker_str, interval, end_datetime))
        self.horizon = compute_horizon(self._bars.index[-1], target_datetime, imin, crypto)
        self.candles_ahead = self.horizon.bars
        self.current_price = float(self._bars["Close"].iloc[-1])
        self.target_price_ratio = self.price / self.current_price
        self._data_arg = data
        self.model: LevelForecaster | None = None
        self._constant_prediction = None

        # Legacy attributes used by analysis scripts
        c = self._bars["Close"]
        fut = c.shift(-self.candles_ahead)
        frame = self._bars.copy()
        frame["Future_close"] = fut
        frame["Future_close_ratio"] = fut / c
        frame["Target"] = (frame["Future_close_ratio"] > self.target_price_ratio).astype(int)
        frame["Datetime"] = frame.index
        self.data_withna = frame
        self.data = frame.dropna().reset_index(drop=True)

    # ---- training / prediction ------------------------------------------------
    def train_xgb(self, evaluate: bool = True) -> None:
        self.model = forecast(self.ticker_str, self.target_datetime, self.interval,
                              as_of=self.end_datetime if self.is_backtest else None,
                              data=self._data_arg if self._data_arg is not None else self._bars,
                              evaluate=evaluate, config=self.config)
        self.features = self.model.frame
        self.available_features = self.model.feature_names
        lb = self.model.level_backtest(self.price) if evaluate else None
        self.level_metrics = lb
        if lb:
            self.xgb_expected_model_metrics = (lb["blend"]["auc"], lb["precision_at_60"], lb["avg_precision"])
        else:
            self.xgb_expected_model_metrics = (None, None, None)

    def predict_xgb(self) -> float:
        if self.model is None:
            raise RuntimeError("Call train_xgb() first.")
        prediction = float(self.model.prob_above([self.price])[0])
        self._log_prediction(prediction, self.is_backtest)
        return prediction

    def get_xgb_model_metrics(self):
        return getattr(self, "xgb_expected_model_metrics", (None, None, None))

    def print_xgb_model_metrics(self) -> None:
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
        auc, ps, pr = self.get_xgb_model_metrics()
        print(f"Expected Model Metrics: AUC: {fmt(auc)}, PS: {fmt(ps)}, PR: {fmt(pr)}")

    def print_candles_ahead(self) -> None:
        print(f"Predicting for {self.candles_ahead} candles ahead "
              f"(resolves {self.horizon.effective_target:%Y-%m-%d %H:%M} ET)")
        print(f"Most recent candle datetime: {self.data_withna['Datetime'].iloc[-1]}")

    # Regression API: the median of the forecast distribution
    def train_regression(self) -> None:
        if self.model is None:
            self.train_xgb()
        median = self.model.quantiles((0.5,))[0.5]
        self._median_ratio = median / self.current_price
        y = self.data["Future_close_ratio"].to_numpy()
        self.regression_expected_model_metrics = (
            float(np.mean(np.abs(y - 1))), float(np.sqrt(np.mean((y - 1) ** 2))), 0.0)

    def predict_regression(self) -> float:
        if not hasattr(self, "_median_ratio"):
            self.train_regression()
        return float(self._median_ratio)

    def get_regression_model_metrics(self):
        return self.regression_expected_model_metrics

    def print_regression_model_metrics(self) -> None:
        mae, rmse, r2 = self.regression_expected_model_metrics
        print(f"Regression (median forecast) naive-move scale: MAE {mae:.6f}, RMSE {rmse:.6f}")

    # ---- logging ----------------------------------------------------------------
    def log_record(self, prediction: float) -> dict:
        auc, ps, pr = self.get_xgb_model_metrics()
        return {
            "logged_at_utc": (self.end_datetime if self.is_backtest
                              else datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
            "ticker": self.ticker_str,
            "is_crypto": self.isCrypto,
            "interval": self.interval,
            "interval_minutes": self.interval_min,
            "target_datetime": self.target_datetime.isoformat(),
            "price_level": self.price,
            "current_price": self.current_price,
            "target_price_ratio": self.target_price_ratio,
            "candles_ahead": self.candles_ahead,
            "prediction": float(prediction),
            "model_type": MODEL_TYPE,
            "model_auc": auc,
            "model_ps": ps,
            "model_pr": pr,
        }

    def _log_prediction(self, prediction: float, is_backtest: bool = False) -> None:
        if os.environ.get("LEVELEDGE_DISABLE_LOGGING"):
            return
        try:
            from dotenv import load_dotenv
            from supabase import create_client
            load_dotenv()
            url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
            if not (url and key):
                return
            table = "backtest_logs" if is_backtest else "logs"
            create_client(url, key).table(table).insert(self.log_record(prediction)).execute()
        except Exception:
            pass  # logging must never break prediction
