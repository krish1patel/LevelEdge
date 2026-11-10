"""
Stationary feature engineering.

Every feature is a ratio, a return, or a quantity in volatility units, so a
model trained when SPY was $450 still applies at $650. (v1 fed raw prices such
as Open/SMA_200/VWAP/OBV to the trees, which cannot extrapolate.)

Every feature at row ``i`` uses bars ``<= i`` only. ``tests/test_features.py``
checks this by recomputing on truncated data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from leveledge.sessions import DAILY_MIN, bar_ends, session_ordinals

EWMA_SPAN = 40  # bars; volatility estimator used for vol-unit scaling


def _safe_div(a, b):
    return a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / (b if b else np.nan)


def ewma_vol(log_ret: pd.Series, span: int = EWMA_SPAN) -> pd.Series:
    """EWMA standard deviation of per-bar log returns (zero-mean RiskMetrics form)."""
    return np.sqrt((log_ret ** 2).ewm(span=span, adjust=False, min_periods=span // 2).mean())


def diurnal_vol_factor(log_ret: pd.Series, slot: pd.Series) -> pd.Series:
    """Intraday volatility seasonality: |ret| in each time-of-day slot relative to
    the overall mean, estimated from *past* data only (expanding mean per slot).

    Intraday volatility is U-shaped (high at open/close, low at lunch). A
    plain EWMA over the last N bars mixes these regimes; dividing returns by
    this factor deseasonalises them.
    """
    a = log_ret.abs()
    slot_mean = a.groupby(slot).transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
    overall = a.shift(1).expanding(min_periods=20).mean()
    f = _safe_div(slot_mean, overall)
    return f.clip(0.3, 4.0).fillna(1.0)


def build_features(bars: pd.DataFrame, interval_min: int, is_crypto: bool) -> pd.DataFrame:
    """Return a feature frame aligned to ``bars`` (same index).

    Also adds helper columns used by the model (prefixed ``_``) that are not
    fed to the classifier: ``_logret``, ``_sigma``, ``_session``, ``_end``.
    """
    o, h, l, c, v = (bars[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    f = pd.DataFrame(index=bars.index)
    lr = np.log(c / c.shift(1))
    intraday = interval_min < DAILY_MIN

    # --- returns & momentum -------------------------------------------------
    for lag in (1, 2, 3, 5):
        f[f"ret_lag_{lag}"] = lr.shift(lag - 1)  # lag_1 = most recent bar's return
    for n in (5, 10, 20, 50):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    f["accel"] = lr - lr.shift(1)

    # --- volatility ---------------------------------------------------------
    sigma = ewma_vol(lr)
    rv = {n: lr.rolling(n).std() for n in (5, 10, 20, 50)}
    for n, s in rv.items():
        f[f"rvol_{n}"] = s
    f["rvol_ratio_5_50"] = _safe_div(rv[5], rv[50])
    f["rvol_ratio_20_50"] = _safe_div(rv[20], rv[50])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f["atr_pct"] = atr / c
    f["range_vs_atr"] = _safe_div(h - l, atr)
    f["ret_in_atr"] = _safe_div(c - c.shift(), atr)
    # Parkinson range-based vol: more efficient than close-to-close
    pk = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2)))
    f["parkinson_20"] = pk
    f["pk_vs_cc"] = _safe_div(pk, rv[20])

    # --- trend / mean reversion (all relative to price) ---------------------
    for n in (5, 10, 20, 50, 200):
        f[f"dist_sma_{n}"] = c / c.rolling(n).mean() - 1
    for n in (5, 10, 20):
        f[f"dist_ema_{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
    f["sma_5_20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1
    f["sma_20_50"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + _safe_div(gain, loss))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_pct"] = macd / c
    f["macd_hist_pct"] = (macd - sig) / c
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = _safe_div(c - (m20 - 2 * s20), 4 * s20)
    f["bb_width_pct"] = 4 * s20 / c
    f["zscore_20"] = _safe_div(c - m20, s20)
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    f["pos_in_range_20"] = _safe_div(c - lo20, hi20 - lo20)

    # --- candle shape -------------------------------------------------------
    rng = (h - l).replace(0, np.nan)
    f["body_ratio"] = (c - o) / rng            # signed
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["gap"] = np.log(o / c.shift(1))
    direction = np.sign(lr).fillna(0).to_numpy()
    streak = np.zeros(len(direction))
    for i in range(1, len(direction)):
        d = direction[i]
        if d == 0:
            streak[i] = streak[i - 1]
        elif np.sign(streak[i - 1]) in (0, d):
            streak[i] = streak[i - 1] + d
        else:
            streak[i] = d
    f["streak"] = streak

    # --- volume -------------------------------------------------------------
    has_vol = v.sum() > 0
    if has_vol:
        vma = v.rolling(20).mean()
        f["vol_ratio"] = _safe_div(v, vma)
        f["vol_ratio_5"] = _safe_div(v.rolling(5).mean(), vma)
        f["signed_vol_20"] = _safe_div((np.sign(lr) * v).rolling(20).sum(), v.rolling(20).sum())
        f["vol_x_ret"] = lr * f["vol_ratio"]

    # --- session context ----------------------------------------------------
    idx = bars.index
    if intraday:
        sess = pd.Series(session_ordinals(idx, is_crypto), index=idx)
        day_key = sess if not is_crypto else pd.Series(idx.normalize(), index=idx)
        day_open = o.groupby(day_key).transform("first")
        f["ret_since_open"] = np.log(c / day_open)
        tp = (h + l + c) / 3
        if has_vol:
            vwap = (tp * v).groupby(day_key).cumsum() / v.groupby(day_key).cumsum().replace(0, np.nan)
            f["vwap_dist"] = c / vwap - 1
        day_hi = h.groupby(day_key).cummax()
        day_lo = l.groupby(day_key).cummin()
        f["pos_in_day"] = _safe_div(c - day_lo, day_hi - day_lo)
        # previous session high/low distance
        agg = pd.DataFrame({"h": h, "l": l, "c": c, "k": day_key}).groupby("k").agg(
            h=("h", "max"), l=("l", "min"), c=("c", "last")
        ).shift(1)
        f["dist_prev_high"] = c / day_key.map(agg["h"]) - 1
        f["dist_prev_low"] = c / day_key.map(agg["l"]) - 1
        f["ret_vs_prev_close"] = np.log(c / day_key.map(agg["c"]))
        ends = bar_ends(idx, interval_min, is_crypto)
        minutes = ends.hour * 60 + ends.minute
        f["minute_of_day"] = minutes
        f["dow"] = ends.day_of_week
        slot = pd.Series(minutes, index=idx)
        f["diurnal_factor"] = diurnal_vol_factor(lr, slot)
    else:
        sess = pd.Series(np.arange(len(idx)), index=idx)
        ends = bar_ends(idx, interval_min, is_crypto)
        f["dow"] = ends.day_of_week
        f["dist_52w_high"] = c / h.rolling(252, min_periods=50).max() - 1

    f = f.replace([np.inf, -np.inf], np.nan)

    # helper columns (not model inputs)
    f["_logret"] = lr
    f["_sigma"] = sigma
    f["_session"] = sess.to_numpy()
    f["_end"] = ends
    return f


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if not col.startswith("_")]
