"""
The live paper portfolio: 70% in a 60/40 core plus a 30% trend sleeve.

Design (chosen before the 80-variant search, which found nothing better):
* core   = 60% SPY / 40% IEF
* sleeve = the original trend strategy in :mod:`leveledge.portfolio`
  (12-month time-series momentum, long/flat, 10% vol target, EWMA vol,
  no leverage, 25% cap per ETF)
* book   = 0.7 * core + 0.3 * sleeve, rebalanced on the last trading day of
  each week after the close; orders fill at the next open.

Its goal is smaller crashes than 60/40 at a similar return per unit of risk,
not beating the market.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from leveledge import portfolio as P
from leveledge.constants import US_EASTERN
from leveledge.sessions import nyse_schedule

CORE = {"SPY": 0.6, "IEF": 0.4}
CORE_SHARE = 0.7
TREND_SHARE = 0.3
# Chosen from the reward/risk frontier (benchmarks/construction_research.py):
# 2007-2026 backtest 9.0%/yr, max drawdown -35%, with borrowing at T-bill + 2%.
LEVERAGE = 1.5
MAX_GROSS = 1.6            # hard ceiling for order planning (drift above 1.5x is trimmed back)
STRATEGY_ID = "core6040_trend30_x150_v2"


def config() -> P.StrategyConfig:
    return P.StrategyConfig()


def book_from_trend(trend: pd.Series) -> pd.Series:
    """Combine one decision's trend-sleeve weights with the fixed core, at the book's leverage."""
    core = pd.Series(CORE, dtype=float).reindex(trend.index).fillna(0.0)
    return LEVERAGE * (CORE_SHARE * core + TREND_SHARE * trend)


def book_decisions(opens, closes, cash, cfg: P.StrategyConfig | None = None, vol=None, signal=None, corr=None):
    """Weekly decision-date weights for the whole book, plus the sleeve's own weights."""
    cfg = cfg or config()
    trend = P.strategy_weights(opens, closes, cash, cfg, vol=vol, signal=signal, corr=corr)
    book = trend.apply(book_from_trend, axis=1)
    return book, trend


def completed_daily(prices: dict[str, pd.DataFrame], now: datetime | None = None) -> dict[str, pd.DataFrame]:
    """Drop today's daily bar unless the NYSE session has closed (yfinance serves a partial bar)."""
    now = now or datetime.now(tz=US_EASTERN)
    today = pd.Timestamp(now.date())
    sched = nyse_schedule(today - timedelta(days=7), today)
    closed_today = today in sched.index and now >= sched.loc[today, "market_close"] + pd.Timedelta(minutes=15)
    out = {}
    for s, df in prices.items():
        out[s] = df if closed_today else df[df.index < today]
    return out


def is_rebalance_day(day: pd.Timestamp) -> bool:
    """True on the last NYSE trading day of the week containing ``day``."""
    day = pd.Timestamp(day).normalize()
    week_end = day + pd.Timedelta(days=6 - day.weekday())
    sched = nyse_schedule(day - pd.Timedelta(days=7), week_end)
    days = [pd.Timestamp(d).normalize() for d in sched.index if pd.Timestamp(d).normalize() <= week_end]
    this_week = [d for d in days if d.isocalendar()[1] == day.isocalendar()[1] and d.year == day.year]
    return bool(this_week) and day == max(this_week)


def next_rebalance_date(now: datetime | None = None) -> pd.Timestamp:
    """The next weekly rebalance session (today if it is one and the job has not run yet)."""
    now = now or datetime.now(tz=US_EASTERN)
    today = pd.Timestamp(now.date())
    sched = nyse_schedule(today, today + pd.Timedelta(days=14))
    for d in sched.index:
        d = pd.Timestamp(d).normalize()
        if d == today and now.hour * 60 + now.minute >= 17 * 60 + 45:
            continue        # this week's job has already run after today's close
        if is_rebalance_day(d):
            return d
    return today


# Execution: trade an ETF only when it drifts more than BAND from target
# (benchmarks/literature_research.py: 419 -> 64 orders a year, 21 -> 14 bp/yr of costs,
# 0.45%/yr tracking error against weekly full rebalancing).
BAND = 0.02
LIVE_START = "2026-11-10"          # first paper fills

# Shadow book: the same design with the Keller & Keuning DAA-G12 sleeve in place of
# the trend sleeve. It won the pre-declared sleeve selection but is not proven
# (holdout Sharpe +0.06, CI -0.11 to +0.21), so it is tracked from public prices
# rather than traded. Switch only on months of forward evidence.
SHADOW_ID = "core6040_daa30_x150_shadow"
SHADOW_START = "2026-11-10"


def shadow_decisions(closes: pd.DataFrame) -> pd.DataFrame:
    from leveledge import strategies as S

    core = pd.Series(CORE, dtype=float).reindex(closes.columns).fillna(0.0)
    sleeve = S.daa_g12(closes).reindex(columns=closes.columns).fillna(0.0)
    return LEVERAGE * (CORE_SHARE * core.values + TREND_SHARE * sleeve)


def tracking_since(start: str, cache_dir: str | None = None, now: datetime | None = None) -> pd.DataFrame:
    """Growth of $1 from ``start`` for the live book's rules, the shadow book, 60/40 and SPY.

    Rebuilt from public prices and the published rules every time, so the shadow
    record cannot be adjusted after the fact.
    """
    from leveledge import strategies as S

    syms = sorted(set(config().universe) | set(S.DAA_SYMBOLS))
    prices = completed_daily(P.load_prices(syms, start="2005-01-01", cache_dir=cache_dir), now)
    opens, closes, cash = P.panel(prices, syms)
    tu = config().universe
    book, _ = book_decisions(opens[tu], closes[tu], cash)
    series = {
        "book_rules": P.backtest(opens, closes, cash, book.reindex(columns=closes.columns).fillna(0.0), 5.0)["ret"],
        "shadow_daa": P.backtest(opens, closes, cash, shadow_decisions(closes), 5.0)["ret"],
        "sixty_forty": P.buy_and_hold(closes, {"SPY": 0.6, "IEF": 0.4}, cash, "monthly")["ret"],
        "spy_buy_hold": P.buy_and_hold(closes, {"SPY": 1.0}, cash, None)["ret"],
    }
    df = pd.DataFrame(series).loc[pd.Timestamp(start):].fillna(0.0)
    if df.empty:
        return df
    df.iloc[0] = 0.0                                   # growth starts at the start date's close
    return (1 + df).cumprod()


def trend_flip_prices(closes: pd.DataFrame, cash: pd.Series, lookback: int) -> pd.Series:
    """Price at which each ETF's trend signal flips at the latest close.

    The signal is on while the trailing return beats cash:
    P / P_ago - 1 > sum of daily cash returns over the window, so the flip is
    P_ago * (1 + that sum). Above it the trend sleeve holds the ETF; below, it sells.
    The reference price rolls forward a day at a time, so the level moves slowly.
    """
    p_ago = closes.iloc[-1 - lookback]
    cash_sum = float(cash.iloc[-lookback:].sum())
    return p_ago * (1 + cash_sum)


def current_targets(cache_dir: str | None = None, now: datetime | None = None) -> dict:
    """Target weights from the latest completed close, with the reasoning behind them."""
    cfg = config()
    prices = completed_daily(P.load_prices(cfg.universe, start="2005-01-01", cache_dir=cache_dir), now)
    opens, closes, cash = P.panel(prices, cfg.universe)
    vol = np.log(closes).diff().apply(P.ewma_vol)
    signal = P.momentum_signal(closes, cash, cfg.lookback)
    d = closes.index[-1]
    corr = P.rolling_corr(closes, [d], cfg.corr_window)
    trend = P.target_weights(signal.loc[d], vol.loc[d], corr[d], cfg)
    book = book_from_trend(trend)
    lookback_ret = closes.iloc[-1] / closes.iloc[-1 - cfg.lookback] - 1
    flip = trend_flip_prices(closes, cash, cfg.lookback)
    return {
        "strategy": STRATEGY_ID,
        "as_of_close": d.strftime("%Y-%m-%d"),
        "rebalance_today": is_rebalance_day(d),
        "weights": {k: round(float(v), 5) for k, v in book.items()},
        "cash_weight": round(float(1 - book.sum()), 5),
        "gross": round(float(book.sum()), 4),
        "leverage": LEVERAGE,
        "assets": [
            {"symbol": s, "name": P.UNIVERSE[s], "weight": round(float(book[s]), 5),
             "core_weight": round(LEVERAGE * CORE_SHARE * CORE.get(s, 0.0), 5),
             "trend_weight": round(LEVERAGE * TREND_SHARE * float(trend[s]), 5),
             "trend_on": bool(signal.loc[d, s] == 1), "return_12m": round(float(lookback_ret[s]), 4),
             "vol_annual": round(float(vol.loc[d, s] * np.sqrt(P.TRADING_DAYS)), 4),
             "last_close": round(float(closes.loc[d, s]), 4),
             "flip_price": round(float(flip[s]), 4),
             "flip_move": round(float(flip[s] / closes.loc[d, s] - 1), 4)}
            for s in cfg.universe],
        "sleeve_gross": round(float(trend.sum()), 4),
        "next_rebalance": next_rebalance_date(now).strftime("%Y-%m-%d"),
    }


__all__ = ["CORE", "CORE_SHARE", "TREND_SHARE", "LEVERAGE", "MAX_GROSS", "STRATEGY_ID", "BAND", "LIVE_START",
           "SHADOW_ID", "SHADOW_START", "shadow_decisions", "tracking_since", "book_from_trend", "book_decisions",
           "completed_daily", "is_rebalance_day", "next_rebalance_date", "current_targets", "trend_flip_prices"]
