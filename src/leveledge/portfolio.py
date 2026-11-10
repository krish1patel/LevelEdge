"""
Volatility-targeted ETF trend portfolio: data, signals, sizing and backtest.

Strategy (rules fixed from the literature before any backtest was run)
-----------------------------------------------------------------------
* Universe: liquid ETFs spanning US and international equities, Treasuries,
  credit, gold, commodities and real estate (:data:`UNIVERSE`).
* Signal: time-series momentum (Moskowitz, Ooi & Pedersen 2012). Hold an ETF
  when its trailing 12-month total return beats cash; otherwise that sleeve
  sits in cash. Long or flat, never short.
* Sizing: each held ETF gets ``target_vol / N / sigma_i`` of capital, then the
  whole book is scaled so forecast portfolio volatility hits ``target_vol``.
  Gross exposure is capped at 100% (no leverage) and each ETF at 25%.
* Trading: signals at the close of the last trading day of each week, orders
  filled at the next session's open, 5 bp cost per unit of turnover.
  Uninvested capital earns the 13-week T-bill rate.

Everything here is pure functions of past data; :func:`backtest` never uses
a price after the decision time to make that decision.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leveledge.model import VolScaler

UNIVERSE: dict[str, str] = {
    "SPY": "US large caps",
    "QQQ": "Nasdaq-100",
    "IWM": "US small caps",
    "EFA": "Developed ex-US equities",
    "EEM": "Emerging-market equities",
    "IEF": "7-10y Treasuries",
    "TLT": "20+y Treasuries",
    "LQD": "Investment-grade credit",
    "GLD": "Gold",
    "DBC": "Commodities",
    "VNQ": "US real estate",
}
TBILL = "^IRX"
TRADING_DAYS = 252


@dataclass
class StrategyConfig:
    lookback: int = 252             # momentum window (trading days)
    target_vol: float = 0.10        # annualised portfolio volatility target
    max_gross: float = 1.0          # no leverage
    max_weight: float = 0.25
    cost_bps: float = 5.0           # per unit of turnover, each side
    vol_model: str = "ewma"         # "ewma" or "leveledge"
    vol_horizon: int = 5            # days the vol forecast covers (weekly rebalance)
    corr_window: int = 126
    rebalance: str = "weekly"       # "weekly" | "monthly"
    universe: list[str] = field(default_factory=lambda: list(UNIVERSE))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_prices(symbols: list[str], start: str = "2003-01-01", cache_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Daily total-return-adjusted OHLC (dividends reinvested) per symbol, plus the T-bill yield."""
    import yfinance as yf

    out = {}
    for s in list(symbols) + [TBILL]:
        path = os.path.join(cache_dir, f"{s.replace('^', '')}_1d.parquet") if cache_dir else None
        if path and os.path.exists(path):
            out[s] = pd.read_parquet(path)
            continue
        df = yf.Ticker(s).history(start=start, interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError(f"No daily data for {s}")
        df.index = pd.DatetimeIndex(df.index.tz_localize(None).normalize())
        df = df[["Open", "High", "Low", "Close"]].dropna()
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_parquet(path)
        out[s] = df
    return out


def panel(prices: dict[str, pd.DataFrame], symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Aligned open and close panels on common trading days, and daily cash return."""
    closes = pd.concat({s: prices[s]["Close"] for s in symbols}, axis=1)
    opens = pd.concat({s: prices[s]["Open"] for s in symbols}, axis=1)
    idx = closes.dropna().index
    closes, opens = closes.loc[idx], opens.loc[idx]
    irx = prices[TBILL]["Close"].reindex(idx).ffill() / 100.0
    cash = (irx / TRADING_DAYS).fillna(0.0)      # daily T-bill return (approximate)
    return opens, closes, cash


# ---------------------------------------------------------------------------
# volatility forecasts
# ---------------------------------------------------------------------------
def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA daily volatility, known at each close."""
    return np.sqrt((returns ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=30).mean())


def leveledge_vol(returns: pd.Series, horizon: int, refit_every: int = 5, min_obs: int = 500) -> pd.Series:
    """LevelEdge's mean-reverting variance model on daily returns, as a daily vol.

    Refit every ``refit_every`` days on data known at that close only: the
    likelihood fit uses rows whose forward ``horizon``-day move had already
    happened. Returns the forecast for the next ``horizon`` days, de-annualised
    to a per-day volatility so it is comparable with :func:`ewma_vol`.
    """
    r = returns.fillna(0.0).to_numpy()
    n = len(r)
    fwd = np.full(n, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(r)])
    fwd[: n - horizon] = cs[1 + horizon: n + 1] - cs[1: n - horizon + 1]
    out = np.full(n, np.nan)
    slots = np.zeros(n, dtype=int)
    for t in range(min_obs, n, refit_every):
        known = np.zeros(t + 1, bool)
        known[: t + 1 - horizon] = True             # forward move complete by close t
        vs = VolScaler().fit(r[: t + 1], slots[: t + 1], known, fwd[: t + 1], horizon)
        sig_h = vs.live([0] * horizon)
        end = min(t + refit_every, n)
        out[t] = sig_h / np.sqrt(horizon)
        # between refits keep the fitted parameters and roll the EWMA state forward
        # (same span-40 EWMA VolScaler uses), so each day's forecast uses that day's close
        alpha = 2 / 41
        for u in range(t + 1, end):
            vs.sig2 = np.append(vs.sig2, vs.sig2[-1] + alpha * (r[u] ** 2 - vs.sig2[-1]))
            out[u] = vs.live([0] * horizon) / np.sqrt(horizon)
    return pd.Series(out, index=returns.index)


def vol_forecasts(closes: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    rets = np.log(closes).diff()
    if cfg.vol_model == "ewma":
        return rets.apply(ewma_vol)
    if cfg.vol_model == "leveledge":
        return rets.apply(lambda s: leveledge_vol(s, cfg.vol_horizon))
    raise ValueError(f"unknown vol model {cfg.vol_model!r}")


# ---------------------------------------------------------------------------
# signals and sizing
# ---------------------------------------------------------------------------
def momentum_signal(closes: pd.DataFrame, cash: pd.Series, lookback: int) -> pd.DataFrame:
    """1 if trailing total return beats cash over ``lookback`` days, else 0."""
    excess = closes / closes.shift(lookback) - 1 - cash.rolling(lookback).sum().to_numpy()[:, None]
    sig = (excess > 0).astype(float)
    return sig.where(closes.shift(lookback).notna())


def target_weights(signal: pd.Series, daily_vol: pd.Series, corr: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """Vol-targeted long/flat weights for one decision date."""
    ann = daily_vol * np.sqrt(TRADING_DAYS)
    n = len(signal)
    ok = signal.notna() & ann.notna() & (ann > 0)
    w = pd.Series(0.0, index=signal.index)
    w[ok] = signal[ok] * cfg.target_vol / n / ann[ok]
    if w.abs().sum() == 0:
        return w
    cov = corr.loc[w.index, w.index].fillna(0).to_numpy() * np.outer(ann.fillna(0), ann.fillna(0))
    port_vol = float(np.sqrt(max(w.to_numpy() @ cov @ w.to_numpy(), 1e-12)))
    w *= cfg.target_vol / port_vol
    w = w.clip(upper=cfg.max_weight)
    if w.sum() > cfg.max_gross:
        w *= cfg.max_gross / w.sum()
    return w


def rebalance_dates(index: pd.DatetimeIndex, how: str) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    key = index.to_period("W-FRI") if how == "weekly" else index.to_period("M")
    return pd.DatetimeIndex(s.groupby(key).last().to_numpy())


def rolling_corr(closes: pd.DataFrame, dates, window: int) -> dict:
    """Correlation matrix of daily log returns over ``window`` days ending at each date."""
    rets = np.log(closes).diff()
    out = {}
    for d in dates:
        i = closes.index.get_loc(d)
        if i >= window + 1:
            out[d] = rets.iloc[i - window + 1: i + 1].corr()
    return out


def strategy_weights(opens, closes, cash, cfg: StrategyConfig, vol: pd.DataFrame | None = None,
                     signal: pd.DataFrame | None = None, corr: dict | None = None) -> pd.DataFrame:
    """Target weights decided at each rebalance close (rows = decision dates)."""
    vol = vol_forecasts(closes, cfg) if vol is None else vol
    signal = momentum_signal(closes, cash, cfg.lookback) if signal is None else signal
    dates = rebalance_dates(closes.index, cfg.rebalance)
    corr = rolling_corr(closes, dates, cfg.corr_window) if corr is None else corr
    rows = {}
    for d in dates:
        i = closes.index.get_loc(d)
        if i < max(cfg.lookback, cfg.corr_window) + 1 or d not in corr:
            continue
        rows[d] = target_weights(signal.loc[d], vol.loc[d], corr[d], cfg)
    return pd.DataFrame(rows).T.reindex(columns=closes.columns).fillna(0.0)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------
def backtest(opens: pd.DataFrame, closes: pd.DataFrame, cash: pd.Series, decisions: pd.DataFrame,
             cost_bps: float = 5.0) -> pd.DataFrame:
    """Daily simulation: weights decided at close t trade at the open of t+1.

    Holdings drift with prices between trades; uninvested capital earns cash.
    Returns the daily portfolio return, gross exposure and turnover, starting
    from the first decision.
    """
    idx = closes.index
    C, O = closes.to_numpy(float), opens.to_numpy(float)
    cr = cash.reindex(idx).fillna(0.0).to_numpy(float)
    prevC = np.vstack([C[:1], C[:-1]])
    on, gap, intra = C / prevC, O / prevC, C / O
    dec = decisions.reindex(columns=closes.columns).fillna(0.0)
    pos = {idx.get_loc(d): dec.loc[d].to_numpy(float) for d in dec.index if d in idx}
    first = min(pos) if pos else len(idx)
    w = np.zeros(C.shape[1])
    pending = None
    rets, gross, turns = [], [], []
    for t in range(first, len(idx)):
        turnover = 0.0
        if pending is not None:
            v1 = 1 + w @ (gap[t] - 1)                  # overnight on old weights (cash ~0 overnight)
            w_open = w * gap[t] / v1
            turnover = float(np.abs(pending - w_open).sum())
            v2 = 1 + pending @ (intra[t] - 1) + (1 - pending.sum()) * cr[t]
            r = v1 * v2 * (1 - turnover * cost_bps / 1e4) - 1
            w = pending * intra[t] / v2
            pending = None
        else:
            r = w @ (on[t] - 1) + (1 - w.sum()) * cr[t]
            w = w * on[t] / (1 + r)
        if t in pos:
            pending = pos[t]
        rets.append(r)
        gross.append(float(w.sum()))
        turns.append(turnover)
    return pd.DataFrame({"ret": rets, "gross": gross, "turnover": turns}, index=idx[first:])


def buy_and_hold(closes: pd.DataFrame, weights: dict[str, float], cash: pd.Series, rebalance: str | None = "monthly",
                 cost_bps: float = 5.0) -> pd.DataFrame:
    """Fixed-weight benchmark, rebalanced at close on the given schedule."""
    cols = list(weights)
    target = pd.Series(weights, dtype=float)
    rets = closes[cols].pct_change().fillna(0.0)
    reb = set(rebalance_dates(closes.index, rebalance)) if rebalance else set()
    w = target.copy()
    out = []
    for t in closes.index:
        g = 1 + rets.loc[t]
        r = float((w * (g - 1)).sum()) + (1 - w.sum()) * cash.loc[t]
        w = w * g / (1 + r)
        turnover = 0.0
        if t in reb:
            turnover = float((target - w).abs().sum())
            r -= turnover * cost_bps / 1e4
            w = target.copy()
        out.append((t, r, float(w.sum()), turnover))
    return pd.DataFrame(out, columns=["date", "ret", "gross", "turnover"]).set_index("date")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(ret: pd.Series, cash: pd.Series, turnover: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    if len(ret) < 20:
        return {}
    c = cash.reindex(ret.index).fillna(0.0)
    ex = ret - c
    eq = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    dd = eq / eq.cummax() - 1
    downside = ex[ex < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    monthly = (1 + ret).groupby(ret.index.to_period("M")).prod() - 1
    out = {
        "start": str(ret.index[0].date()), "end": str(ret.index[-1].date()), "years": round(years, 2),
        "cagr": round(cagr, 4), "vol": round(vol, 4),
        "sharpe": round(float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)), 3),
        "sortino": round(float(ex.mean() * TRADING_DAYS / downside), 3) if downside > 0 else None,
        "max_drawdown": round(float(dd.min()), 4),
        "calmar": round(cagr / abs(float(dd.min())), 3) if dd.min() < 0 else None,
        "worst_month": round(float(monthly.min()), 4),
        "pct_positive_months": round(float((monthly > 0).mean()), 3),
        "total_return": round(float(eq.iloc[-1] - 1), 4),
    }
    if turnover is not None:
        out["turnover_per_year"] = round(float(turnover.reindex(ret.index).fillna(0).sum() / years), 2)
    return out


def sharpe_diff_ci(a: pd.Series, b: pd.Series, cash: pd.Series, block: int = 21, n: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap 95% CI for Sharpe(a) - Sharpe(b) on common days."""
    idx = a.dropna().index.intersection(b.dropna().index)
    c = cash.reindex(idx).fillna(0.0).to_numpy()
    xa, xb = a.loc[idx].to_numpy() - c, b.loc[idx].to_numpy() - c
    T = len(idx)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(T / block))

    def sh(x):
        s = x.std(ddof=1)
        return x.mean() / s * np.sqrt(TRADING_DAYS) if s > 0 else 0.0

    diffs = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block, size=k)
        ii = (starts[:, None] + np.arange(block)).ravel()[:T]
        diffs[i] = sh(xa[ii]) - sh(xb[ii])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float(sh(xa) - sh(xb)), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss for variance forecasts (robust to noisy realised variance; lower is better)."""
    ok = np.isfinite(realized_var) & np.isfinite(forecast_var) & (forecast_var > 0) & (realized_var > 0)
    r, f = realized_var[ok], forecast_var[ok]
    return float(np.mean(r / f - np.log(r / f) - 1))


__all__ = ["UNIVERSE", "StrategyConfig", "load_prices", "panel", "ewma_vol", "leveledge_vol", "vol_forecasts",
           "momentum_signal", "target_weights", "strategy_weights", "backtest", "buy_and_hold", "metrics",
           "sharpe_diff_ci", "qlike", "rebalance_dates", "rolling_corr"]
