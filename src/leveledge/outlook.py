"""
Market outlook: how far the S&P 500 might move over the next 1 day to 3 months, with
calibrated ranges, and what that means for the book.

Method (chosen by benchmarks/regime_research.py, out of sample 2000-2026): the
average of three distribution forecasts' quantiles, each built from data up to the
last completed close:
    vix      VIX-implied volatility x empirical standardized quantiles
    har      log-HAR on range-based daily variance
    gjr_fhs  GJR-GARCH(1,1)-t paths by filtered historical simulation
It forecasts the SIZE of moves well (beats climatology at 1 week and 1 month in both
periods, 80% ranges held 81-84% of outcomes). It does not forecast direction or the
timing of falls: no tested signal (HMM, jump model, trend, VIX, turbulence,
absorption ratio) predicted 5% monthly losses reliably out of sample.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import live
from . import portfolio as P
from . import regime as R
from .constants import US_EASTERN

H_FAN = (1, 2, 3, 5, 7, 10, 15, 21, 30, 42, 52, 63)
LV = np.round(np.arange(0.005, 1.0, 0.005), 3)
SHOW = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
LABELS = {5: "1 week", 21: "1 month", 63: "3 months"}


def _q(arr, lv):
    return np.quantile(arr, lv)


def _risk_label(pct: float) -> str:
    return "calm" if pct < 0.4 else "normal" if pct < 0.75 else "elevated" if pct < 0.9 else "stressed"


def market_outlook(cache_dir: str | None = None, now: datetime | None = None, book_weights: dict | None = None,
                   n_sims: int = 4000) -> dict:
    now = now or datetime.now(tz=US_EASTERN)
    px = live.completed_daily(P.load_prices(["^GSPC", "^VIX"], start="1990-01-01", cache_dir=cache_dir), now)
    g = px["^GSPC"].dropna()
    idx = g.index
    C = g["Close"].to_numpy(float)
    r = np.diff(np.log(C)) * 100                                   # r[k] = return into day k+1
    vix = px["^VIX"]["Close"].reindex(idx).ffill().to_numpy()
    rv = R.daily_variance(g) * 1e4
    har_x = R.har_design(rv)
    T = len(C) - 1                                                 # index of the last close
    X = np.c_[np.ones(len(idx)), har_x.to_numpy()]

    fit = R.gjr_fit(r)
    s2 = float(R.gjr_filter(fit, r)[-1])
    paths = R.gjr_simulate(fit, s2, max(H_FAN), n=n_sims, seed=7)

    logC = np.log(C)
    fan, dense = [], {}
    for h in H_FAN:
        o = np.arange(30, T - h + 1)
        fut = (logC[o + h] - logC[o]) * 100
        okv = np.isfinite(vix[o])
        q_vix = _q(fut[okv] / np.sqrt(h * vix[o][okv] ** 2 / P.TRADING_DAYS), LV) * np.sqrt(h * vix[T] ** 2 / P.TRADING_DAYS)
        target = np.log(pd.Series(rv.to_numpy()).rolling(h).mean().shift(-h).to_numpy())
        ok = np.isfinite(X).all(1) & np.isfinite(target)
        beta, *_ = np.linalg.lstsq(X[ok], target[ok], rcond=None)
        s2h = np.var(target[ok] - X[ok] @ beta)
        var_h = h * np.exp(X @ beta + 0.5 * s2h)
        oo = np.where(ok)[0]
        oo = oo[(oo >= 30) & (oo <= T - h)]
        u = ((logC[oo + h] - logC[oo]) * 100) / np.sqrt(var_h[oo])
        q_har = _q(u, LV) * np.sqrt(var_h[T])
        q_gjr = _q(paths[:, h - 1], LV)
        q = np.maximum.accumulate((q_vix + q_har + q_gjr) / 3)
        dense[h] = q
        fan.append({"step": h, **{f"q{int(round(l * 1000)):03d}": float(C[T] * np.exp(np.interp(l, LV, q) / 100)) for l in SHOW}})

    def p_below(h, move):  # P(simple return <= move)
        return float(np.interp(np.log1p(move) * 100, dense[h], LV, left=0.0, right=1.0))

    horizons = []
    for h, lab in LABELS.items():
        q = dense[h]
        horizons.append({
            "h": h, "label": lab,
            "ret": {f"q{int(round(l * 1000)):03d}": float(np.expm1(np.interp(l, LV, q) / 100)) for l in SHOW},
            "p_down5": p_below(h, -0.05), "p_down10": p_below(h, -0.10),
            "p_up5": 1 - p_below(h, 0.05), "p_up": 1 - p_below(h, 0.0),
        })

    # volatility state: forecast 1-month vol vs the history of realized 1-month vol
    sd21 = float((dense[21][np.searchsorted(LV, 0.84)] - dense[21][np.searchsorted(LV, 0.16)]) / 2)
    vol_fc = sd21 / 100 * np.sqrt(P.TRADING_DAYS / 21)
    realized = pd.Series(r).rolling(21).std().dropna().to_numpy() / 100 * np.sqrt(P.TRADING_DAYS)
    pct = float((realized < vol_fc).mean())

    sma10 = float(np.mean(C[-210:]))
    cash = px[P.TBILL]["Close"].reindex(idx).ffill() / 100 / P.TRADING_DAYS
    ex12 = C[T] / C[T - 252] - 1 - float(cash.iloc[-252:].sum())

    out = {
        "as_of_close": idx[-1].strftime("%Y-%m-%d"), "index": "S&P 500", "spot": float(C[T]),
        "method": "average of VIX-implied, HAR and GJR-GARCH (filtered historical simulation) distributions",
        "risk": {"vol_forecast_annual": vol_fc, "vol_percentile": pct, "label": _risk_label(pct), "vix": float(vix[T])},
        "state": {"trend_12m_excess": ex12, "above_sma10": bool(C[T] > sma10),
                  "drawdown_from_high": float(C[T] / C.max() - 1)},
        "horizons": horizons, "fan": fan,
        "history": [{"date": d.strftime("%Y-%m-%d"), "close": float(c)} for d, c in zip(idx[-126:], C[-126:])],
    }
    if book_weights:
        out["book"] = _book_view(book_weights, g, dense, now)
    return out


def _book_view(weights: dict, g: pd.DataFrame, dense: dict, now: datetime) -> dict | None:
    """Map the S&P outlook to the book: beta and residual vol of today's weights over the past year."""
    syms = [s for s, w in weights.items() if abs(w) > 1e-6]
    try:
        import yfinance as yf
        start = (now - timedelta(days=400)).strftime("%Y-%m-%d")
        closes = pd.concat({s: yf.Ticker(s).history(start=start, interval="1d", auto_adjust=True)["Close"] for s in syms}, axis=1)
    except Exception:  # noqa: BLE001
        return None
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    rets = closes.pct_change().dropna()
    w = pd.Series(weights).reindex(rets.columns).fillna(0.0)
    book = rets @ w
    mkt = g["Close"].pct_change().reindex(book.index).dropna()
    book = book.reindex(mkt.index)
    if len(book) < 120:
        return None
    beta = float(np.cov(book, mkt)[0, 1] / mkt.var())
    resid_sd = float((book - beta * mkt).std())
    rng = np.random.default_rng(11)
    res = {}
    for h, lab in LABELS.items():
        u = rng.uniform(0.005, 0.995, 20000)
        m = np.expm1(np.interp(u, LV, dense[h]) / 100)
        b = beta * m + rng.normal(0, resid_sd * np.sqrt(h), len(u))
        res[lab] = {f"q{int(round(l * 1000)):03d}": float(np.quantile(b, l)) for l in SHOW}
        res[lab]["p_down5"] = float((b <= -0.05).mean())
    return {"beta": beta, "resid_vol_annual": resid_sd * np.sqrt(P.TRADING_DAYS), "horizons": res}


# ---------------------------------------------------------------------------
# live forward record
# ---------------------------------------------------------------------------
BANDS = {"cover50": ("q250", "q750"), "cover80": ("q100", "q900"), "cover90": ("q050", "q950"), "cover95": ("q025", "q975")}


def target_dates(as_of: str, horizons=tuple(LABELS)) -> dict:
    """The NYSE session that closes h trading days after ``as_of``."""
    from .sessions import nyse_schedule

    d = pd.Timestamp(as_of)
    sched = nyse_schedule(d + timedelta(days=1), d + timedelta(days=int(max(horizons) * 1.6) + 10))
    days = [pd.Timestamp(x).normalize() for x in sched.index]
    return {h: days[h - 1].strftime("%Y-%m-%d") for h in horizons}


def forward_entry(o: dict, targets: dict | None, now: datetime) -> dict:
    tds = target_dates(o["as_of_close"])
    e = {"logged_at": now.isoformat(), "as_of_close": o["as_of_close"], "index": "^GSPC", "spot": o["spot"],
         "method": o["method"], "risk": o["risk"],
         "horizons": [{"h": h["h"], "target_date": tds[h["h"]], "ret": h["ret"], "p_up": h["p_up"], "p_down5": h["p_down5"]}
                      for h in o["horizons"]]}
    if targets:
        e["book"] = {"strategy": targets["strategy"], "as_of_close": targets["as_of_close"], "weights": targets["weights"]}
    return e


def score_forward(entries: list[dict], closes: pd.Series) -> dict:
    """Score every matured forecast in the log against realized index closes."""
    closes = closes.copy()
    closes.index = pd.DatetimeIndex(closes.index).normalize()
    rows, pending = [], []
    for e in entries:
        for h in e["horizons"]:
            td = pd.Timestamp(h["target_date"])
            if td not in closes.index:
                pending.append(h["target_date"])
                continue
            y = float(closes[td] / e["spot"] - 1)
            q = h["ret"]
            row = {"h": h["h"], "as_of": e["as_of_close"], "target_date": h["target_date"], "ret": y,
                   "brier_up": (float(y > 0) - h["p_up"]) ** 2}
            for k, (lo, hi) in BANDS.items():
                row[k] = bool(q[lo] < y < q[hi])
            rows.append(row)
    out = {"logged": len(entries), "first": entries[0]["as_of_close"] if entries else None,
           "next_result": min(pending) if pending else None, "horizons": {}}
    df = pd.DataFrame(rows)
    for h, lab in LABELS.items():
        d = df[df.h == h] if len(df) else df
        out["horizons"][str(h)] = {"label": lab, "n": int(len(d)),
                                   **({k: float(d[k].mean()) for k in BANDS} | {"brier_up": float(d["brier_up"].mean())} if len(d) else {})}
    out["recent"] = df.sort_values("target_date").tail(10).to_dict("records") if len(df) else []
    return out
