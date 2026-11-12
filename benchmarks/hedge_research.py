#!/usr/bin/env python3
"""
Sell the core before it falls? Warning-signal overlays on the book's 60/40 core.

The trend sleeve (30%) already sells on a falling signal. The core (70% of the
book at 1.5x = 105% of equity in SPY and IEF) is always held, and it caused most of
the 2008 and 2022 losses. Each overlay below is a published or textbook warning
signal, used with its standard parameters (nothing tuned here), checked weekly
with the book's 2-point band.

Overlays
--------
current        the traded book, unchanged.
core_sma10     Faber (2007): each core ETF held only while its close is above its
               10-month (210-day) average, else that weight sits in T-bills.
core_tsmom12   Moskowitz, Ooi & Pedersen (2012) / Antonacci absolute momentum: each
               core ETF held only while its 12-month return beats T-bills.
vol_managed    Moreira & Muir (JF 2017): scale the whole book down when its recent
               volatility (21 days) is above its own trailing 3-year median vol
               (scale = median / recent, capped at 1; never levers up).
drawdown_cut   Risk-manager rule: drop to 1.0x leverage when the book is 10% below
               its peak, back to 1.5x once it is within 5% of the peak.
vix30          De-lever to 1.0x while the VIX closes above 30.
curve_invert   Yield-curve warning: SPY core weight to T-bills while the 10-year
               Treasury yield is below the 3-month T-bill yield.

Protocol: development 2008-2019, holdout 2020+ (these overlays are new to the
project, so 2020+ is unseen for them). Pre-declared adoption rule: an overlay
replaces the current book only if, in BOTH periods, its max drawdown is at least
5 points shallower and its Sharpe is not lower, and its full-period CAGR is within
1 point a year of the current book. A deflated Sharpe corrects for the 6 overlays
tried.

Usage: python benchmarks/hedge_research.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge import live  # noqa: E402
from leveledge import portfolio as P  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "hedge")
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"
CRISES = [("2008 crisis", "2007-10-09", "2009-03-09"), ("2009 rebound", "2009-03-10", "2009-12-31"),
          ("2018 Q4", "2018-09-20", "2018-12-24"), ("COVID crash", "2020-02-19", "2020-03-23"),
          ("2020 rebound", "2020-03-24", "2020-12-31"), ("2022 bear", "2022-01-03", "2022-10-12"),
          ("2023-24 rally", "2023-01-01", "2024-12-31")]


def index_close(sym: str) -> pd.Series:
    path = os.path.join(CACHE, f"{sym.replace('^', '')}_1d.parquet")
    if not os.path.exists(path):
        P.load_prices([], cache_dir=CACHE)  # makes sure the cache dir exists
        import yfinance as yf
        df = yf.Ticker(sym).history(start="2003-01-01", interval="1d", auto_adjust=True)
        df.index = pd.DatetimeIndex(df.index.tz_localize(None).normalize())
        df[["Open", "High", "Low", "Close"]].dropna().to_parquet(path)
    return pd.read_parquet(path)["Close"]


def simulate(closes, cash, targets: pd.DataFrame, band: float, scale_fn=None, cost_bps=5.0, spread=0.02):
    """The book's weekly banded rebalance (as in literature_research.banded_book), with an optional
    path-dependent scale on the targets: scale_fn(t, equity_so_far) -> multiplier."""
    idx = closes.index
    C = closes.to_numpy(float)
    on = np.vstack([np.ones((1, C.shape[1])), C[1:] / C[:-1]])
    cr = cash.reindex(idx).fillna(0).to_numpy()
    T = targets.reindex(columns=closes.columns).fillna(0.0)
    tpos = {idx.get_loc(d): T.loc[d].to_numpy() for d in T.index if d in idx}
    first = min(tpos)
    w = np.zeros(C.shape[1])
    eq, rets, turns, orders = [1.0], [], [], 0
    for t in range(first, len(idx)):
        r = w @ (on[t] - 1) + (1 - w.sum()) * cr[t] - max(0.0, w.sum() - 1) * spread / P.TRADING_DAYS
        w = w * on[t] / (1 + r)
        turnover = 0.0
        if t in tpos:
            tgt = tpos[t] * (scale_fn(t, eq) if scale_fn else 1.0)
            dev = np.abs(w - tgt)
            hit = (dev > band) | ((tgt == 0) & (w > 0))
            if hit.any():
                turnover = float(dev[hit].sum())
                orders += int(hit.sum())
                w = np.where(hit, tgt, w)
                r -= turnover * cost_bps / 1e4
        rets.append(r)
        turns.append(turnover)
        eq.append(eq[-1] * (1 + r))
    out = pd.DataFrame({"ret": rets, "turnover": turns}, index=idx[first:])
    out.attrs["orders"] = orders
    return out


def core_overlay(trend: pd.DataFrame, keep: pd.DataFrame) -> pd.DataFrame:
    """Book targets with each core ETF's core weight multiplied by keep[d, etf] (0 or 1)."""
    cols = trend.columns
    core = pd.Series(live.CORE, dtype=float).reindex(cols).fillna(0.0)
    k = keep.reindex(index=trend.index, columns=cols).fillna(1.0)
    return live.LEVERAGE * (live.CORE_SHARE * k * core.values + live.TREND_SHARE * trend)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = live.config()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    book, trend = live.book_decisions(opens, closes, cash, cfg)
    dates = book.index
    core_cols = list(live.CORE)

    sma = P.sma_signal(closes[core_cols], 210)
    tsm = P.momentum_signal(closes[core_cols], cash, 252)
    vix = index_close("^VIX").reindex(closes.index).ffill()
    tnx = index_close("^TNX").reindex(closes.index).ffill()
    irx = prices[P.TBILL]["Close"].reindex(closes.index).ffill()
    inverted = (tnx < irx)

    base = simulate(closes, cash, book, live.BAND)
    bret = base["ret"]
    rv = bret.rolling(21).std() * np.sqrt(P.TRADING_DAYS)
    med = rv.rolling(756, min_periods=252).median()
    vm_scale = (med / rv).clip(upper=1.0).fillna(1.0)

    lo = 1.0 / live.LEVERAGE
    state = {"cut": False}

    def dd_scale(t, eq):
        e = np.asarray(eq)
        dd = e[-1] / e.max() - 1
        if not state["cut"] and dd <= -0.10:
            state["cut"] = True
        elif state["cut"] and dd >= -0.05:
            state["cut"] = False
        return lo if state["cut"] else 1.0

    idx = closes.index
    runs = {
        "current": base,
        "core_sma10": simulate(closes, cash, core_overlay(trend, sma.loc[dates]), live.BAND),
        "core_tsmom12": simulate(closes, cash, core_overlay(trend, tsm.loc[dates]), live.BAND),
        "vol_managed": simulate(closes, cash, book.mul(vm_scale.reindex(dates).fillna(1.0), axis=0), live.BAND),
        "drawdown_cut": simulate(closes, cash, book, live.BAND, scale_fn=dd_scale),
        "vix30": simulate(closes, cash, book.mul(np.where(vix.reindex(dates) > 30, lo, 1.0), axis=0), live.BAND),
        "curve_invert": simulate(closes, cash, core_overlay(
            trend, pd.DataFrame({"SPY": (~inverted.reindex(dates)).astype(float), "IEF": 1.0}, index=dates)), live.BAND),
    }
    flags = {"core_sma10": (sma.loc[dates, "SPY"] == 0), "core_tsmom12": (tsm.loc[dates, "SPY"] == 0),
             "vix30": vix.reindex(dates) > 30, "curve_invert": inverted.reindex(dates)}

    start = max(r.index[0] for r in runs.values())
    sp = closes["SPY"].pct_change()
    res = {"period": [str(start.date()), str(idx[-1].date())], "variants": {}}
    for k, bt in runs.items():
        r = bt["ret"].loc[start:]
        eq = (1 + r).cumprod()
        yrs = len(r) / P.TRADING_DAYS
        yearly = (1 + r).groupby(r.index.year).prod() - 1
        v = {"full": P.metrics(r, cash), "dev": P.metrics(r.loc[:DEV_END], cash), "holdout": P.metrics(r.loc[HOLD_START:], cash),
             "orders_per_year": round(bt.attrs["orders"] / yrs, 1),
             "losing_years": int((yearly.loc[:2025] < 0).sum()),
             "worst_year": round(float(yearly.loc[:2025].min()), 4),
             "crises": {n: round(float((1 + r.loc[a:b]).prod() - 1), 4) for n, a, b in CRISES}}
        if k in flags:
            f = flags[k].loc[start:].astype(bool)
            v["share_of_weeks_out"] = round(float(f.mean()), 3)
            v["sell_signals_per_year"] = round(float((f & ~f.shift(1, fill_value=False)).sum() / yrs), 2)
        v["equity_end"] = round(float(eq.iloc[-1]), 4)
        res["variants"][k] = v

    cur = res["variants"]["current"]
    trials = []
    for k, v in res["variants"].items():
        trials.append(float((runs[k]["ret"].loc[start:] - cash.reindex(runs[k]["ret"].loc[start:].index)).mean()
                            / (runs[k]["ret"].loc[start:]).std()))
        if k == "current":
            continue
        v["adopt"] = bool(all(v[p]["max_drawdown"] >= cur[p]["max_drawdown"] + 0.05 and v[p]["sharpe"] >= cur[p]["sharpe"]
                              for p in ("dev", "holdout")) and v["full"]["cagr"] >= cur["full"]["cagr"] - 0.01)
        v["sharpe_diff_full"] = P.sharpe_diff_ci(runs[k]["ret"].loc[start:], base["ret"].loc[start:], cash)
    best = max((k for k in res["variants"] if k != "current"), key=lambda k: res["variants"][k]["dev"]["sharpe"])
    rb = runs[best]["ret"].loc[start:]
    res["deflated_best_dev"] = {"variant": best, **P.deflated_sharpe(rb - cash.reindex(rb.index), np.array(trials))}
    res["adopted"] = [k for k, v in res["variants"].items() if v.get("adopt")]
    eqw = {k: (1 + bt["ret"].loc[start:]).cumprod() for k, bt in runs.items()}
    wk = pd.DataFrame(eqw).resample("W-FRI").last()
    res["equity_weekly"] = [{"date": str(d.date()), **{k: round(float(x), 4) for k, x in row.items()}} for d, row in wk.iterrows()]
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)

    print(f"period {res['period']}")
    print(f"{'variant':13s} {'CAGR':>6s} {'Sharpe':>6s} {'MaxDD':>6s} | {'dev Sh':>6s} {'dev DD':>6s} | {'20+ Sh':>6s} {'20+ DD':>6s} | "
          f"{'worst yr':>8s} lose ord/yr out  adopt")
    for k, v in res["variants"].items():
        print(f"{k:13s} {v['full']['cagr']:6.1%} {v['full']['sharpe']:6.2f} {v['full']['max_drawdown']:6.1%} | "
              f"{v['dev']['sharpe']:6.2f} {v['dev']['max_drawdown']:6.1%} | {v['holdout']['sharpe']:6.2f} {v['holdout']['max_drawdown']:6.1%} | "
              f"{v['worst_year']:8.1%} {v['losing_years']:4d} {v['orders_per_year']:6.1f} {v.get('share_of_weeks_out', float('nan')):4.0%}  {v.get('adopt', '-')}")
    print("\ncrisis windows")
    print(f"{'variant':13s} " + " ".join(f"{n[:13]:>13s}" for n, _, _ in CRISES))
    for k, v in res["variants"].items():
        print(f"{k:13s} " + " ".join(f"{v['crises'][n]:13.1%}" for n, _, _ in CRISES))
    print("\nsharpe diff vs current (full):", {k: v.get("sharpe_diff_full") for k, v in res["variants"].items() if k != "current"})
    print("deflated:", res["deflated_best_dev"], "adopted:", res["adopted"])


if __name__ == "__main__":
    main()
