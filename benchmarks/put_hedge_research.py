#!/usr/bin/env python3
"""
Hedge the core with options instead of selling it: monthly SPY puts and collars.

Each month-end the book buys 1-month SPY options on the core's SPY exposure
(0.63 x equity), holds them to the next month-end and rolls. Nothing else changes.

Overlays (standard choices, nothing tuned here)
-----------------------------------------------
put5      long 5%-out-of-the-money puts (the CBOE PPUT index design).
put10     long 10%-out-of-the-money puts (cheaper crash insurance).
collar5   long 5%-OTM puts, paid for by selling 5%-OTM calls (caps the upside).

Pricing: no free history of option prices exists, so options are priced with
Black-Scholes at an implied vol taken from the VIX: 5%-OTM put = VIX, 10%-OTM put
= VIX + 4 points (put skew), 5%-OTM call = VIX - 4 points, plus 3% of premium paid
in spread on every option bought or sold. Options are marked to this model daily.
A sensitivity run shifts every implied vol by -2 / +2 points: if the conclusion
flips inside that range, the result is not trustworthy.

Protocol and adoption rule: as in hedge_research.py (dev 2008-2019, holdout 2020+;
adopt only if max drawdown is 5+ points shallower and Sharpe not lower in both
periods, and full-period CAGR within 1 point a year of the current book). The
12-month trend filter on the core (hedge_research.py) is shown for comparison.

Usage: python benchmarks/put_hedge_research.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from leveledge import live  # noqa: E402
from leveledge import portfolio as P  # noqa: E402
from hedge_research import CACHE, CRISES, DEV_END, HOLD_START, core_overlay, index_close, simulate  # noqa: E402

OUT = os.path.join(ROOT, "benchmarks", "results", "put_hedge")
HEDGE_FRAC = live.LEVERAGE * live.CORE_SHARE * live.CORE["SPY"]      # 0.63 of equity
SPREAD = 0.03
OVERLAYS = {"put5": [("put", 0.95, 0.00, +1)],
            "put10": [("put", 0.90, 0.04, +1)],
            "collar5": [("put", 0.95, 0.00, +1), ("call", 1.05, -0.04, -1)]}


def bs(kind, S, K, tau, sig, r):
    if tau <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    sig = max(sig, 0.05)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * tau) / (sig * np.sqrt(tau))
    d2 = d1 - sig * np.sqrt(tau)
    if kind == "put":
        return float(K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return float(S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2))


def with_options(base_ret: pd.Series, spy: pd.Series, vix: pd.Series, cash: pd.Series, legs, shift=0.0):
    idx = base_ret.index
    me = set(pd.Series(idx, index=idx).groupby(idx.to_period("M")).last())
    rolls = [i for i, d in enumerate(idx) if d in me]
    S, V, cr = spy.reindex(idx).to_numpy(), vix.reindex(idx).ffill().to_numpy() / 100, cash.reindex(idx).fillna(0).to_numpy()
    E, out, premium_paid = 1.0, [], 0.0
    book = []                      # open legs: (kind, K, sign, qty, expiry index, vol add)
    value_prev = 0.0
    nxt = {a: b for a, b in zip(rolls, rolls[1:])}
    for t in range(len(idx)):
        if t < rolls[0]:
            out.append(np.nan)
            continue
        val = sum(sg * q * bs(k, S[t], K, (ex - t) / P.TRADING_DAYS, V[t] + va + shift, cr[t] * P.TRADING_DAYS)
                  for k, K, sg, q, ex, va in book)
        r = base_ret.iloc[t] + (val - value_prev - cr[t] * value_prev) / E if t > rolls[0] else 0.0
        E *= 1 + r
        if t in nxt:                                   # expire the old legs at intrinsic, buy the next month's
            q = HEDGE_FRAC * E / S[t]
            book = [(k, m * S[t], sg, q, nxt[t], va) for k, m, va, sg in legs]
            prices = [bs(k, S[t], K, (ex - t) / P.TRADING_DAYS, V[t] + va + shift, cr[t] * P.TRADING_DAYS)
                      for k, K, sg, q_, ex, va in book]
            cost = sum(SPREAD * p * q for p in prices)
            net = sum(sg * p * q for (k, K, sg, q_, ex, va), p in zip(book, prices))
            premium_paid += net / E
            r_cost = cost / E
            E *= 1 - r_cost
            r = (1 + r) * (1 - r_cost) - 1
            value_prev = net
        else:
            value_prev = val
        out.append(r)
    s = pd.Series(out, index=idx).dropna()
    s.attrs["premium_per_year"] = premium_paid / (len(s) / P.TRADING_DAYS)
    return s


def summarize(r: pd.Series, cash: pd.Series) -> dict:
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    return {"full": P.metrics(r, cash), "dev": P.metrics(r.loc[:DEV_END], cash), "holdout": P.metrics(r.loc[HOLD_START:], cash),
            "losing_years": int((yearly.loc[:2025] < 0).sum()), "worst_year": round(float(yearly.loc[:2025].min()), 4),
            "cvar95_month": round(float(monthly[monthly <= monthly.quantile(0.05)].mean()), 4),
            "crises": {n: round(float((1 + r.loc[a:b]).prod() - 1), 4) for n, a, b in CRISES}}


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = live.config()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    book, trend = live.book_decisions(opens, closes, cash, cfg)
    vix = index_close("^VIX")
    base = simulate(closes, cash, book, live.BAND)["ret"]
    tsm = P.momentum_signal(closes[list(live.CORE)], cash, 252)
    trendf = simulate(closes, cash, core_overlay(trend, tsm.loc[book.index]), live.BAND)["ret"]

    runs = {"current": base, "core_tsmom12 (reference)": trendf}
    for k, legs in OVERLAYS.items():
        runs[k] = with_options(base, closes["SPY"], vix, cash, legs)
    start = max(r.index[0] for r in runs.values())
    runs = {k: r.loc[start:] for k, r in runs.items()}
    res = {"period": [str(start.date()), str(closes.index[-1].date())], "hedge_frac": HEDGE_FRAC, "variants": {}}
    for k, r in runs.items():
        res["variants"][k] = summarize(r, cash)
        if "premium_per_year" in r.attrs:
            res["variants"][k]["net_premium_per_year"] = round(r.attrs["premium_per_year"], 4)
    cur = res["variants"]["current"]
    for k in OVERLAYS:
        v = res["variants"][k]
        v["adopt"] = bool(all(v[p]["max_drawdown"] >= cur[p]["max_drawdown"] + 0.05 and v[p]["sharpe"] >= cur[p]["sharpe"]
                              for p in ("dev", "holdout")) and v["full"]["cagr"] >= cur["full"]["cagr"] - 0.01)
        v["sharpe_diff_full"] = P.sharpe_diff_ci(runs[k], runs["current"], cash)
        v["sensitivity"] = {}
        for sh in (-0.02, 0.02):
            rs = with_options(base, closes["SPY"], vix, cash, OVERLAYS[k], shift=sh).loc[start:]
            m = P.metrics(rs, cash)
            v["sensitivity"][f"{sh:+.0%} vol"] = {"cagr": m["cagr"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]}
    res["adopted"] = [k for k in OVERLAYS if res["variants"][k]["adopt"]]
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)

    print("period", res["period"])
    print(f"{'variant':26s} {'CAGR':>6s} {'Sharpe':>6s} {'MaxDD':>6s} | {'dev Sh':>6s} {'dev DD':>6s} | {'20+ Sh':>6s} {'20+ DD':>6s} | "
          f"{'worst yr':>8s} {'CVaR m':>7s} {'prem/yr':>7s} adopt")
    for k, v in res["variants"].items():
        print(f"{k:26s} {v['full']['cagr']:6.1%} {v['full']['sharpe']:6.2f} {v['full']['max_drawdown']:6.1%} | "
              f"{v['dev']['sharpe']:6.2f} {v['dev']['max_drawdown']:6.1%} | {v['holdout']['sharpe']:6.2f} {v['holdout']['max_drawdown']:6.1%} | "
              f"{v['worst_year']:8.1%} {v['cvar95_month']:7.1%} {v.get('net_premium_per_year', float('nan')):7.1%} {v.get('adopt', '-')}")
    print("\ncrisis windows")
    for k, v in res["variants"].items():
        print(f"{k:26s} " + " ".join(f"{n[:12]:>12s} {x:6.1%}" for n, x in v["crises"].items()))
    for k in OVERLAYS:
        print(k, "sensitivity", res["variants"][k]["sensitivity"], "sharpe diff", res["variants"][k]["sharpe_diff_full"])
    print("adopted:", res["adopted"])


if __name__ == "__main__":
    main()
