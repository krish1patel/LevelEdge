#!/usr/bin/env python3
"""
Reward through construction: risk-balanced, volatility-targeted books, with
leverage paid for, and a reward/risk frontier for the selected design.

Protocol (fixed before running)
-------------------------------
Designs (weekly decisions, next-open fills, 5 bp costs, borrowing at T-bill + 2%,
gross <= 150%, any single ETF <= 60%):
  vt6040        60% SPY / 40% IEF scaled to a 10% vol forecast
  rp            equal risk per asset class (equities, bonds, real assets), 10% vol
  rp_trend      0.7 * rp + 0.3 * trend sleeve
  vt6040_trend  0.7 * vt6040 + 0.3 * trend sleeve
References: 60/40 (monthly), the current book, SPY.
Selection: best Sharpe on 2007-2019; then 2020+ is read once.
Frontier: the selected design at 8-16% vol targets, gross <= 200%, borrowing
spreads of 1%, 2% and 3% over T-bills.

Usage: python benchmarks/construction_research.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import replace

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge import live  # noqa: E402
from leveledge import portfolio as P  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "construction")
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"
GROSS, MAXW, TV = 1.5, 0.6, 0.10
CRISES = [("2008 crash", "2007-10-09", "2009-03-09"), ("2020 Covid crash", "2020-02-19", "2020-03-23"),
          ("2022 rate shock", "2022-01-03", "2022-10-12")]


def designs(closes, cash, vol, corr, dates, trend, tv=TV, gross=GROSS):
    core = pd.Series({"SPY": 0.6, "IEF": 0.4}).reindex(closes.columns).fillna(0.0)
    out = {k: {} for k in ("vt6040", "rp", "rp_trend", "vt6040_trend")}
    for d in dates:
        if d not in corr or d not in trend.index:
            continue
        v, c = vol.loc[d], corr[d]
        vt = P.scale_to_vol(core, v, c, tv, gross, MAXW)
        rp = P.risk_parity_weights(v, c, tv, gross, max_weight=MAXW)
        tr = trend.loc[d]
        out["vt6040"][d] = vt
        out["rp"][d] = rp
        out["rp_trend"][d] = 0.7 * rp + 0.3 * tr
        out["vt6040_trend"][d] = 0.7 * vt + 0.3 * tr
    return {k: pd.DataFrame(v).T.reindex(columns=closes.columns).fillna(0.0) for k, v in out.items()}


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cfg = live.config()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    vol = np.log(closes).diff().apply(P.ewma_vol)
    dates = P.rebalance_dates(closes.index, "weekly")
    corr = P.rolling_corr(closes, dates, cfg.corr_window)
    signal = P.momentum_signal(closes, cash, cfg.lookback)
    trend = P.strategy_weights(opens, closes, cash, cfg, vol=vol, signal=signal, corr=corr)

    D = designs(closes, cash, vol, corr, dates, trend)
    runs = {k: P.backtest(opens, closes, cash, w, 5.0, 0.02) for k, w in D.items()}
    book, _ = live.book_decisions(opens, closes, cash, cfg, vol=vol, signal=signal, corr=corr)
    runs["current_book"] = P.backtest(opens, closes, cash, book, 5.0)
    runs["sixty_forty"] = P.buy_and_hold(closes, {"SPY": 0.6, "IEF": 0.4}, cash, "monthly")
    runs["spy_buy_hold"] = P.buy_and_hold(closes, {"SPY": 1.0}, cash, None)
    start = max(r.index[0] for r in runs.values())
    runs = {k: v.loc[start:] for k, v in runs.items()}
    gross_avg = {k: round(float(v["gross"].mean()), 3) for k, v in runs.items()}

    res = {"period": [str(start.date()), str(closes.index[-1].date())], "avg_gross": gross_avg,
           "full": {k: P.metrics(v["ret"], cash, v["turnover"]) for k, v in runs.items()},
           "dev": {k: P.metrics(v["ret"].loc[:DEV_END], cash) for k, v in runs.items()},
           "holdout": {k: P.metrics(v["ret"].loc[HOLD_START:], cash) for k, v in runs.items()}}
    cands = ["vt6040", "rp", "rp_trend", "vt6040_trend"]
    sel = max(cands, key=lambda k: res["dev"][k]["sharpe"])
    res["selected"] = sel
    res["holdout_sharpe_vs"] = {k: P.sharpe_diff_ci(runs[sel]["ret"].loc[HOLD_START:], runs[k]["ret"].loc[HOLD_START:], cash)
                                for k in ("sixty_forty", "current_book", "spy_buy_hold")}
    res["full_sharpe_vs"] = {k: P.sharpe_diff_ci(runs[sel]["ret"], runs[k]["ret"], cash)
                             for k in ("sixty_forty", "current_book", "spy_buy_hold")}
    res["crises"] = [{"label": l, **{k: round(float((1 + v["ret"].loc[a:b]).prod() - 1), 4) for k, v in runs.items()}}
                     for l, a, b in CRISES]
    res["calendar_years"] = pd.DataFrame({k: (1 + v["ret"]).groupby(v.index.year).prod() - 1 for k, v in runs.items()}).round(4).to_dict(orient="index")

    # reward frontier: how much return each extra unit of risk buys, with leverage paid for.
    # (A per-ETF cap below the gross cap would silently stop the leverage, so caps = gross here.)
    def fixed_lever(dec: pd.DataFrame, L: float) -> pd.DataFrame:
        return dec * L

    sixty = pd.DataFrame([pd.Series({"SPY": 0.6, "IEF": 0.4})] * len(D["vt6040"]), index=D["vt6040"].index).reindex(columns=closes.columns).fillna(0.0)
    frontier = []
    families = {
        "current_book_levered": [(L, fixed_lever(book, L)) for L in (1.0, 1.25, 1.5, 1.75, 2.0)],
        "sixty_forty_levered": [(L, fixed_lever(sixty, L)) for L in (1.0, 1.25, 1.5, 1.75, 2.0)],
        "vt6040_targeted": [],
    }
    for tv in (0.10, 0.12, 0.14, 0.16, 0.18):
        core = pd.Series({"SPY": 0.6, "IEF": 0.4}).reindex(closes.columns).fillna(0.0)
        rows = {d: P.scale_to_vol(core, vol.loc[d], corr[d], tv, 2.0, 2.0) for d in D["vt6040"].index}
        families["vt6040_targeted"].append((tv, pd.DataFrame(rows).T.reindex(columns=closes.columns).fillna(0.0)))
    for fam, items in families.items():
        for knob, dec in items:
            for spread in (0.01, 0.02, 0.03):
                r = P.backtest(opens, closes, cash, dec, 5.0, spread).loc[start:]
                m, mh = P.metrics(r["ret"], cash), P.metrics(r["ret"].loc[HOLD_START:], cash)
                yrs = (1 + r["ret"]).groupby(r.index.year).prod() - 1
                frontier.append({"family": fam, "knob": knob, "borrow_spread": spread, "avg_gross": round(float(r["gross"].mean()), 3),
                                 "cagr": m["cagr"], "vol": m["vol"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                                 "worst_year": round(float(yrs.min()), 4), "holdout_cagr": mh["cagr"],
                                 "holdout_sharpe": mh["sharpe"], "holdout_max_drawdown": mh["max_drawdown"]})
    res["frontier"] = frontier
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    pd.DataFrame({k: (1 + v["ret"]).cumprod() for k, v in runs.items()}).to_csv(os.path.join(OUT, "equity.csv"))

    cols = ["cagr", "vol", "sharpe", "max_drawdown", "turnover_per_year"]
    print("FULL", res["period"]); print(pd.DataFrame(res["full"]).T[cols].assign(gross=pd.Series(gross_avg)).to_string())
    print("DEV sharpe:", {k: v["sharpe"] for k, v in res["dev"].items()})
    print("HOLDOUT:", {k: (v["cagr"], v["sharpe"], v["max_drawdown"]) for k, v in res["holdout"].items()})
    print("selected:", sel, "holdout vs:", json.dumps(res["holdout_sharpe_vs"]), "full vs:", json.dumps(res["full_sharpe_vs"]))
    print("crises:", res["crises"])
    F = pd.DataFrame(frontier)
    print(F[F.borrow_spread == 0.02].drop(columns="borrow_spread").to_string())


if __name__ == "__main__":
    main()
