#!/usr/bin/env python3
"""
Do richer return models (pooled ridge, splines, mixed effects, boosted trees) beat
today's trend rule, and can a better Sharpe buy more return at a hedged risk level?

Protocol (fixed before running)
-------------------------------
* Weekly decisions, yearly expanding-window refits from 2011, penalties chosen by
  inner time-series CV on the last 3 training years, purged one-week labels.
* "auto" picks, each year, the model with the best out-of-sample R^2 over the
  previous 3 years of forecasts already made, so model choice is out of sample too.
* Every model uses the same construction (risk budget proportional to the
  positive forecast, 10% vol target, no leverage, 25% cap), the same daily
  engine (next-open fills, 5 bp costs) and the same period (2011 onward).
* Primary comparison: auto vs the rule, Sharpe difference with a block-bootstrap CI.
* Reward test: the auto book at a 15% vol target with up to 1.5x gross, against
  60/40 levered to the same realised volatility (financing at the T-bill rate).

Usage: python benchmarks/alpha_research.py [--universe main|fresh]
"""
from __future__ import annotations

import argparse
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
from leveledge import signals as S  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
UNIVERSES = {
    "main": list(P.UNIVERSE),
    # never used in any earlier search: US sectors, developed-country equity, plus bonds/gold as hedges
    "fresh": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "EWJ", "EWG", "EWU", "EWC", "EWA",
              "SPY", "IEF", "TLT", "GLD"],
}
NAMES = ["rule", "ridge", "spline", "mixed", "xgb"]


def lever_to(ret: pd.Series, cash: pd.Series, target: float) -> pd.Series:
    c = cash.reindex(ret.index).fillna(0)
    L = target / (ret.std() * np.sqrt(P.TRADING_DAYS))
    return c + L * (ret - c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="main", choices=list(UNIVERSES))
    ap.add_argument("--first-year", type=int, default=2011)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.join(ROOT, "benchmarks", "results", f"alpha_research_{args.universe}")
    os.makedirs(out_dir, exist_ok=True)
    syms = UNIVERSES[args.universe]
    prices = P.load_prices(syms, start="2003-01-01", cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, syms)
    print(f"[{args.universe}] {len(syms)} ETFs, {closes.index[0].date()} .. {closes.index[-1].date()}", flush=True)
    dates = P.rebalance_dates(closes.index, "weekly")
    panel = S.weekly_panel(closes, cash, dates)
    print(f"panel {len(panel)} rows, {panel['date'].min().date()} .. {panel['date'].max().date()}", flush=True)

    oos, log = S.walk_forward(panel, NAMES, args.first_year)
    oos.to_csv(os.path.join(out_dir, "oos_forecasts.csv"), index=False)
    cols = NAMES + ["auto"]
    res = {"universe": args.universe, "symbols": syms, "first_year": args.first_year,
           "forecast": {c: S.forecast_stats(oos, c) for c in cols},
           "auto_picks": log.drop_duplicates("year")[["year", "auto_pick"]].to_dict(orient="records"),
           "penalties": log.pivot(index="year", columns="model", values="alpha").reset_index().to_dict(orient="records")}
    print("forecast:", json.dumps(res["forecast"]), flush=True)

    corr = P.rolling_corr(closes, dates, 126)
    base = S.Construction()
    start = pd.Timestamp(f"{args.first_year}-01-01")
    runs = {}
    for c in cols:
        w = S.decisions_from(oos, c, closes, corr, base)
        runs[c] = P.backtest(opens, closes, cash, w, 5.0).loc[start:]
    reward = S.Construction(target_vol=0.15, max_gross=1.5, max_weight=0.4)
    for c in ("auto", "rule"):
        w = S.decisions_from(oos, c, closes, corr, reward)
        runs[f"{c}_reward"] = P.backtest(opens, closes, cash, w, 5.0).loc[start:]
    if "IEF" in closes and "SPY" in closes:
        runs["sixty_forty"] = P.buy_and_hold(closes, {"SPY": 0.6, "IEF": 0.4}, cash, "monthly").loc[start:]
        runs["spy_buy_hold"] = P.buy_and_hold(closes, {"SPY": 1.0}, cash, None).loc[start:]
    if args.universe == "main":
        book, _ = live.book_decisions(opens, closes, cash)
        runs["current_book"] = P.backtest(opens, closes, cash, book, 5.0).loc[start:]
    common = runs["rule"].index
    for k in runs:
        runs[k] = runs[k].reindex(common).dropna()

    res["metrics"] = {k: P.metrics(v["ret"], cash, v["turnover"]) for k, v in runs.items()}
    res["sharpe_vs_rule"] = {c: P.sharpe_diff_ci(runs[c]["ret"], runs["rule"]["ret"], cash) for c in cols if c != "rule"}
    if "sixty_forty" in runs:
        res["sharpe_vs_sixty_forty"] = {c: P.sharpe_diff_ci(runs[c]["ret"], runs["sixty_forty"]["ret"], cash)
                                        for c in cols + ["auto_reward"]}
        rv = runs["auto_reward"]["ret"].std() * np.sqrt(P.TRADING_DAYS)
        lev = lever_to(runs["sixty_forty"]["ret"], cash, rv)
        res["reward_vs_levered_sixty_forty"] = {"vol": round(float(rv), 4),
                                                "auto_reward": P.metrics(runs["auto_reward"]["ret"], cash),
                                                "sixty_forty_levered": P.metrics(lev, cash)}
    res["subperiods"] = {f"{a}-{b}": {k: P.metrics(runs[k]["ret"].loc[a:b], cash) for k in ("rule", "auto", "spline", "mixed")
                                      if len(runs[k]["ret"].loc[a:b]) > 50} for a, b in [("2011", "2015"), ("2016", "2020"), ("2021", "2026")]}
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    pd.DataFrame({k: (1 + v["ret"]).cumprod() for k, v in runs.items()}).to_csv(os.path.join(out_dir, "equity.csv"))
    cols_m = ["cagr", "vol", "sharpe", "max_drawdown", "turnover_per_year"]
    print(pd.DataFrame(res["metrics"]).T[cols_m].to_string())
    print("sharpe vs rule:", json.dumps(res["sharpe_vs_rule"]))
    print("auto picks:", [p["auto_pick"] for p in res["auto_picks"]])
    if "reward_vs_levered_sixty_forty" in res:
        r = res["reward_vs_levered_sixty_forty"]
        print("reward:", {k: (v["cagr"], v["sharpe"], v["max_drawdown"]) for k, v in r.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
