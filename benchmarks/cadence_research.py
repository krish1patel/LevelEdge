#!/usr/bin/env python3
"""
Check the book's triggers daily instead of weekly?

Weekly (current): targets computed on the last trading day of each week; an ETF
trades only if it is more than 2 points off target. Daily: the same targets and
band, checked every trading day, so a trend flip or a large drift is acted on the
next morning instead of after Friday.

Pre-declared adoption rule: switch to daily only if Sharpe is not lower in
2007-2019 or in 2020+, and orders a year at most double.

Usage: python benchmarks/cadence_research.py
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
sys.path.insert(0, HERE)

from leveledge import live  # noqa: E402
from leveledge import portfolio as P  # noqa: E402
from literature_research import banded_book  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "cadence")


def book_targets(opens, closes, cash, cfg, dates, vol, sig):
    corr = P.rolling_corr(closes, dates, cfg.corr_window)
    rows = {}
    for d in dates:
        i = closes.index.get_loc(d)
        if i < max(cfg.lookback, cfg.corr_window) + 1 or d not in corr:
            continue
        rows[d] = live.book_from_trend(P.target_weights(sig.loc[d], vol.loc[d], corr[d], cfg))
    return pd.DataFrame(rows).T.reindex(columns=closes.columns).fillna(0.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = live.config()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    vol = np.log(closes).diff().apply(P.ewma_vol)
    sig = P.momentum_signal(closes, cash, cfg.lookback)
    weekly = book_targets(opens, closes, cash, cfg, P.rebalance_dates(closes.index, "weekly"), vol, sig)
    daily = book_targets(opens, closes, cash, cfg, closes.index, vol, sig)
    runs = {"weekly_band2": banded_book(opens, closes, cash, weekly, live.BAND),
            "daily_band2": banded_book(opens, closes, cash, daily, live.BAND),
            "daily_band3": banded_book(opens, closes, cash, daily, 0.03)}
    start = max(r.index[0] for r in runs.values())
    res = {}
    for k, bt in runs.items():
        r = bt["ret"].loc[start:]
        yrs = len(r) / P.TRADING_DAYS
        res[k] = {"full": P.metrics(r, cash), "dev": P.metrics(r.loc[:"2019-12-31"], cash), "holdout": P.metrics(r.loc["2020-01-01":], cash),
                  "orders_per_year": round(bt.attrs["orders"] / (len(bt) / P.TRADING_DAYS), 1),
                  "cost_bp_per_year": round(float(bt["turnover"].loc[start:].sum() * 5 / yrs), 1)}
    base = res["weekly_band2"]
    for k in ("daily_band2", "daily_band3"):
        v = res[k]
        v["adopt"] = (v["dev"]["sharpe"] >= base["dev"]["sharpe"] and v["holdout"]["sharpe"] >= base["holdout"]["sharpe"]
                      and v["orders_per_year"] <= 2 * base["orders_per_year"])
        v["sharpe_diff_full"] = P.sharpe_diff_ci(runs[k]["ret"].loc[start:], runs["weekly_band2"]["ret"].loc[start:], cash)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    for k, v in res.items():
        print(f"{k:13s} full cagr {v['full']['cagr']:.4f} sharpe {v['full']['sharpe']:.3f} maxdd {v['full']['max_drawdown']:.3f} | "
              f"dev sharpe {v['dev']['sharpe']:.3f} | 2020+ sharpe {v['holdout']['sharpe']:.3f} | orders/yr {v['orders_per_year']} | "
              f"cost {v['cost_bp_per_year']} bp/yr | adopt {v.get('adopt', '-')} {v.get('sharpe_diff_full', '')}")


if __name__ == "__main__":
    main()
