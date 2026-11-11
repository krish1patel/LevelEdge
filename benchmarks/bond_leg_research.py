#!/usr/bin/env python3
"""
One hypothesis: trend-filter the book's bond leg.

In the 1.5x book, the core's 40% bond leg holds IEF only while IEF's 12-month
return beats cash, and SHY (short-term Treasuries) otherwise. Checked weekly,
like the rest of the book. Nothing else changes.

Pre-declared adoption rule: adopt only if (1) Sharpe on 2007-2019 improves,
(2) the 2007-2026 worst drawdown is no deeper, and (3) Sharpe from 2020 is not
lower. Caveat: the idea came from 2022 (stocks and margin-financed bonds fell
together), so 2020+ is not independent evidence for it; 2007-2019 carries the
weight (it includes the 2013 taper tantrum and other bond sell-offs).

Usage: python benchmarks/bond_leg_research.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge import live  # noqa: E402
from leveledge import portfolio as P  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "bond_leg")
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = live.config()
    syms = list(cfg.universe) + ["SHY"]
    prices = P.load_prices(syms, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, syms)
    tu = cfg.universe
    vol = np.log(closes[tu]).diff().apply(P.ewma_vol)
    dates = P.rebalance_dates(closes.index, "weekly")
    corr = P.rolling_corr(closes[tu], dates, cfg.corr_window)
    sig = P.momentum_signal(closes[tu], cash, cfg.lookback)
    trend = P.strategy_weights(opens[tu], closes[tu], cash, cfg, vol=vol, signal=sig, corr=corr)
    trend = trend.reindex(columns=closes.columns).fillna(0.0)
    ief_on = P.momentum_signal(closes[["IEF"]], cash, 252)["IEF"]

    def book(filtered: bool) -> pd.DataFrame:
        rows = {}
        for d in trend.index:
            core = pd.Series(0.0, index=closes.columns)
            core["SPY"] = 0.6
            bond = "IEF" if (not filtered or ief_on.get(d, 1.0) == 1.0) else "SHY"
            core[bond] = 0.4
            rows[d] = live.LEVERAGE * (live.CORE_SHARE * core + live.TREND_SHARE * trend.loc[d])
        return pd.DataFrame(rows).T

    runs = {"current": P.backtest(opens, closes, cash, book(False), 5.0, 0.02)["ret"],
            "bond_filtered": P.backtest(opens, closes, cash, book(True), 5.0, 0.02)["ret"],
            "sixty_forty": P.buy_and_hold(closes, {"SPY": 0.6, "IEF": 0.4}, cash, "monthly")["ret"]}
    start = max(r.index[0] for r in runs.values())
    runs = {k: v.loc[start:] for k, v in runs.items()}
    res = {"period": [str(start.date()), str(closes.index[-1].date())],
           "share_of_weeks_in_shy": round(float((ief_on.reindex(trend.index) == 0).mean()), 3)}
    for part, sl in (("full", slice(None)), ("dev", slice(None, DEV_END)), ("holdout", slice(HOLD_START, None))):
        res[part] = {k: P.metrics(v.loc[sl], cash) for k, v in runs.items()}
    res["sharpe_diff_dev"] = P.sharpe_diff_ci(runs["bond_filtered"].loc[:DEV_END], runs["current"].loc[:DEV_END], cash)
    res["sharpe_diff_holdout"] = P.sharpe_diff_ci(runs["bond_filtered"].loc[HOLD_START:], runs["current"].loc[HOLD_START:], cash)
    res["crises"] = [{"label": l, **{k: round(float((1 + runs[k].loc[a:b]).prod() - 1), 4) for k in runs}}
                     for l, a, b in [("2008 crash", "2007-10-09", "2009-03-09"), ("2013 taper tantrum", "2013-05-01", "2013-09-05"),
                                     ("2020 Covid crash", "2020-02-19", "2020-03-23"), ("2022 rate shock", "2022-01-03", "2022-10-12")]]
    d, h, f = res["dev"], res["holdout"], res["full"]
    res["rule"] = {"dev_sharpe_better": d["bond_filtered"]["sharpe"] > d["current"]["sharpe"],
                   "drawdown_not_deeper": f["bond_filtered"]["max_drawdown"] >= f["current"]["max_drawdown"],
                   "holdout_sharpe_not_lower": h["bond_filtered"]["sharpe"] >= h["current"]["sharpe"]}
    res["adopt"] = all(res["rule"].values())
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    cols = ["cagr", "vol", "sharpe", "max_drawdown"]
    for part in ("full", "dev", "holdout"):
        print(f"--- {part}"); print(pd.DataFrame(res[part]).T[cols].to_string())
    print("weeks in SHY:", res["share_of_weeks_in_shy"])
    print("sharpe diff dev:", res["sharpe_diff_dev"], "holdout:", res["sharpe_diff_holdout"])
    for c in res["crises"]:
        print(" ", c)
    print("rule:", res["rule"], "-> adopt:", res["adopt"])


if __name__ == "__main__":
    main()
