#!/usr/bin/env python3
"""
Most return within a drawdown limit: trend-sleeve share x leverage.

Protocol (fixed before running)
-------------------------------
Book(s, L) = L * [(1 - s) * 60/40 + s * trend sleeve], weekly, next-open fills,
5 bp costs, borrowing at T-bill + 2%.
Grid: s in {0.2, 0.3, 0.4, 0.5}, L in {1.0, 1.25, 1.5, 1.75, 2.0}  (20 books).
Selection on 2007-2019: highest CAGR among books with max drawdown no worse
than -35%. 2020+ is then read once for the selected book, the current book
(s = 0.3, L = 1) and SPY. Reported with the deflated Sharpe for 20 trials.

Usage: python benchmarks/reward_research.py
"""
from __future__ import annotations

import itertools
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
OUT = os.path.join(ROOT, "benchmarks", "results", "reward")
DD_LIMIT = -0.35
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = live.config()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    vol = np.log(closes).diff().apply(P.ewma_vol)
    dates = P.rebalance_dates(closes.index, "weekly")
    corr = P.rolling_corr(closes, dates, cfg.corr_window)
    signal = P.momentum_signal(closes, cash, cfg.lookback)
    trend = P.strategy_weights(opens, closes, cash, cfg, vol=vol, signal=signal, corr=corr)
    core = pd.Series({"SPY": 0.6, "IEF": 0.4}).reindex(closes.columns).fillna(0.0)
    core_df = pd.DataFrame([core] * len(trend), index=trend.index)

    runs, rows = {}, []
    for s, L in itertools.product((0.2, 0.3, 0.4, 0.5), (1.0, 1.25, 1.5, 1.75, 2.0)):
        dec = L * ((1 - s) * core_df + s * trend)
        r = P.backtest(opens, closes, cash, dec, 5.0, 0.02)
        key = f"s={s}|L={L}"
        runs[key] = r["ret"]
        md, mh, mf = (P.metrics(r["ret"].loc[:DEV_END], cash), P.metrics(r["ret"].loc[HOLD_START:], cash),
                      P.metrics(r["ret"], cash))
        rows.append({"book": key, "trend_share": s, "leverage": L, "dev_cagr": md["cagr"], "dev_maxdd": md["max_drawdown"],
                     "dev_sharpe": md["sharpe"], "hold_cagr": mh["cagr"], "hold_maxdd": mh["max_drawdown"],
                     "hold_sharpe": mh["sharpe"], "full_cagr": mf["cagr"], "full_maxdd": mf["max_drawdown"],
                     "full_vol": mf["vol"]})
    V = pd.DataFrame(rows)
    ok = V[V["dev_maxdd"] >= DD_LIMIT]
    sel = ok.sort_values("dev_cagr", ascending=False).iloc[0]["book"]
    spy = P.buy_and_hold(closes, {"SPY": 1.0}, cash, None)["ret"].loc[runs[sel].index[0]:]
    ex = runs[sel].loc[:DEV_END] - cash.reindex(runs[sel].loc[:DEV_END].index).fillna(0)
    trial_sr = np.array([(runs[k].loc[:DEV_END] - cash.reindex(runs[k].loc[:DEV_END].index).fillna(0)).pipe(
        lambda x: x.mean() / x.std(ddof=1)) for k in runs])
    res = {"dd_limit": DD_LIMIT, "selected": sel, "grid": V.to_dict(orient="records"),
           "deflated_sharpe_dev": P.deflated_sharpe(ex, trial_sr),
           "holdout": {"selected": P.metrics(runs[sel].loc[HOLD_START:], cash),
                       "current_book": P.metrics(runs["s=0.3|L=1.0"].loc[HOLD_START:], cash),
                       "spy_buy_hold": P.metrics(spy.loc[HOLD_START:], cash)},
           "full": {"selected": P.metrics(runs[sel], cash), "current_book": P.metrics(runs["s=0.3|L=1.0"], cash),
                    "spy_buy_hold": P.metrics(spy, cash)}}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    pd.set_option("display.width", 200)
    print(V.round(4).to_string())
    print("selected:", sel)
    for part in ("full", "holdout"):
        print(part, {k: (v["cagr"], v["vol"], v["sharpe"], v["max_drawdown"]) for k, v in res[part].items()})


if __name__ == "__main__":
    main()
