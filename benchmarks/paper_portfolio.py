#!/usr/bin/env python3
"""
Backtest of the exact book the paper loop trades (leveledge.live), for the dashboard.

Book = 70% (60% SPY / 40% IEF) + 30% trend sleeve, rebalanced weekly at the
next open with 5 bp costs. Compared with 60/40, SPY, the trend sleeve alone
and a "random sleeve" book (same core, random long/flat sleeve signals).
Development (to 2019) and holdout (2020+) are reported separately, matching
etf_research.py.

Usage: python benchmarks/paper_portfolio.py
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
OUT = os.path.join(ROOT, "benchmarks", "results", "paper_portfolio")


def main(n_random: int = 200):
    os.makedirs(OUT, exist_ok=True)
    cfg = live.config()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    vol = np.log(closes).diff().apply(P.ewma_vol)
    signal = P.momentum_signal(closes, cash, cfg.lookback)
    dates = P.rebalance_dates(closes.index, cfg.rebalance)
    corr = P.rolling_corr(closes, dates, cfg.corr_window)
    book, trend = live.book_decisions(opens, closes, cash, cfg, vol=vol, signal=signal, corr=corr)
    bt = P.backtest(opens, closes, cash, book, cfg.cost_bps)
    start = bt.index[0]
    s = {
        "book": bt,
        "sixty_forty": P.buy_and_hold(closes, {"SPY": 0.6, "IEF": 0.4}, cash, "monthly").loc[start:],
        "spy_buy_hold": P.buy_and_hold(closes, {"SPY": 1.0}, cash, None).loc[start:],
        "trend_sleeve_alone": P.backtest(opens, closes, cash, trend, cfg.cost_bps).loc[start:],
    }
    labels = {"book": "LevelEdge book (1.5x: 60/40 + 30% trend)", "sixty_forty": "60/40", "spy_buy_hold": "SPY buy-and-hold",
              "trend_sleeve_alone": "Trend sleeve alone"}

    # random-sleeve books: same core, sleeve signals drawn at the sleeve's long rate
    p_long = float(signal.loc[trend.index].stack().mean())
    rng = np.random.default_rng(11)
    rsh, rdd = [], []
    for _ in range(n_random):
        rs = signal.copy()
        rs[:] = np.where(signal.notna(), (rng.random(signal.shape) < p_long).astype(float), np.nan)
        rb, _ = live.book_decisions(opens, closes, cash, cfg, vol=vol, signal=rs, corr=corr)
        r = P.backtest(opens, closes, cash, rb, cfg.cost_bps)["ret"]
        m = P.metrics(r, cash)
        rsh.append(m["sharpe"])
        rdd.append(m["max_drawdown"])

    res = {"strategy": live.STRATEGY_ID, "period": [str(start.date()), str(closes.index[-1].date())],
           "labels": labels, "cost_bps": cfg.cost_bps,
           "metrics": {k: P.metrics(v["ret"], cash, v.get("turnover")) for k, v in s.items()},
           "dev": {k: P.metrics(v["ret"].loc[:"2019-12-31"], cash) for k, v in s.items()},
           "holdout": {k: P.metrics(v["ret"].loc["2020-01-01":], cash) for k, v in s.items()},
           "sharpe_vs_sixty_forty": {
               "full": P.sharpe_diff_ci(s["book"]["ret"], s["sixty_forty"]["ret"], cash),
               "holdout": P.sharpe_diff_ci(s["book"]["ret"].loc["2020-01-01":], s["sixty_forty"]["ret"].loc["2020-01-01":], cash)},
           "random_sleeve": {"sims": n_random, "p_long": round(p_long, 3),
                             "sharpe_p5_median_p95": [round(float(np.percentile(rsh, q)), 3) for q in (5, 50, 95)],
                             "max_dd_median": round(float(np.median(rdd)), 4),
                             "p_value_book_sharpe": round(float((np.array(rsh) >= P.metrics(bt["ret"], cash)["sharpe"]).mean()), 4)}}
    windows = [("2008 crash", "2007-10-09", "2009-03-09"), ("2020 Covid crash", "2020-02-19", "2020-03-23"),
               ("2022 rate shock", "2022-01-03", "2022-10-12")]
    res["crisis_windows"] = [{"label": l, "start": a, "end": b,
                              **{k: round(float((1 + v["ret"].loc[a:b]).prod() - 1), 4) for k, v in s.items()}}
                             for l, a, b in windows]
    res["calendar_years"] = [{"year": int(y), **{k: round(float(v), 4) for k, v in row.items()}} for y, row in
                             pd.DataFrame({k: (1 + v["ret"]).groupby(v.index.year).prod() - 1 for k, v in s.items()}).iterrows()]
    eq = pd.DataFrame({k: (1 + v["ret"]).cumprod() for k, v in s.items()})
    weekly = eq.resample("W-FRI").last().dropna()
    dd = (eq / eq.cummax() - 1).resample("W-FRI").min().dropna()
    res["equity_weekly"] = [{"date": d.strftime("%Y-%m-%d"), **{k: round(float(x), 5) for k, x in r.items()}}
                            for d, r in weekly.iterrows()]
    res["drawdown_weekly"] = [{"date": d.strftime("%Y-%m-%d"), **{k: round(float(x), 5) for k, x in r.items()}}
                              for d, r in dd.iterrows()]
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    cols = ["cagr", "vol", "sharpe", "max_drawdown", "calmar", "turnover_per_year"]
    print(pd.DataFrame(res["metrics"]).T[cols].to_string())
    print("holdout:", {k: (v["cagr"], v["sharpe"], v["max_drawdown"]) for k, v in res["holdout"].items()})
    print("sharpe vs 60/40:", res["sharpe_vs_sixty_forty"], "random:", res["random_sleeve"])
    print("crises:", res["crisis_windows"])


if __name__ == "__main__":
    main()
