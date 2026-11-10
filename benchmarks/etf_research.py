#!/usr/bin/env python3
"""
Strategy search with a held-out period, so the search can't fool us.

Protocol (fixed before running)
-------------------------------
* Development: first decision (Mar 2007) .. 2019-12-31. All 80 variants are
  scored here; the one with the highest development Sharpe is selected.
* Deflated Sharpe ratio for the selected variant, given all 80 trials.
* Holdout: 2020-01-01 .. latest. The selected variant is compared with 60/40
  and SPY once, with block-bootstrap CIs. Caveat: the original mom12/voltarget
  strategy's post-2020 results were seen before this search.
* Rank stability: Spearman correlation of development vs holdout Sharpe across
  all variants (reported after selection; it does not change the selection).

Grid: 8 signals x {equal, invvol} x {weekly, monthly} x {cash, IEF}
      + 8 signals x voltarget(10%) x {weekly, monthly} x cash  = 80 variants.

Usage: python benchmarks/etf_research.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge import portfolio as P  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "etf_research")
DEV_END = "2019-12-31"
HOLD_START = "2020-01-01"


def grid() -> list[dict]:
    specs = []
    for sig, sz, reb, dfn in itertools.product(P.SIGNALS, ["equal", "invvol"], ["weekly", "monthly"], [None, "IEF"]):
        specs.append({"signal": sig, "sizing": sz, "rebalance": reb, "defensive": dfn})
    for sig, reb in itertools.product(P.SIGNALS, ["weekly", "monthly"]):
        specs.append({"signal": sig, "sizing": "voltarget", "rebalance": reb, "defensive": None})
    return specs


def name(s: dict) -> str:
    return f"{s['signal']}|{s['sizing']}|{s['rebalance']}|{s['defensive'] or 'cash'}"


def sharpe(r: pd.Series, cash: pd.Series) -> float:
    ex = r - cash.reindex(r.index).fillna(0)
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(P.TRADING_DAYS))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cfg = P.StrategyConfig()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    vol = np.log(closes).diff().apply(P.ewma_vol)          # chosen vol engine (etf_trend.py)
    corr = {}
    for reb in ("weekly", "monthly"):
        corr.update(P.rolling_corr(closes, P.rebalance_dates(closes.index, reb), cfg.corr_window))

    specs = grid()
    rets, turns = {}, {}
    for s in specs:
        w = P.variant_weights(closes, cash, vol, corr, s)
        bt = P.backtest(opens, closes, cash, w, cfg.cost_bps)
        rets[name(s)], turns[name(s)] = bt["ret"], bt["turnover"]
    R = pd.DataFrame(rets)
    start = R.dropna().index[0]          # common start so every variant covers the same days
    R = R.loc[start:]
    T = pd.DataFrame(turns).loc[start:]
    bench = {
        "sixty_forty": P.buy_and_hold(closes, {"SPY": 0.6, "IEF": 0.4}, cash, "monthly")["ret"].loc[start:],
        "spy_buy_hold": P.buy_and_hold(closes, {"SPY": 1.0}, cash, rebalance=None)["ret"].loc[start:],
    }
    dev, hold = R.loc[:DEV_END], R.loc[HOLD_START:]

    rows = []
    for c in R.columns:
        md, mh = P.metrics(dev[c], cash, T[c].loc[:DEV_END]), P.metrics(hold[c], cash, T[c].loc[HOLD_START:])
        rows.append({"variant": c, "dev_sharpe": md["sharpe"], "dev_cagr": md["cagr"], "dev_maxdd": md["max_drawdown"],
                     "dev_turnover": md.get("turnover_per_year"), "hold_sharpe": mh["sharpe"], "hold_cagr": mh["cagr"],
                     "hold_maxdd": mh["max_drawdown"]})
    V = pd.DataFrame(rows).sort_values("dev_sharpe", ascending=False).reset_index(drop=True)
    V.to_csv(os.path.join(OUT, "variants.csv"), index=False)

    best = V.iloc[0]["variant"]
    trial_sr_daily = np.array([sharpe(dev[c], cash) / np.sqrt(P.TRADING_DAYS) for c in R.columns])
    ex_dev = dev[best] - cash.reindex(dev.index).fillna(0)
    res = {
        "period": {"dev": [str(dev.index[0].date()), DEV_END], "holdout": [HOLD_START, str(hold.index[-1].date())]},
        "trials": len(specs),
        "selected": best,
        "deflated_sharpe_dev": P.deflated_sharpe(ex_dev, trial_sr_daily),
        "dev": {"selected": P.metrics(dev[best], cash, T[best].loc[:DEV_END]),
                "sixty_forty": P.metrics(bench["sixty_forty"].loc[:DEV_END], cash),
                "spy_buy_hold": P.metrics(bench["spy_buy_hold"].loc[:DEV_END], cash),
                "original_mom12_voltarget_weekly": P.metrics(dev["mom12|voltarget|weekly|cash"], cash)},
        "holdout": {"selected": P.metrics(hold[best], cash, T[best].loc[HOLD_START:]),
                    "sixty_forty": P.metrics(bench["sixty_forty"].loc[HOLD_START:], cash),
                    "spy_buy_hold": P.metrics(bench["spy_buy_hold"].loc[HOLD_START:], cash)},
        "holdout_sharpe_vs": {k: P.sharpe_diff_ci(hold[best], bench[k].loc[HOLD_START:], cash)
                              for k in ("sixty_forty", "spy_buy_hold")},
    }
    combo = 0.7 * bench["sixty_forty"] + 0.3 * R[best]
    res["holdout_combo_6040_plus_30pct_selected"] = {
        "metrics": P.metrics(combo.loc[HOLD_START:], cash),
        "sharpe_vs_sixty_forty": P.sharpe_diff_ci(combo.loc[HOLD_START:], bench["sixty_forty"].loc[HOLD_START:], cash)}
    rho, pv = spearmanr(V["dev_sharpe"], V["hold_sharpe"])
    res["rank_stability"] = {"spearman_dev_vs_holdout": round(float(rho), 3), "p_value": round(float(pv), 4),
                             "top10_dev_holdout_ranks": V["hold_sharpe"].rank(ascending=False).head(10).astype(int).tolist(),
                             "holdout_sharpe_of_dev_top10_mean": round(float(V["hold_sharpe"].head(10).mean()), 3),
                             "holdout_sharpe_all_mean": round(float(V["hold_sharpe"].mean()), 3)}
    res["by_signal_dev_sharpe_mean"] = V.assign(sig=V.variant.str.split("|").str[0]).groupby("sig")["dev_sharpe"].mean().round(3).to_dict()
    res["by_signal_hold_sharpe_mean"] = V.assign(sig=V.variant.str.split("|").str[0]).groupby("sig")["hold_sharpe"].mean().round(3).to_dict()
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(V.head(12).to_string())
    print(json.dumps({k: res[k] for k in ["selected", "deflated_sharpe_dev", "holdout_sharpe_vs", "rank_stability"]}, indent=1))


if __name__ == "__main__":
    main()
