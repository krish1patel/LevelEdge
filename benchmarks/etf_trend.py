#!/usr/bin/env python3
"""
Vol-targeted ETF trend portfolio: does it earn its keep against simple benchmarks?

Pre-declared before running (see leveledge.portfolio for the rules):
  * strategy = 12-month time-series momentum, long/flat, 10% vol target,
    weekly rebalance, next-open fills, 5 bp costs, no leverage;
  * vol engine = whichever of EWMA and LevelEdge's VolScaler has the lower
    out-of-sample QLIKE on next-week realised variance (step 1);
  * primary comparison = Sharpe vs SPY buy-and-hold, with a block-bootstrap CI;
  * the random picker gets the same sizing with random long/flat signals at
    the strategy's average long rate.

Usage: python benchmarks/etf_trend.py [--random 300]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge import portfolio as P  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "etf_trend")
SUBPERIODS = [("2007", "2014"), ("2015", "2019"), ("2020", "2026")]


def vol_comparison(closes: pd.DataFrame, horizon: int) -> tuple[dict, dict]:
    rets = np.log(closes).diff()
    fwd_var = sum(rets.shift(-k) ** 2 for k in range(1, horizon + 1))   # next-week realised variance
    ew = rets.apply(P.ewma_vol)
    le = rets.apply(lambda s: P.leveledge_vol(s, horizon))
    start = le.dropna(how="all").index[0]
    out = {}
    for c in closes.columns:
        m = (fwd_var.index >= start) & fwd_var[c].notna()
        rv = fwd_var[c][m].to_numpy()
        out[c] = {"ewma": round(P.qlike(rv, horizon * ew[c][m].to_numpy() ** 2), 4),
                  "leveledge": round(P.qlike(rv, horizon * le[c][m].to_numpy() ** 2), 4)}
    avg = {k: round(float(np.mean([v[k] for v in out.values()])), 4) for k in ("ewma", "leveledge")}
    return {"per_asset_qlike": out, "mean_qlike": avg, "from": str(start.date())}, {"ewma": ew, "leveledge": le}


_G: dict = {}


def _init(g):
    global _G
    _G = g


def _random_run(seed):
    g = _G
    rng = np.random.default_rng(seed)
    sig = g["signal"].copy()
    mask = sig.notna().to_numpy()
    sig[:] = np.where(mask, (rng.random(sig.shape) < g["p_long"]).astype(float), np.nan)
    # random draws are only used on decision dates; hold them fixed within the week
    w = P.strategy_weights(g["opens"], g["closes"], g["cash"], g["cfg"], vol=g["vol"], signal=sig, corr=g["corr"])
    bt = P.backtest(g["opens"], g["closes"], g["cash"], w, g["cfg"].cost_bps)
    r = bt["ret"].loc[g["start"]:]
    ex = r - g["cash"].reindex(r.index).fillna(0)
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(252)), float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=300)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cfg = P.StrategyConfig()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    print(f"panel {closes.index[0].date()} .. {closes.index[-1].date()}  ({len(closes)} days)", flush=True)

    # 1. vol engine, chosen by out-of-sample QLIKE
    vc, vols = vol_comparison(closes, cfg.vol_horizon)
    winner = min(vc["mean_qlike"], key=vc["mean_qlike"].get)
    print("vol QLIKE", vc["mean_qlike"], "->", winner, flush=True)
    cfg = replace(cfg, vol_model=winner)

    # 2. strategy and its sensitivity to the vol engine
    signal = P.momentum_signal(closes, cash, cfg.lookback)
    dates = P.rebalance_dates(closes.index, cfg.rebalance)
    corr = P.rolling_corr(closes, dates, cfg.corr_window)
    runs = {}
    for name, vm in [("strategy", winner), ("strategy_other_vol", "leveledge" if winner == "ewma" else "ewma")]:
        w = P.strategy_weights(opens, closes, cash, replace(cfg, vol_model=vm), vol=vols[vm], signal=signal, corr=corr)
        runs[name] = (P.backtest(opens, closes, cash, w, cfg.cost_bps), w)
    start = runs["strategy"][0].index[0]

    # attribution: trend without vol targeting (equal weight among held), vol targeting without trend
    ones = signal.notna().astype(float).where(signal.notna())
    w_vt = P.strategy_weights(opens, closes, cash, cfg, vol=vols[winner], signal=ones, corr=corr)
    eqw = signal.loc[dates].dropna(how="all").fillna(0)
    eqw = eqw.div(len(closes.columns))
    bench = {
        "spy_buy_hold": P.buy_and_hold(closes, {"SPY": 1.0}, cash, rebalance=None),
        "sixty_forty": P.buy_and_hold(closes, {"SPY": 0.6, "IEF": 0.4}, cash, "monthly"),
        "equal_weight_universe": P.buy_and_hold(closes, {c: 1 / len(closes.columns) for c in closes.columns}, cash, "monthly"),
        "vol_target_spy_only": P.backtest(opens, closes, cash, P.strategy_weights(
            opens[["SPY"]], closes[["SPY"]], cash, replace(cfg, universe=["SPY"], max_weight=1.0),
            vol=vols[winner][["SPY"]], signal=ones[["SPY"]],
            corr=P.rolling_corr(closes[["SPY"]], dates, cfg.corr_window)), cfg.cost_bps),
        "trend_no_vol_target": P.backtest(opens, closes, cash, eqw, cfg.cost_bps),
        "vol_target_no_trend": P.backtest(opens, closes, cash, w_vt, cfg.cost_bps),
    }
    series = {"strategy": runs["strategy"][0], "strategy_other_vol": runs["strategy_other_vol"][0], **bench}
    series = {k: v.loc[start:] for k, v in series.items()}

    res = {"config": {k: (v if not isinstance(v, list) else v) for k, v in cfg.__dict__.items()},
           "vol_engine": vc, "period": [str(start.date()), str(closes.index[-1].date())]}
    res["metrics"] = {k: P.metrics(v["ret"], cash, v["turnover"]) for k, v in series.items()}
    res["avg_gross_exposure"] = round(float(series["strategy"]["gross"].mean()), 3)
    res["sharpe_vs"] = {k: P.sharpe_diff_ci(series["strategy"]["ret"], series[k]["ret"], cash)
                        for k in ["spy_buy_hold", "sixty_forty", "equal_weight_universe", "vol_target_spy_only"]}
    res["subperiods"] = {f"{a}-{b}": {k: P.metrics(series[k]["ret"].loc[a:b], cash) for k in
                                       ["strategy", "spy_buy_hold", "sixty_forty"]} for a, b in SUBPERIODS}
    calendar = {k: ((1 + v["ret"]).groupby(v.index.year).prod() - 1).round(4) for k, v in series.items()
                if k in ("strategy", "spy_buy_hold", "sixty_forty")}
    res["calendar_years"] = pd.DataFrame(calendar).to_dict(orient="index")
    res["crisis_windows"] = {}
    for label, a, b in [("GFC 2007-10..2009-03", "2007-10-09", "2009-03-09"), ("COVID 2020-02..03", "2020-02-19", "2020-03-23"),
                        ("2022 rates", "2022-01-03", "2022-10-12")]:
        res["crisis_windows"][label] = {k: round(float((1 + series[k]["ret"].loc[a:b]).prod() - 1), 4)
                                        for k in ("strategy", "spy_buy_hold", "sixty_forty")}

    # 3. random picker
    held = signal.loc[runs["strategy"][1].index]
    p_long = float(held.stack().mean())
    g = dict(opens=opens, closes=closes, cash=cash, cfg=cfg, vol=vols[winner], signal=signal, corr=corr,
             p_long=p_long, start=start)
    with ProcessPoolExecutor(max_workers=args.procs, initializer=_init, initargs=(g,)) as ex:
        sims = list(ex.map(_random_run, range(args.random), chunksize=8))
    sh = np.array([s for s, _ in sims])
    mdd = np.array([d for _, d in sims])
    strat_sh = res["metrics"]["strategy"]["sharpe"]
    res["random_picker"] = {"sims": args.random, "p_long": round(p_long, 3),
                            "sharpe_mean": round(float(sh.mean()), 3),
                            "sharpe_p5_p95": [round(float(np.percentile(sh, 5)), 3), round(float(np.percentile(sh, 95)), 3)],
                            "max_dd_median": round(float(np.median(mdd)), 4),
                            "p_value_strategy_sharpe": round(float((sh >= strat_sh).mean()), 4)}
    res["seconds"] = round(time.time() - t0, 1)

    eq = pd.DataFrame({k: (1 + v["ret"]).cumprod() for k, v in series.items()})
    eq.to_csv(os.path.join(OUT, "equity.csv"))
    runs["strategy"][1].to_csv(os.path.join(OUT, "weights.csv"))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    cols = ["cagr", "vol", "sharpe", "max_drawdown", "calmar", "turnover_per_year"]
    print(pd.DataFrame(res["metrics"]).T[cols].to_string())
    print("sharpe vs:", json.dumps(res["sharpe_vs"]))
    print("random:", json.dumps(res["random_picker"]))


if __name__ == "__main__":
    main()
