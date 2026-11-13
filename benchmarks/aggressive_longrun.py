#!/usr/bin/env python3
"""
Long-history check of the aggressive candidates (benchmarks/aggressive_research.py), whose
2007-2026 window favours big tech. Same leveraged-ETF model on index data: Nasdaq-100 from
1985 and S&P 500 from 1960 (price indexes: dividends are missing, so returns are understated
by the yield, roughly 1-3%/yr for the S&P and under 1% for the Nasdaq-100). Weekly decisions
at Friday's close, applied from the next session; 5 bp per switch.

Usage: python -I benchmarks/aggressive_longrun.py
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
from leveledge import portfolio as P  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "longrun")
OUT = os.path.join(ROOT, "benchmarks", "results", "aggressive")
SWAP, EXPENSE = 0.005, 0.009
WINDOWS = [("Dot-com crash", "2000-03-24", "2002-10-09"), ("2008 crash", "2007-10-09", "2009-03-09"),
           ("1987 crash", "1987-08-25", "1987-12-04"), ("1973-74 bear", "1973-01-11", "1974-10-03"),
           ("2020 Covid", "2020-02-19", "2020-03-23"), ("2022", "2022-01-03", "2022-10-12")]


def run(c: pd.Series, cash: pd.Series, L: int, rule: str) -> pd.Series:
    r = c.pct_change().fillna(0.0)
    fin = (L - 1) * (cash + SWAP / 252) + (EXPENSE / 252 if L > 1 else 0.0)
    rl = L * r - fin
    if rule == "none":
        on = pd.Series(1.0, index=c.index)
    else:
        sig = (c > c.rolling(200).mean()) if rule == "sma200" else (c / c.shift(252) - 1 > cash.rolling(252).sum())
        fri = c.index.to_series().dt.dayofweek.ge(c.index.to_series().shift(-1).dt.dayofweek.fillna(0).astype(int))
        on = sig.where(fri).ffill().fillna(0.0).astype(float)
    pos = on.shift(1).fillna(0.0)
    switch = pos.diff().abs().fillna(0.0)
    return pos * rl + (1 - pos) * cash - switch * 5e-4


def stats(r: pd.Series) -> dict:
    g = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"cagr": float(g.iloc[-1] ** (1 / yrs) - 1), "max_dd": float((g / g.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)), "growth_of_1": float(g.iloc[-1])}


def main():
    px = P.load_prices(["^NDX", "^GSPC"], start="1960-01-01", cache_dir=CACHE)
    cash = (px[P.TBILL]["Close"] / 100 / 252)
    out, lines = {}, []
    for idx, start in (("^NDX", "1985-10-01"), ("^GSPC", "1960-01-01")):
        c = px[idx]["Close"].loc[start:]
        cs = cash.reindex(c.index).ffill().fillna(0.0)
        for L in (1, 2, 3):
            for rule in ("none", "sma200", "tsmom12"):
                r = run(c, cs, L, rule).iloc[252:]
                k = f"{idx.strip('^')}x{L}_{rule}"
                s = stats(r)
                s["windows"] = {n: float((1 + r.loc[a:b]).prod() - 1) for n, a, b in WINDOWS if pd.Timestamp(a) >= r.index[0]}
                s["by_decade"] = {f"{d}s": stats(r[(r.index.year >= d) & (r.index.year < d + 10)])["cagr"]
                                  for d in range(1960, 2030, 10) if ((r.index.year >= d) & (r.index.year < d + 10)).sum() > 200}
                out[k] = s
                lines.append((k, s))
    with open(os.path.join(OUT, "longrun.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, s in lines:
        w = " ".join(f"{n.split()[0]} {v:+.0%}" for n, v in s["windows"].items())
        dec = " ".join(f"{d} {v:+.0%}" for d, v in s["by_decade"].items())
        print(f"{k:20s} CAGR {s['cagr']:+.1%} DD {s['max_dd']:.0%} Sh {s['sharpe']:.2f} | {w} | {dec}")


if __name__ == "__main__":
    main()
