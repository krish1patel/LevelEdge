#!/usr/bin/env python3
"""
Greedier books for a real cash account (no margin): how much more return, at what risk?

Leverage through a retail margin loan loses (about T-bill + 7.5% at Fidelity), but leveraged
ETFs borrow inside the fund at roughly swap rates. They are modelled from the underlying's
daily total return:  r_L = L x r - (L-1) x (T-bill + 0.5%)/252 - 0.9%/252  (expense ratio),
reset daily, so volatility drag is included. The model is checked against the real funds
(SSO, UPRO, QLD, TQQQ) where they exist.

Candidates (fixed before running; weekly decisions at the close, fills at the next open,
5 bp costs; "off" means T-bills):
  underlying  SPY or QQQ
  leverage    1x, 2x, 3x
  sell rule   none | sma200 (hold while the underlying closes above its 200-day average)
              | tsmom12 (hold while its 12-month return beats T-bills; the book's rule)
  plus references: the traded book (no margin), the book at 1.5x financed like a 2x ETF
  (T-bill + 1.4%), 60/40.

Read: 2007-2019 development, 2020+ holdout read once. A candidate is "credible" only if it
beats SPY on CAGR in BOTH periods and its Sharpe is not below SPY's in either. With 18
candidates tried, the best development result is optimistic; the holdout is the check.

Usage: python -I benchmarks/aggressive_research.py
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
OUT = os.path.join(ROOT, "benchmarks", "results", "aggressive")
SWAP, EXPENSE = 0.005, 0.009
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"
REAL = {("SPY", 2): "SSO", ("SPY", 3): "UPRO", ("QQQ", 2): "QLD", ("QQQ", 3): "TQQQ"}


def synthetic(o: pd.Series, c: pd.Series, cash: pd.Series, L: float) -> tuple[pd.Series, pd.Series]:
    """Daily-reset L-times fund: synthetic open and close series."""
    prev = c.shift(1)
    fin = ((L - 1) * (cash + SWAP / P.TRADING_DAYS) + EXPENSE / P.TRADING_DAYS) if L != 1 else 0.0
    r = (L * (c / prev - 1) - fin).fillna(0.0)
    close = (1 + r).cumprod() * 100
    gap = L * (o / prev - 1)
    open_ = close.shift(1) * (1 + gap)
    open_.iloc[0] = close.iloc[0]
    return open_, close


def main():
    os.makedirs(OUT, exist_ok=True)
    syms = ["SPY", "QQQ", "IEF", "SSO", "UPRO", "QLD", "TQQQ"] + list(P.UNIVERSE)
    prices = P.load_prices(sorted(set(syms)), start="2003-01-01", cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, ["SPY", "QQQ", "IEF"])
    daily_cash = cash.reindex(closes.index).fillna(0.0)
    res = {"model": {"swap_spread": SWAP, "expense": EXPENSE}}

    # ---- check the leveraged-ETF model against the real funds -----------
    chk = {}
    for (u, L), real in REAL.items():
        if real not in prices or prices[real].empty:
            continue
        rc = prices[real]["Close"]
        rc.index = pd.DatetimeIndex(rc.index).normalize()
        _, sc = synthetic(opens[u], closes[u], daily_cash, L)
        j = rc.index.intersection(sc.index)
        j = j[j >= j[0] + pd.Timedelta(days=5)]
        a, b = rc.loc[j], sc.loc[j]
        yrs = (j[-1] - j[0]).days / 365.25
        chk[real] = {"from": str(j[0].date()), "real_cagr": float((a.iloc[-1] / a.iloc[0]) ** (1 / yrs) - 1),
                     "model_cagr": float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1),
                     "daily_corr": float(a.pct_change().corr(b.pct_change()))}
    res["model_check"] = chk

    # ---- candidates ---------------------------------------------------------
    O, C = {}, {}
    for u in ("SPY", "QQQ"):
        for L in (1, 2, 3):
            O[f"{u}x{L}"], C[f"{u}x{L}"] = synthetic(opens[u], closes[u], daily_cash, L)
    SO, SC = pd.DataFrame(O), pd.DataFrame(C)
    dates = P.rebalance_dates(closes.index, "weekly")
    dates = dates[dates >= pd.Timestamp("2007-01-01")]
    sma = {u: closes[u] > closes[u].rolling(200).mean() for u in ("SPY", "QQQ")}
    tsm = P.momentum_signal(closes[["SPY", "QQQ"]], cash, 252)
    runs = {}
    for u in ("SPY", "QQQ"):
        for L in (1, 2, 3):
            k = f"{u}x{L}"
            for rule in ("none", "sma200", "tsmom12"):
                on = pd.Series(1.0, index=dates) if rule == "none" else \
                    (sma[u].reindex(dates).astype(float) if rule == "sma200" else tsm[u].reindex(dates).fillna(0.0))
                dec = pd.DataFrame(0.0, index=dates, columns=SC.columns)
                dec[k] = on.values
                runs[f"{k}_{rule}"] = P.backtest(SO, SC, cash, dec, 5.0, borrow_spread=0.0)["ret"]

    # references
    cfg = live.config()
    po, pc, pcash = P.panel(prices, cfg.universe)
    for name, lev, spread in (("book_1x", 1.0, 0.0), ("book_1p5x_etf_financed", 1.5, SWAP + EXPENSE)):
        live.LEVERAGE = lev
        book, _ = live.book_decisions(po, pc, pcash, cfg)
        runs[name] = P.backtest(po, pc, pcash, book, cfg.cost_bps, borrow_spread=spread)["ret"]
    live.LEVERAGE = 1.0
    runs["sixty_forty"] = P.buy_and_hold(pc, {"SPY": 0.6, "IEF": 0.4}, pcash, "monthly")["ret"]

    start = max(r.index[0] for r in runs.values())
    rows, table = {}, []
    for k, r in runs.items():
        r = r.loc[start:]
        f, d, h = P.metrics(r, cash), P.metrics(r.loc[:DEV_END], cash), P.metrics(r.loc[HOLD_START:], cash)
        yr = (1 + r).groupby(r.index.year).prod() - 1
        rows[k] = {"full": f, "dev": d, "holdout": h, "worst_year": float(yr.min()), "worst_year_at": int(yr.idxmin())}
        table.append({"name": k, "CAGR": f["cagr"], "maxDD": f["max_drawdown"], "Sharpe": f["sharpe"],
                      "dev CAGR": d["cagr"], "dev DD": d["max_drawdown"], "hold CAGR": h["cagr"], "hold DD": h["max_drawdown"],
                      "dev Sh": d["sharpe"], "hold Sh": h["sharpe"], "worst yr": float(yr.min())})
    spy = rows["SPYx1_none"]
    for k, v in rows.items():
        v["credible"] = bool(all(v[p]["cagr"] > spy[p]["cagr"] and v[p]["sharpe"] >= spy[p]["sharpe"] for p in ("dev", "holdout")))
    res["period"] = [str(start.date()), str(closes.index[-1].date())]
    res["runs"] = rows
    pick = max((k for k in rows if k.startswith(("SPY", "QQQ"))), key=lambda k: rows[k]["dev"]["cagr"])
    res["best_dev_cagr"] = pick
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)

    pd.set_option("display.width", 220)
    print("model check:", json.dumps(chk, indent=None, default=lambda x: round(x, 4)))
    t = pd.DataFrame(table).set_index("name").sort_values("dev CAGR", ascending=False)
    t["credible"] = [rows[k]["credible"] for k in t.index]
    print(t.round(3).to_string())
    print("period", res["period"], "best on development CAGR:", pick)


if __name__ == "__main__":
    main()
