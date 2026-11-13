#!/usr/bin/env python3
"""
Buy the dip: after a big drop, does buying and holding for 3 months pay?

Prompted by NVDA in Nov 2026 (-25% in a month, -32% from its October high).

Protocol (fixed before running)
-------------------------------
Events are read at a close; the buy fills at the next open; the position is held 63
trading days (about 3 months) and sold at that close. 10 bp cost each way. One event per
symbol per 63 days (no double counting of one slide).

A. Stocks: point-in-time S&P 500 members (fja05680/sp500 lists; Yahoo prices). A stock
   counts only on days it was a member.
     primary    fell_20_1m      21-day return <= -20%                       (NVDA now)
     secondary  off_30_high     close >= 30% below its 252-day high
                fell_12_1w      5-day return <= -12%
                z_minus2        21-day return <= -2 x its 1-year monthly volatility
                fallen_winner   fell_20_1m and the prior 11 months (t-252..t-21) up >= 30%
   Benchmark: the equal-weight average of all members bought at the same open and sold at
   the same close ("any S&P stock bought that day"), and SPY over the same window.
B. ETFs (sectors, countries, styles, bonds, commodities, ~40 funds):
     primary    fell_10_1m      21-day return <= -10%
     secondary  off_20_high, z_minus2
   Benchmark: equal-weight average of the ETFs available that day, and SPY.
C. Strategy form: SPY plus 5% slots in each live dip position (up to 20 slots), held 63
   days; compared with SPY.
D. Other baskets: the book's trend sleeve run on the broad ETF list instead of the 11
   ETFs (same rules), inside the book. One pre-declared alternative, not a search.

Pass rule: a dip rule "works" only if its mean 63-day return beats the same-day benchmark
with a 95% confidence interval above zero (block bootstrap over entry months) in BOTH
2006-2019 and 2020+. For D: book Sharpe not lower and worst fall not deeper in both.

Bias warning: Yahoo has no prices for ~300 former members (mostly failed or acquired),
so stocks that dipped and then died are missing. That flatters dip buying; a failing
result is robust to it, a passing one would not be.

Usage: python -I benchmarks/dip_research.py
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
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from leveledge import live  # noqa: E402
from leveledge import portfolio as P  # noqa: E402
from stock_factors import load_membership, yahoo_symbol  # noqa: E402

SP_CACHE = os.path.join(ROOT, "benchmarks", "cache", "sp500")
ETF_CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "dip")
H = 63
COST = 10 / 1e4
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"
BROAD_ETFS = ["SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "VGK", "EWJ", "EWZ", "EWG", "EWU", "EWA", "EWC",
              "EWY", "EWT", "EWH", "FXI", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH",
              "IBB", "XBI", "KRE", "ITB", "GDX", "IEF", "TLT", "LQD", "HYG", "TIP", "SHY", "GLD", "SLV", "DBC",
              "USO", "VNQ"]


def member_mask(membership: pd.Series, index: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    rows = {d: {yahoo_symbol(t) for t in m} for d, m in membership.items()}
    m = pd.DataFrame({d: pd.Series(True, index=sorted(s)) for d, s in rows.items()}).T
    m = m.reindex(columns=cols).fillna(False).astype(bool)
    return m.reindex(m.index.union(index)).ffill().reindex(index).fillna(False).astype(bool)


def signals(C: pd.DataFrame, kind: str) -> dict[str, pd.DataFrame]:
    r21 = C / C.shift(21) - 1
    r5 = C / C.shift(5) - 1
    hi = C.rolling(252, min_periods=200).max()
    vol_m = np.log(C).diff().rolling(252, min_periods=200).std() * np.sqrt(21)
    prior = C.shift(21) / C.shift(252) - 1
    if kind == "stock":
        return {"fell_20_1m": r21 <= -0.20, "off_30_high": C <= 0.70 * hi, "fell_12_1w": r5 <= -0.12,
                "z_minus2": r21 <= -2 * vol_m, "fallen_winner": (r21 <= -0.20) & (prior >= 0.30)}
    return {"fell_10_1m": r21 <= -0.10, "off_20_high": C <= 0.80 * hi, "z_minus2": r21 <= -2 * vol_m}


def events(sig: pd.DataFrame, fwd: pd.DataFrame, bench: pd.Series, spy_fwd: pd.Series,
           allowed: pd.DataFrame | None) -> pd.DataFrame:
    s = sig.fillna(False)
    if allowed is not None:
        s &= allowed
    s &= fwd.notna()
    idx = s.index
    out = []
    for col in s.columns:
        hits = np.flatnonzero(s[col].to_numpy())
        last = -10 ** 9
        for i in hits:
            if i - last < H:
                continue
            last = i
            d = idx[i]
            out.append({"symbol": col, "date": d, "ret": fwd.iat[i, fwd.columns.get_loc(col)],
                        "bench": bench.iat[i], "spy": spy_fwd.iat[i]})
    return pd.DataFrame(out)


def month_boot(x: pd.Series, months: pd.Series, n: int = 2000, seed: int = 0) -> list[float]:
    g = pd.DataFrame({"x": x.to_numpy(), "m": months.to_numpy()}).groupby("m")["x"]
    sums, cnts = g.sum().to_numpy(), g.count().to_numpy()
    rng = np.random.default_rng(seed)
    k = len(sums)
    draws = rng.integers(0, k, size=(n, k))
    est = sums[draws].sum(1) / cnts[draws].sum(1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def summarize(ev: pd.DataFrame) -> dict:
    if ev.empty:
        return {"n": 0}
    ex = ev["ret"] - ev["bench"]
    m = ev["date"].dt.to_period("M").astype(str)
    return {"n": int(len(ev)), "mean_ret": float(ev["ret"].mean()), "median_ret": float(ev["ret"].median()),
            "p_profit": float((ev["ret"] > 0).mean()), "p10_ret": float(ev["ret"].quantile(0.1)),
            "p90_ret": float(ev["ret"].quantile(0.9)),
            "bench_mean": float(ev["bench"].mean()), "bench_p_profit": float((ev["bench"] > 0).mean()),
            "mean_excess": float(ex.mean()), "excess_ci95": month_boot(ex, m),
            "p_beat_bench": float((ex > 0).mean()), "mean_excess_spy": float((ev["ret"] - ev["spy"]).mean())}


def split(ev: pd.DataFrame) -> dict:
    return {"all": summarize(ev), "dev": summarize(ev[ev.date <= DEV_END]), "holdout": summarize(ev[ev.date >= HOLD_START])}


def passes(r: dict) -> bool:
    return all(r[p]["n"] > 0 and r[p]["excess_ci95"][0] > 0 for p in ("dev", "holdout"))


def dip_book(ev: pd.DataFrame, O: pd.DataFrame, C: pd.DataFrame, spy_ret: pd.Series, slot: float = 0.05) -> pd.Series:
    """SPY plus a 5% slot in each live dip position (scaled down past 20); daily returns."""
    idx = C.index
    dr = C.pct_change()
    w = pd.DataFrame(0.0, index=idx, columns=sorted(ev.symbol.unique()))
    entry = pd.DataFrame(0.0, index=idx, columns=w.columns)
    for e in ev.itertuples():
        i = idx.get_loc(e.date)
        if i + H >= len(idx):
            continue
        w.iloc[i + 2:i + H + 1, w.columns.get_loc(e.symbol)] += 1.0      # held from the entry day's close onward
        entry.iat[i + 1, entry.columns.get_loc(e.symbol)] += 1.0
    n = w.sum(1)
    k = np.minimum(1.0, 1.0 / (slot * n.replace(0, np.nan))).fillna(1.0) * slot
    pos = (w.mul(k, axis=0) * dr.reindex(columns=w.columns).fillna(0.0)).sum(1)
    # entry day: open-to-close for new positions
    oc = (C / O - 1).reindex(columns=w.columns).fillna(0.0)
    ne = entry.sum(1)
    k_e = np.minimum(1.0, 1.0 / (slot * (n + ne).replace(0, np.nan))).fillna(1.0) * slot
    pos += (entry.mul(k_e, axis=0) * oc).sum(1)
    invested = w.mul(k, axis=0).sum(1) + entry.mul(k_e, axis=0).sum(1)
    cost = (entry.sum(1) * k_e + w.diff().clip(upper=0).abs().sum(1) * k) * COST
    return pos + (1 - invested) * spy_ret.reindex(idx).fillna(0.0) - cost


def part_d():
    syms = sorted(set(BROAD_ETFS) | set(P.UNIVERSE))
    prices = P.load_prices(syms, start="2003-01-01", cache_dir=ETF_CACHE)
    first = {s: df.index[0] for s, df in prices.items() if s != P.TBILL and len(df)}
    broad = [s for s in syms if s in first and first[s] <= pd.Timestamp("2006-01-31")]
    res = {}
    for name, uni in (("current_11", list(P.UNIVERSE)), ("broad", broad)):
        cfg = P.StrategyConfig(universe=uni)
        opens, closes, cash = P.panel(prices, uni)
        book, trend = live.book_decisions(opens, closes, cash, cfg)
        bt = P.backtest(opens, closes, cash, book, cfg.cost_bps)["ret"]
        sl = P.backtest(opens, closes, cash, trend, cfg.cost_bps)["ret"]
        res[name] = {"universe": uni, "book": {p: P.metrics(r, cash) for p, r in
                                                (("full", bt), ("dev", bt.loc[:DEV_END]), ("holdout", bt.loc[HOLD_START:]))},
                     "sleeve": {p: P.metrics(r, cash) for p, r in
                                (("full", sl), ("dev", sl.loc[:DEV_END]), ("holdout", sl.loc[HOLD_START:]))}}
    a, b = res["current_11"]["book"], res["broad"]["book"]
    res["adopt_broad"] = bool(all(b[p]["sharpe"] >= a[p]["sharpe"] and b[p]["max_drawdown"] >= a[p]["max_drawdown"]
                                  for p in ("dev", "holdout")))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"protocol": {"hold_days": H, "cost_each_way_bp": 10}}

    # ---- A: stocks -------------------------------------------------------
    C = pd.read_parquet(os.path.join(SP_CACHE, "close.parquet"))
    O = pd.read_parquet(os.path.join(SP_CACHE, "open.parquet")).reindex(C.index)
    C = C.loc["2005-01-01":]
    O = O.loc[C.index]
    mem = member_mask(load_membership(), C.index, list(C.columns))
    fwd = (C.shift(-H) / O.shift(-1)) * (1 - COST) ** 2 - 1
    ew = fwd.where(mem).mean(1)
    spy = P.load_prices(["SPY"], start="2004-01-01", cache_dir=ETF_CACHE)["SPY"].reindex(C.index)
    spy_fwd = spy["Close"].shift(-H) / spy["Open"].shift(-1) - 1
    sig = signals(C, "stock")
    stock_ev = {}
    res["stocks"] = {}
    for k, s in sig.items():
        ev = events(s.loc["2006-01-01":], fwd.loc["2006-01-01":], ew.loc["2006-01-01":], spy_fwd.loc["2006-01-01":],
                    mem.loc["2006-01-01":])
        stock_ev[k] = ev
        res["stocks"][k] = split(ev) | {"passes": passes(split(ev))}
        ev.to_csv(os.path.join(OUT, f"stock_events_{k}.csv"), index=False)
    # what any member did over 3 months, all days (the base rate)
    allf = fwd.where(mem).loc["2006-01-01":].stack().dropna()
    res["stocks"]["any_member_any_day"] = {"mean_ret": float(allf.mean()), "p_profit": float((allf > 0).mean()),
                                           "p10_ret": float(allf.quantile(0.1))}
    res["stocks_coverage_note"] = "members without Yahoo prices are absent (survivorship bias in favour of dip buying)"

    # ---- C: strategy form on the primary stock rule ---------------------
    spy_ret = spy["Close"].pct_change()
    book = dip_book(stock_ev["fell_20_1m"], O, C, spy_ret).loc["2006-03-01":]
    _, _, tb = P.panel(P.load_prices(["SPY"], start="2004-01-01", cache_dir=ETF_CACHE), ["SPY"])
    sr = spy_ret.reindex(book.index).fillna(0.0)
    res["strategy_spy_plus_dips"] = {
        p: {"dips": P.metrics(book.loc[a:b], tb), "spy": P.metrics(sr.loc[a:b], tb)}
        for p, a, b in (("full", None, None), ("dev", None, DEV_END), ("holdout", HOLD_START, None))}

    # ---- B: ETFs -----------------------------------------------------------
    prices = P.load_prices(BROAD_ETFS, start="2003-01-01", cache_dir=ETF_CACHE)
    have = [s for s in BROAD_ETFS if s in prices and len(prices[s])]
    EO, EC, _ = P.panel(prices, have)
    EC = EC.loc["2005-01-01":]
    EO = EO.loc[EC.index]
    efwd = (EC.shift(-H) / EO.shift(-1)) * (1 - COST) ** 2 - 1
    eew = efwd.mean(1)
    espy = spy_fwd.reindex(EC.index)
    res["etfs"] = {"universe": have}
    for k, s in signals(EC, "etf").items():
        ev = events(s.loc["2006-01-01":], efwd.loc["2006-01-01":], eew.loc["2006-01-01":], espy.loc["2006-01-01":], None)
        res["etfs"][k] = split(ev) | {"passes": passes(split(ev))}
        ev.to_csv(os.path.join(OUT, f"etf_events_{k}.csv"), index=False)

    # ---- D: the trend sleeve on a broader basket -------------------------
    res["broad_basket"] = part_d()

    # ---- today's situation -------------------------------------------------
    last = C.index[-1]
    today = {k: sorted(s.columns[s.loc[last].fillna(False) & mem.loc[last]]) for k, s in sig.items()}
    res["today"] = {"as_of": str(last.date()), "members_flagged": today}

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)

    # ---- print -------------------------------------------------------------
    def row(name, r):
        a = r["all"]
        if not a.get("n"):
            return f"{name:16s} no events"
        d, h = r["dev"], r["holdout"]
        return (f"{name:16s} n={a['n']:5d} mean {a['mean_ret']:+.1%} median {a['median_ret']:+.1%} "
                f"P(up) {a['p_profit']:.0%} p10 {a['p10_ret']:+.1%} | same-day avg {a['bench_mean']:+.1%} P(up) {a['bench_p_profit']:.0%}"
                f" | excess {a['mean_excess']:+.2%} beat {a['p_beat_bench']:.0%} "
                f"dev {d.get('mean_excess', float('nan')):+.2%} {np.round(d.get('excess_ci95', [np.nan] * 2), 3)} "
                f"hold {h.get('mean_excess', float('nan')):+.2%} {np.round(h.get('excess_ci95', [np.nan] * 2), 3)} "
                f"pass={r['passes']}")
    print("A. S&P 500 members, buy next open, hold 63 days")
    for k in sig:
        print(row(k, res["stocks"][k]))
    am = res["stocks"]["any_member_any_day"]
    print(f"   any member, any day: mean {am['mean_ret']:+.1%}, P(up) {am['p_profit']:.0%}, p10 {am['p10_ret']:+.1%}")
    print("C. SPY + dip slots vs SPY")
    for p, v in res["strategy_spy_plus_dips"].items():
        print(f"   {p:8s} dips CAGR {v['dips']['cagr']:+.1%} Sharpe {v['dips']['sharpe']:.2f} DD {v['dips']['max_drawdown']:.0%}"
              f" | SPY CAGR {v['spy']['cagr']:+.1%} Sharpe {v['spy']['sharpe']:.2f} DD {v['spy']['max_drawdown']:.0%}")
    print(f"B. ETFs ({len(have)})")
    for k in ("fell_10_1m", "off_20_high", "z_minus2"):
        print(row(k, res["etfs"][k]))
    print("D. trend sleeve on a broader basket (book)")
    bb = res["broad_basket"]
    for name in ("current_11", "broad"):
        b = bb[name]["book"]
        print(f"   {name:10s} ({len(bb[name]['universe'])} ETFs) CAGR {b['full']['cagr']:.1%} Sharpe {b['full']['sharpe']:.2f} "
              f"DD {b['full']['max_drawdown']:.0%} | dev {b['dev']['sharpe']:.2f} hold {b['holdout']['sharpe']:.2f} "
              f"hold DD {b['holdout']['max_drawdown']:.0%}")
    print(f"   adopt broad: {bb['adopt_broad']}")
    print("today:", {k: len(v) for k, v in today.items()}, "NVDA in:", [k for k, v in today.items() if "NVDA" in v])


if __name__ == "__main__":
    main()
