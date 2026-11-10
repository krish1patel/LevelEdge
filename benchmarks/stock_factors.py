#!/usr/bin/env python3
"""
Stock-level factors on historical S&P 500 members (survivorship-aware).

Data
----
* Membership: fja05680/sp500 "S&P 500 Historical Components & Changes (Updated).csv"
  (point-in-time member lists). At each month end the strategy may only hold
  stocks that were members on that date.
* Prices: Yahoo Finance, total-return adjusted. Yahoo keeps only currently
  listed tickers, so failed, acquired and renamed members are missing. The
  coverage gap is measured and reported, and every factor is compared with an
  equal-weight portfolio of the same covered members, which carries the same gap.

Protocol (fixed before running)
-------------------------------
Monthly decisions at the last close of each month, fills at the next open,
10 bp cost per unit of turnover, long-only, equal weight within the top 20% of
covered members by the factor.
  primary   mom_12_1   12-month return skipping the last month
            mom_lowvol average of cross-sectional z-scores of mom_12_1 and -vol_1y
  secondary lowvol     lowest 1-year daily volatility
            rev_1m     worst last-month return (short-term reversal)
Development 2006-2019, holdout 2020+ (read once). Comparisons: excess return
over the equal-weight member portfolio (information ratio with block-bootstrap
CI), Sharpe vs SPY.

Usage: python -I benchmarks/stock_factors.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.request
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge import portfolio as P  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "sp500")
OUT = os.path.join(ROOT, "benchmarks", "results", "stock_factors")
MEMBERS_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
START = "2004-01-01"
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"
TOP = 0.2
COST_BPS = 10.0


def load_membership() -> pd.Series:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "members.csv")
    if not os.path.exists(path):
        req = urllib.request.Request(MEMBERS_URL, headers={"User-Agent": "leveledge-research"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    rows = {}
    with open(path) as f:
        rd = csv.reader(f)
        header = next(rd)
        if header[:2] != ["date", "tickers"]:
            raise ValueError(f"unexpected membership header {header}")
        for d, t in rd:
            rows[pd.Timestamp(d)] = frozenset(x.strip() for x in t.split(",") if x.strip())
    return pd.Series(rows).sort_index()


def yahoo_symbol(t: str) -> str:
    return t.replace(".", "-")


def retry_missing(O: pd.DataFrame, C: pd.DataFrame, tickers: list[str], cp: str, op: str):
    """Bulk downloads get rate-limited; fetch absent symbols one by one, slowly, and record what's truly gone."""
    import yfinance as yf

    miss_path = os.path.join(CACHE, "missing.json")
    known_gone = set(json.load(open(miss_path))) if os.path.exists(miss_path) else set()
    todo = [s for s in sorted({yahoo_symbol(t) for t in tickers}) if s not in C.columns and s not in known_gone]
    print(f"  retrying {len(todo)} symbols individually", flush=True)
    got_c, got_o, gone = {}, {}, set(known_gone)
    for i, s in enumerate(todo):
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = yf.Ticker(s).history(start=START, interval="1d", auto_adjust=True)
                break
            except Exception:  # noqa: BLE001 - network hiccup: back off and retry
                time.sleep(2 + 3 * attempt)
        time.sleep(0.4)
        if df.empty or len(df) < 20:
            gone.add(s)
            continue
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        got_c[s], got_o[s] = df["Close"], df["Open"]
        if (i + 1) % 50 == 0:
            print(f"  retry {i + 1}/{len(todo)}: recovered {len(got_c)}", flush=True)
    if got_c:
        C = pd.concat([C, pd.DataFrame(got_c)], axis=1).sort_index()
        O = pd.concat([O, pd.DataFrame(got_o)], axis=1).reindex(C.index)
        C.to_parquet(cp)
        O.to_parquet(op)
    with open(miss_path, "w") as f:
        json.dump(sorted(gone), f)
    print(f"  recovered {len(got_c)}; unavailable on Yahoo: {len(gone)}", flush=True)
    return O, C


def load_prices(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cp, op = os.path.join(CACHE, "close.parquet"), os.path.join(CACHE, "open.parquet")
    if os.path.exists(cp):
        return retry_missing(pd.read_parquet(op), pd.read_parquet(cp), tickers, cp, op)
    import yfinance as yf

    closes, opens = [], []
    syms = sorted({yahoo_symbol(t) for t in tickers})
    for i in range(0, len(syms), 80):
        chunk = syms[i:i + 80]
        df = yf.download(chunk, start=START, interval="1d", auto_adjust=True, group_by="column",
                         threads=True, progress=False)
        if df.empty:
            continue
        closes.append(df["Close"])
        opens.append(df["Open"])
        print(f"  prices {min(i + 80, len(syms))}/{len(syms)}", flush=True)
    C = pd.concat(closes, axis=1)
    O = pd.concat(opens, axis=1)
    C.index = pd.DatetimeIndex(C.index).tz_localize(None).normalize()
    O.index = C.index
    C = C.loc[:, C.notna().sum() > 20]
    O = O[C.columns]
    C.to_parquet(cp)
    O.to_parquet(op)
    return retry_missing(O, C, tickers, cp, op)


def members_on(membership: pd.Series, d: pd.Timestamp) -> frozenset:
    i = membership.index.searchsorted(d, side="right") - 1
    return membership.iloc[max(i, 0)]


def zs(x: pd.Series) -> pd.Series:
    x = x.clip(x.quantile(0.01), x.quantile(0.99))
    return (x - x.mean()) / x.std(ddof=0)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    membership = load_membership()
    since = membership[membership.index >= "2005-01-01"]
    tickers = sorted(set().union(*since.tolist()))
    print(f"membership rows {len(membership)}, tickers since 2005: {len(tickers)}", flush=True)
    O, C = load_prices(tickers)
    spy = P.load_prices(["SPY"], start=START, cache_dir=os.path.join(ROOT, "benchmarks", "cache", "etf"))
    idx = C.index[(C.index >= "2005-01-01")]
    C, O = C.loc[idx], O.loc[idx]
    cash = (spy[P.TBILL]["Close"].reindex(idx).ffill() / 100 / P.TRADING_DAYS).fillna(0.0)
    rets = C.pct_change(fill_method=None)
    vol1y = rets.rolling(252, min_periods=200).std()
    mom = C.shift(21) / C.shift(252) - 1
    rev = -(C / C.shift(21) - 1)

    month_ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    month_ends = pd.DatetimeIndex(month_ends[month_ends >= pd.Timestamp("2006-01-01")].to_numpy())

    coverage, dec = [], {k: {} for k in ("mom_12_1", "mom_lowvol", "lowvol", "rev_1m", "ew_members")}
    for d in month_ends:
        mem = {yahoo_symbol(t) for t in members_on(membership, d)}
        have = [t for t in mem if t in C.columns and pd.notna(C.at[d, t]) and pd.notna(mom.at[d, t]) and pd.notna(vol1y.at[d, t])]
        coverage.append({"date": d, "members": len(mem), "covered": len(have)})
        if len(have) < 100:
            continue
        m, v, r = mom.loc[d, have], vol1y.loc[d, have], rev.loc[d, have]
        k = max(1, int(len(have) * TOP))
        scores = {"mom_12_1": m, "lowvol": -v, "rev_1m": r, "mom_lowvol": zs(m) + zs(-v)}
        for name, sc in scores.items():
            top = sc.sort_values(ascending=False).index[:k]
            dec[name][d] = pd.Series(1.0 / k, index=top)
        dec["ew_members"][d] = pd.Series(1.0 / len(have), index=have)
    cov = pd.DataFrame(coverage).set_index("date")
    cov["share"] = cov["covered"] / cov["members"]

    Cf, Of = C.ffill(), O.fillna(C).ffill()            # a stock with no print that day keeps its last price
    Cf, Of = Cf.fillna(0.0), Of.fillna(0.0)
    runs = {}
    for name, rows in dec.items():
        D = pd.DataFrame(rows).T.reindex(columns=C.columns).fillna(0.0)
        valid = Cf.columns[(D != 0).any()]
        bt = P.backtest(Of[valid].replace(0, np.nan).ffill().bfill(), Cf[valid].replace(0, np.nan).ffill().bfill(),
                        cash, D[valid], COST_BPS, 0.0)
        runs[name] = bt
    spy_r = P.buy_and_hold(spy["SPY"][["Close"]].rename(columns={"Close": "SPY"}).reindex(idx).ffill(), {"SPY": 1.0}, cash, None)
    start = runs["ew_members"].index[0]
    runs["spy_buy_hold"] = spy_r.loc[start:]
    runs = {k: v.loc[start:] for k, v in runs.items()}

    def excess_ir(a, b):
        ex = (a - b).dropna()
        ir = ex.mean() / ex.std(ddof=1) * np.sqrt(P.TRADING_DAYS)
        rng = np.random.default_rng(0)
        T, blk = len(ex), 21
        vals = []
        x = ex.to_numpy()
        for _ in range(2000):
            st = rng.integers(0, T - blk, size=int(np.ceil(T / blk)))
            ii = (st[:, None] + np.arange(blk)).ravel()[:T]
            s = x[ii]
            vals.append(s.mean() / s.std(ddof=1) * np.sqrt(P.TRADING_DAYS))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"excess_cagr": round(float((1 + ex).prod() ** (P.TRADING_DAYS / T) - 1), 4),
                "tracking_error": round(float(ex.std(ddof=1) * np.sqrt(P.TRADING_DAYS)), 4),
                "info_ratio": round(float(ir), 3), "ir_ci95": [round(float(lo), 3), round(float(hi), 3)]}

    res = {"period": [str(start.date()), str(idx[-1].date())], "members_last_update": str(membership.index[-1].date()),
           "coverage": {"mean_share": round(float(cov["share"].mean()), 3), "min_share": round(float(cov["share"].min()), 3),
                        "share_2006": round(float(cov.loc[:"2006-12-31", "share"].mean()), 3),
                        "share_2019": round(float(cov.loc["2019", "share"].mean()), 3),
                        "share_2025": round(float(cov.loc["2025", "share"].mean()), 3)}}
    for part, sl in (("full", slice(None)), ("dev", slice(None, DEV_END)), ("holdout", slice(HOLD_START, None))):
        res[part] = {k: P.metrics(v["ret"].loc[sl], cash, v["turnover"].loc[sl]) for k, v in runs.items()}
        res[f"{part}_vs_ew"] = {k: excess_ir(runs[k]["ret"].loc[sl], runs["ew_members"]["ret"].loc[sl])
                                for k in ("mom_12_1", "mom_lowvol", "lowvol", "rev_1m")}
        res[f"{part}_sharpe_vs_spy"] = {k: P.sharpe_diff_ci(runs[k]["ret"].loc[sl], runs["spy_buy_hold"]["ret"].loc[sl], cash)
                                        for k in ("mom_12_1", "mom_lowvol", "lowvol", "ew_members")}
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    pd.DataFrame({k: (1 + v["ret"]).cumprod() for k, v in runs.items()}).to_csv(os.path.join(OUT, "equity.csv"))
    cov.to_csv(os.path.join(OUT, "coverage.csv"))
    cols = ["cagr", "vol", "sharpe", "max_drawdown", "turnover_per_year"]
    print("coverage:", res["coverage"])
    for part in ("full", "dev", "holdout"):
        print(f"--- {part}"); print(pd.DataFrame(res[part]).T[cols].to_string())
        print("vs EW:", json.dumps(res[f"{part}_vs_ew"]))


if __name__ == "__main__":
    main()
