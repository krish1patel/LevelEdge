#!/usr/bin/env python3
"""
Published strategies, implemented as their papers specify, under one protocol.

Strategies (parameters from the papers; nothing tuned here)
-----------------------------------------------------------
mtp_sleeve   Goulding, Harvey & Mazzoleni (JFE 2023) intermediate-speed momentum:
             per ETF, signal = 0.5 * [12-month excess return > 0] + 0.5 * [1-month > 0],
             i.e. full in Bull, half in Correction or Rebound, out in Bear. Used as
             the trend sleeve with the book's existing sizing.
daa_g12      Keller & Keuning (2018) Defensive Asset Allocation, G12 universe:
             13612W momentum; canary VWO and BND, breadth B = 2; top 6 risky assets
             equal weight; cash share = (number of canaries with momentum <= 0) / 2,
             invested in the best of SHY / IEF / LQD. Monthly.
gtaa5        Faber (2007): SPY, EFA, IEF, GSG, VNQ at 20% each, held only when the
             month-end price is above its 10-month average, else cash. Monthly.
tom          McConnell & Xu (FAJ 2008) turn of the month: SPY from the last trading
             day of the month through the third trading day of the next, else cash.
overnight    Lou, Polk & Skouras (JFE 2019): SPY bought at the close and sold at the
             next open every day, at 0, 1 and 2 bp per side.

Protocol: development 2008-2019, holdout 2020+ (all of these are new to this
project, so the holdout is unseen for them). Next-open fills and 5 bp costs except
where stated. Sleeves are also tested inside the live book design:
1.5 x (0.7 x 60/40 + 0.3 x sleeve), against the current book. Selection among the
sleeves uses development Sharpe only. A deflated Sharpe corrects for the number of
strategies tried.

Rebalancing bands (cost per trade): the current book rebalanced weekly only when an
ETF drifts more than 0 / 1 / 2 / 3 / 5 percentage points from target. Pre-declared
choice: the largest band whose tracking error against the band-0 book stays below
0.5% a year.

Usage: python benchmarks/literature_research.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge import live  # noqa: E402
from leveledge import portfolio as P  # noqa: E402
from leveledge import strategies as S  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "literature")
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"
G12 = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD"]
CANARY = ["VWO", "BND"]
CASH_U = ["SHY", "IEF", "LQD"]
GTAA = ["SPY", "EFA", "IEF", "GSG", "VNQ"]
ALL = sorted(set(G12 + CANARY + CASH_U + GTAA + list(P.UNIVERSE)))


def month_ends(idx):
    return P.rebalance_dates(idx, "monthly")


def mom_13612w(closes: pd.DataFrame, me: pd.DatetimeIndex) -> pd.DataFrame:
    m = closes.loc[me]
    r = lambda k: m / m.shift(k) - 1
    return (12 * r(1) + 4 * r(3) + 2 * r(6) + r(12)) / 4


def daa_decisions(closes):
    return S.daa_g12(closes)


def gtaa_decisions(closes):
    return S.gtaa5(closes)


def tom_decisions(closes):
    """Weight 1 in SPY for the days -1..+3 around each month turn (decided the close before)."""
    idx = closes.index
    per = idx.to_period("M")
    pos_in_month = pd.Series(range(len(idx)), index=idx).groupby(per).cumcount().to_numpy()
    days_left = pd.Series(range(len(idx)), index=idx).groupby(per).cumcount(ascending=False).to_numpy()
    in_win = (days_left == 0) | (pos_in_month <= 2)           # last day, then days 1-3
    want = pd.Series(in_win, index=idx).shift(-1, fill_value=False)   # decide at the prior close
    change = want != want.shift(1)
    dates = idx[change.to_numpy()]
    w = pd.DataFrame(0.0, index=dates, columns=closes.columns)
    w.loc[want[change].index[want[change].to_numpy()], "SPY"] = 1.0
    return w


def overnight_returns(opens, closes, cash, bp):
    r = opens["SPY"] / closes["SPY"].shift(1) - 1 - 2 * bp / 1e4   # close -> next open, both sides
    # the day session sits in cash
    return (r.fillna(0.0) + 0.0 * cash).rename("overnight")


def book_with_sleeve(sleeve: pd.DataFrame, cols) -> pd.DataFrame:
    core = pd.Series(live.CORE, dtype=float).reindex(cols).fillna(0.0)
    s = sleeve.reindex(columns=cols).fillna(0.0)
    return live.LEVERAGE * (live.CORE_SHARE * core.values + live.TREND_SHARE * s)


def banded_book(opens, closes, cash, targets: pd.DataFrame, band: float, cost_bps=5.0, spread=0.02):
    """Weekly check; trade an ETF back to target only if it drifted more than ``band`` (abs weight)."""
    idx = closes.index
    C = closes.to_numpy(float)
    on = np.vstack([np.ones((1, C.shape[1])), C[1:] / C[:-1]])
    cr = cash.reindex(idx).fillna(0).to_numpy()
    T = targets.reindex(columns=closes.columns).fillna(0.0)
    tpos = {idx.get_loc(d): T.loc[d].to_numpy() for d in T.index if d in idx}
    first = min(tpos)
    w = np.zeros(C.shape[1])
    rets, turns, orders = [], [], 0
    for t in range(first, len(idx)):
        r = w @ (on[t] - 1) + (1 - w.sum()) * cr[t] - max(0.0, w.sum() - 1) * spread / P.TRADING_DAYS
        w = w * on[t] / (1 + r)
        turnover = 0.0
        if t in tpos:
            tgt = tpos[t]
            dev = np.abs(w - tgt)
            hit = (dev > band) | ((tgt == 0) & (w > 0)) if band > 0 else dev > 1e-9
            if hit.any():
                turnover = float(dev[hit].sum())
                orders += int(hit.sum())
                w = np.where(hit, tgt, w)
                r -= turnover * cost_bps / 1e4
        rets.append(r)
        turns.append(turnover)
    out = pd.DataFrame({"ret": rets, "turnover": turns}, index=idx[first:])
    out.attrs["orders"] = orders
    return out


def m(series, cash, sl=slice(None)):
    return P.metrics(series.loc[sl], cash)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    prices = P.load_prices(ALL, start="2003-01-01", cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, ALL)
    cfg = live.config()
    vol = np.log(closes).diff().apply(P.ewma_vol)

    # --- standalone strategies -------------------------------------------------
    runs = {
        "daa_g12": P.backtest(opens, closes, cash, daa_decisions(closes), 5.0, 0.0)["ret"],
        "gtaa5": P.backtest(opens, closes, cash, gtaa_decisions(closes), 5.0, 0.0)["ret"],
        "tom": P.backtest(opens, closes, cash, tom_decisions(closes), 5.0, 0.0)["ret"],
        "sixty_forty": P.buy_and_hold(closes, {"SPY": 0.6, "IEF": 0.4}, cash, "monthly")["ret"],
        "spy_buy_hold": P.buy_and_hold(closes, {"SPY": 1.0}, cash, None)["ret"],
    }
    for bp in (0, 1, 2):
        runs[f"overnight_{bp}bp"] = overnight_returns(opens, closes, cash, bp)

    # --- trend sleeves inside the live book design ---------------------------
    tu = list(P.UNIVERSE)
    dates_w = P.rebalance_dates(closes.index, "weekly")
    corr = P.rolling_corr(closes[tu], dates_w, cfg.corr_window)
    sig12 = P.momentum_signal(closes[tu], cash, 252)
    sig1 = P.momentum_signal(closes[tu], cash, 21)
    trend_orig = P.strategy_weights(opens[tu], closes[tu], cash, cfg, vol=vol[tu], signal=sig12, corr=corr)
    trend_mtp = P.strategy_weights(opens[tu], closes[tu], cash, cfg, vol=vol[tu], signal=0.5 * sig12 + 0.5 * sig1, corr=corr)
    runs["mtp_sleeve"] = P.backtest(opens, closes, cash, trend_mtp.reindex(columns=closes.columns).fillna(0), 5.0, 0.0)["ret"]
    runs["trend_sleeve_orig"] = P.backtest(opens, closes, cash, trend_orig.reindex(columns=closes.columns).fillna(0), 5.0, 0.0)["ret"]
    sleeves = {"orig_trend": trend_orig, "mtp": trend_mtp, "daa_g12": daa_decisions(closes), "gtaa5": gtaa_decisions(closes)}
    books = {}
    for k, s in sleeves.items():
        dec = book_with_sleeve(s, closes.columns)
        books[f"book_{k}"] = P.backtest(opens, closes, cash, dec, 5.0, 0.02)["ret"]

    start = max(r.dropna().index[0] for r in list(runs.values()) + list(books.values()))
    start = max(start, pd.Timestamp("2008-06-01"))
    allr = {k: v.loc[start:] for k, v in {**runs, **books}.items()}

    res = {"period": [str(start.date()), str(closes.index[-1].date())]}
    for part, sl in (("full", slice(None)), ("dev", slice(None, DEV_END)), ("holdout", slice(HOLD_START, None))):
        res[part] = {k: m(v, cash, sl) for k, v in allr.items()}

    # in-market return per day: the "profit per trade" view of the timing strategies
    tw = tom_decisions(closes)["SPY"].reindex(closes.index).ffill().fillna(0).loc[start:]
    res["tom_days_in_market_share"] = round(float(tw.mean()), 3)
    res["per_day_in_market_bp"] = {
        "tom_window": round(float(allr["tom"][tw.shift(1).fillna(0).astype(bool)].mean() * 1e4), 2),
        "spy_all_days": round(float(allr["spy_buy_hold"].mean() * 1e4), 2),
        "overnight_0bp": round(float(allr["overnight_0bp"].mean() * 1e4), 2),
        "overnight_1bp": round(float(allr["overnight_1bp"].mean() * 1e4), 2),
    }

    # selection among sleeves (dev Sharpe), then one holdout read
    cand = ["book_mtp", "book_daa_g12", "book_gtaa5"]
    sel = max(cand, key=lambda k: res["dev"][k]["sharpe"])
    res["selected_sleeve"] = sel
    res["holdout_vs_current_book"] = P.sharpe_diff_ci(allr[sel].loc[HOLD_START:], allr["book_orig_trend"].loc[HOLD_START:], cash)
    res["full_vs_current_book"] = P.sharpe_diff_ci(allr[sel], allr["book_orig_trend"], cash)
    tried = ["daa_g12", "gtaa5", "tom", "overnight_1bp", "mtp_sleeve", "book_mtp", "book_daa_g12", "book_gtaa5"]
    trial_sr = np.array([(allr[k].loc[:DEV_END] - cash.reindex(allr[k].loc[:DEV_END].index).fillna(0)).pipe(
        lambda x: x.mean() / x.std(ddof=1)) for k in tried])
    ex = allr[sel].loc[:DEV_END] - cash.reindex(allr[sel].loc[:DEV_END].index).fillna(0)
    res["deflated_sharpe_selected"] = P.deflated_sharpe(ex, trial_sr)
    res["crises"] = [{"label": l, **{k: round(float((1 + allr[k].loc[a:b]).prod() - 1), 4) for k in
                                     ("book_orig_trend", "book_mtp", "book_daa_g12", "book_gtaa5", "sixty_forty", "spy_buy_hold")}}
                     for l, a, b in [("2008 crash", "2008-06-01", "2009-03-09"), ("2020 Covid crash", "2020-02-19", "2020-03-23"),
                                     ("2020 rebound", "2020-03-23", "2020-08-31"), ("2022 rate shock", "2022-01-03", "2022-10-12"),
                                     ("2022-23 rebound", "2022-10-12", "2023-07-31")]]

    # --- rebalancing bands on the current book -------------------------------
    core_book, _ = live.book_decisions(opens[tu], closes[tu], cash, cfg, vol=vol[tu], signal=sig12, corr=corr)
    bands = {}
    base = None
    for b in (0.0, 0.01, 0.02, 0.03, 0.05):
        bb = banded_book(opens[tu], closes[tu], cash, core_book, b)
        r = bb["ret"].loc[start:]
        base = r if b == 0 else base
        te = float((r - base).std() * np.sqrt(P.TRADING_DAYS))
        yrs = len(r) / P.TRADING_DAYS
        bands[f"{b:.2f}"] = {"cagr": P.metrics(r, cash)["cagr"], "max_drawdown": P.metrics(r, cash)["max_drawdown"],
                             "turnover_per_year": round(float(bb["turnover"].loc[start:].sum() / yrs), 2),
                             "cost_drag_bp_per_year": round(float(bb["turnover"].loc[start:].sum() * 5 / yrs), 1),
                             "orders_per_year": round(bb.attrs["orders"] / (len(bb) / P.TRADING_DAYS), 1),
                             "tracking_error_vs_full": round(te, 4)}
    ok = [k for k, v in bands.items() if v["tracking_error_vs_full"] < 0.005]
    res["bands"] = bands
    res["band_selected"] = max(ok, key=float)
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)

    cols = ["cagr", "vol", "sharpe", "max_drawdown"]
    for part in ("full", "dev", "holdout"):
        print(f"--- {part} {res['period'] if part == 'full' else ''}")
        print(pd.DataFrame(res[part]).T[cols].to_string())
    print("per-day bp:", res["per_day_in_market_bp"], "tom share of days:", res["tom_days_in_market_share"])
    print("selected sleeve:", sel, "| holdout vs current book:", res["holdout_vs_current_book"], "| full:", res["full_vs_current_book"])
    print("deflated:", res["deflated_sharpe_selected"])
    for c in res["crises"]:
        print(" ", c)
    print(pd.DataFrame(bands).T.to_string(), "\nband selected:", res["band_selected"])


if __name__ == "__main__":
    main()
