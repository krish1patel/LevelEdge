#!/usr/bin/env python3
"""
Phase 3: news sentiment, put on trial like every other idea (not a prediction feature).

Data: the Federal Reserve Bank of San Francisco's Daily News Sentiment Index (Buckman, Shapiro,
Sudhof & Wilson 2020; method in Shapiro, Sudhof & Wilson, FRBSF WP 2017-01): lexical sentiment of
economics articles in 24 major US newspapers, daily since 1980, updated weekly.

Protocol (fixed before running)
-------------------------------
* No look-ahead: the value for day t is treated as known only from t + 8 calendar days (the file is
  published weekly). Weekly decisions at Friday's close, fills at the next open, 5 bp costs, cash
  earns T-bills. Development: start..2019-12-31; holdout 2020+ read once.
* Timing rules on SPY (else T-bills):
    level     hold while sentiment is above its trailing 252-trading-day average
    momentum  hold while sentiment is higher than 21 trading days earlier
  Pass: Sharpe above SPY buy-and-hold in BOTH periods (a timing rule must earn its trades).
* Overlay on the traded book: core SPY and IEF move to T-bills while the level rule is off
  (the trend-filter test in hedge_research.py, same pass rule): worst fall at least 5 points
  shallower in both periods, Sharpe not lower in both, full-period CAGR within 1 point a year.
* Crash warning: AUC of the level and the 21-day change for a 5%+ SPY fall over the next
  21 trading days (0.5 = coin flip), and the correlation of the 21-day change with the next
  month's SPY return, with a block-bootstrap 95% interval.
The trust-score engine (leveledge.integrity) also grades the level rule as "news_sentiment_spy".

Usage: python benchmarks/news_sentiment.py
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
from leveledge.news import load_sentiment, sentiment_signals  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "etf")
OUT = os.path.join(ROOT, "benchmarks", "results", "news_sentiment")
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import rankdata
    ok = np.isfinite(score) & np.isfinite(y)
    s, y = score[ok], y[ok].astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_corr_ci(a: pd.Series, b: pd.Series, block: int = 12, n: int = 2000, seed: int = 0):
    d = pd.concat([a, b], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(d) / block))
    est = []
    for _ in range(n):
        starts = rng.integers(0, len(d) - block, k)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:len(d)]
        est.append(np.corrcoef(d[idx, 0], d[idx, 1])[0, 1])
    return float(np.corrcoef(d[:, 0], d[:, 1])[0, 1]), [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def main():
    os.makedirs(OUT, exist_ok=True)
    sent = load_sentiment()
    cfg = live.config()
    prices = P.load_prices(cfg.universe, cache_dir=CACHE)
    opens, closes, cash = P.panel(prices, cfg.universe)
    sig = sentiment_signals(sent, closes.index)
    dates = P.rebalance_dates(closes.index, "weekly")
    dates = dates[dates >= sig["level"].first_valid_index()]

    def spy_rule(on: pd.Series) -> pd.DataFrame:
        d = pd.DataFrame(0.0, index=dates, columns=closes.columns)
        d["SPY"] = on.reindex(dates).fillna(0.0).to_numpy()
        return d

    hold = pd.DataFrame(0.0, index=dates, columns=closes.columns)
    hold["SPY"] = 1.0
    runs = {"spy_buy_hold": P.backtest(opens, closes, cash, hold, 5.0)["ret"],
            "level": P.backtest(opens, closes, cash, spy_rule(sig["level"]), 5.0)["ret"],
            "momentum": P.backtest(opens, closes, cash, spy_rule(sig["momentum"]), 5.0)["ret"]}

    # overlay on the traded book's core, next to the trend filter
    book, trend = live.book_decisions(opens, closes, cash, cfg)
    core = pd.Series(live.CORE, dtype=float).reindex(closes.columns).fillna(0.0)
    keep = sig["level"].reindex(book.index).fillna(1.0).to_numpy()[:, None]
    news_book = live.LEVERAGE * (live.CORE_SHARE * keep * core.values + live.TREND_SHARE * trend)
    runs["book"] = P.backtest(opens, closes, cash, book, 5.0)["ret"]
    runs["book_news_filter"] = P.backtest(opens, closes, cash, news_book, 5.0)["ret"]
    runs["book_trend_filter"] = P.backtest(opens, closes, cash, live.trend_filter_decisions(opens, closes, cash, cfg), 5.0)["ret"]
    start = max(r.index[0] for r in runs.values())
    runs = {k: v.loc[start:] for k, v in runs.items()}

    res = {"source": "FRBSF Daily News Sentiment Index (Buckman, Shapiro, Sudhof & Wilson 2020)",
           "sentiment_range": [str(sent.index[0].date()), str(sent.index[-1].date())], "lag_days": 8,
           "period": [str(start.date()), str(closes.index[-1].date())], "runs": {}}
    for k, r in runs.items():
        res["runs"][k] = {"full": P.metrics(r, cash), "dev": P.metrics(r.loc[:DEV_END], cash), "holdout": P.metrics(r.loc[HOLD_START:], cash)}
    R = res["runs"]
    for k in ("level", "momentum"):
        res["runs"][k]["passes"] = bool(all(R[k][p]["sharpe"] > R["spy_buy_hold"][p]["sharpe"] for p in ("dev", "holdout")))
        res["runs"][k]["in_market"] = float(sig[k].reindex(dates).mean())
        res["runs"][k]["sharpe_vs_spy"] = {p: P.sharpe_diff_ci(runs[k].loc[a:b], runs["spy_buy_hold"].loc[a:b], cash)
                                           for p, a, b in (("dev", None, DEV_END), ("holdout", HOLD_START, None))}
    b, f = R["book"], R["book_news_filter"]
    res["overlay_passes"] = bool(all(f[p]["max_drawdown"] >= b[p]["max_drawdown"] + 0.05 and f[p]["sharpe"] >= b[p]["sharpe"] for p in ("dev", "holdout"))
                                 and f["full"]["cagr"] >= b["full"]["cagr"] - 0.01)

    # crash warning and return correlation
    spy = closes["SPY"]
    fwd = spy.shift(-21) / spy - 1
    drop = (spy.shift(-21).rolling(21).min().shift(-20) / spy - 1)       # worst close over the next 21 days
    y = (drop <= -0.05).astype(float).where(drop.notna())
    feats = {"level_vs_avg": sig["level_raw"], "change_21d": sig["change_21d"]}
    res["crash_auc"] = {}
    for name, x in feats.items():
        xx = x.reindex(spy.index)
        # low sentiment / falling sentiment should warn: score = minus the feature
        res["crash_auc"][name] = {p: auc(-xx.loc[a:b].to_numpy(), y.loc[a:b].to_numpy())
                                  for p, a, b in (("dev", None, DEV_END), ("holdout", HOLD_START, None))}
    monthly = pd.DataFrame({"chg": sig["change_21d"].reindex(spy.index), "fwd": fwd}).iloc[::21].dropna()
    res["corr_change_next_month"] = {p: dict(zip(("r", "ci95"), block_corr_ci(m["chg"], m["fwd"])))
                                     for p, m in (("dev", monthly.loc[:DEV_END]), ("holdout", monthly.loc[HOLD_START:]))}
    res["base_rate_5pct_drop"] = float(y.dropna().mean())

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    print(f"period {res['period']}, sentiment lagged {res['lag_days']} days")
    for k in runs:
        m = R[k]
        print(f"{k:18s} CAGR {m['full']['cagr']:+.1%} DD {m['full']['max_drawdown']:.0%} Sharpe {m['full']['sharpe']:.2f} | dev {m['dev']['sharpe']:.2f} hold {m['holdout']['sharpe']:.2f}"
              + (f" | in market {m['in_market']:.0%} pass={m['passes']}" if "passes" in m else ""))
    print("overlay passes:", res["overlay_passes"])
    print("crash AUC:", json.dumps(res["crash_auc"], default=lambda x: round(x, 3)))
    print("corr(change, next month):", json.dumps(res["corr_change_next_month"], default=lambda x: round(x, 3)))


if __name__ == "__main__":
    main()
