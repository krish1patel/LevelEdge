#!/usr/bin/env python3
"""
Out-of-sample benchmark: LevelEdge v1 vs v2 on real, held-out sessions.

Protocol
--------
For each ticker x interval x past trading day x decision time (10:00, 12:00,
14:00 ET), every model sees only bars that had *closed* by the decision time
and forecasts P(close at 16:00 ET > K) for strikes at fixed % offsets. The
outcome is the realised 16:00 ET bar close. Nothing after the decision
time is visible to any model.

Models
------
v1          frozen copy of the original Predictor (one XGB per strike)
gauss_rw    naive reference: Gaussian random walk, EWMA vol, no seasonality
v2_base     v2 volatility-scaled empirical baseline only
v2          v2 stacked forecast (what the library returns)

Scores: log loss, Brier, Brier skill vs v1, AUC, calibration error (ECE), and a
day-block bootstrap 95% CI on Brier differences (forecasts on the same
ticker-day share one outcome path, so they are not independent).

Usage
-----
    python benchmarks/compare.py                       # full run (~10-15 min, 8 procs)
    python benchmarks/compare.py --days 5 --quick      # smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
import warnings
from datetime import datetime, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["LEVELEDGE_DISABLE_LOGGING"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leveledge.constants import US_EASTERN as ET  # noqa: E402
from leveledge.data import clean_bars, fetch_history  # noqa: E402
from leveledge.model import ForecastConfig, LevelForecaster  # noqa: E402
from leveledge.sessions import bar_ends, compute_horizon, interval_minutes, is_crypto_ticker, nyse_schedule  # noqa: E402

TICKERS = ["SPY", "QQQ", "NVDA", "TSLA", "BTC-USD"]
INTERVALS = ["15m", "1h"]
DECISION_TIMES = [time(10, 0), time(12, 0), time(14, 0)]
TARGET_TIME = time(16, 0)
STRIKES = [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]
SKIP_V1 = os.environ.get("LEVELEDGE_BENCH_SKIP_V1") == "1"
MODELS = (["v1"] if not SKIP_V1 else []) + ["gauss_rw", "v2_base", "v2"]

_DATA: dict = {}


def _init(data):
    global _DATA
    _DATA = data


def _outcome(full: pd.DataFrame, imin: int, crypto: bool, eff_target) -> float | None:
    ends = bar_ends(full.index, imin, crypto)
    hit = np.where(ends == eff_target)[0]
    return float(full["Close"].iloc[hit[0]]) if len(hit) else None


def run_case(case):
    ticker, interval, day, dt = case
    full = _DATA[(ticker, interval)]
    imin, crypto = interval_minutes(interval), is_crypto_ticker(ticker)
    as_of = datetime.combine(day, dt, tzinfo=ET)
    target = datetime.combine(day, TARGET_TIME, tzinfo=ET)
    try:
        bars = clean_bars(full, interval, ticker, as_of)
        hz = compute_horizon(bars.index[-1], target, imin, crypto)
    except ValueError:
        return []
    outcome = _outcome(full, imin, crypto, hz.effective_target)
    if outcome is None:
        return []
    S = float(bars["Close"].iloc[-1])
    levels = [S * (1 + s / 100) for s in STRIKES]
    out = {m: [np.nan] * len(levels) for m in MODELS}
    t0 = _time.time()

    # v2
    try:
        m = LevelForecaster(bars, imin, crypto, hz, ForecastConfig(n_jobs=1)).fit()
        comp = m.components(levels)
        out["v2"], out["v2_base"] = list(comp["p"]), list(comp["p_baseline"])
    except Exception as e:  # pragma: no cover
        print("v2 fail", case, e, flush=True)
    t_v2 = _time.time() - t0

    # naive Gaussian random walk
    lr = np.log(bars["Close"]).diff().dropna()
    sig = float(np.sqrt((lr ** 2).ewm(span=40, adjust=False).mean().iloc[-1]))
    out["gauss_rw"] = [float(ndtr(-np.log(K / S) / (sig * np.sqrt(hz.bars)))) for K in levels]

    # v1 (one model per strike, exactly like the old forward test)
    t1 = _time.time()
    if SKIP_V1:
        levels_v1 = []
    else:
        levels_v1 = levels
    from legacy_v1 import Predictor as V1
    # v1 received crypto bars from yfinance in UTC; its hourly grouping breaks on
    # US/Eastern DST transitions, so give it its native timestamps.
    v1_bars = bars.tz_convert("UTC") if crypto else bars
    for i, K in enumerate(levels_v1):
        try:
            p = V1(ticker, target, interval, K, end_datetime=as_of, data=v1_bars)
            p.train_xgb(evaluate=False)
            out["v1"][i] = float(p.predict_xgb())
        except Exception as e:
            print("v1 fail", case, K, e, flush=True)
    t_v1 = _time.time() - t1

    rows = []
    for i, (s, K) in enumerate(zip(STRIKES, levels)):
        r = dict(ticker=ticker, interval=interval, day=str(day), decision=dt.strftime("%H:%M"),
                 strike_pct=s, level=K, spot=S, outcome=outcome, y=int(outcome > K),
                 bars_ahead=hz.bars, sec_v1=t_v1, sec_v2=t_v2)
        for mname in MODELS:
            r[mname] = out[mname][i]
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def score(df: pd.DataFrame) -> pd.DataFrame:
    res = {}
    y = df["y"].to_numpy()
    for m in MODELS:
        p = df[m].to_numpy()
        res[m] = dict(log_loss=_ll(y, p), brier=float(np.mean((p - y) ** 2)), auc=_auc(y, p), ece=_ece(y, p))
    out = pd.DataFrame(res).T
    ref = "v1" if "v1" in out.index else "gauss_rw"
    out[f"brier_skill_vs_{ref}"] = 1 - out["brier"] / out.loc[ref, "brier"]
    return out


def bootstrap_diff(df, a, b, n=2000, seed=0):
    """95% CI of Brier(a) - Brier(b), resampling whole ticker-days."""
    rng = np.random.default_rng(seed)
    df = df.assign(block=df["ticker"] + "|" + df["day"])
    g = df.groupby("block").apply(lambda x: pd.Series({
        "da": ((x[a] - x["y"]) ** 2).sum(), "db": ((x[b] - x["y"]) ** 2).sum(), "n": len(x)}))
    vals = []
    for _ in range(n):
        s = g.iloc[rng.integers(0, len(g), len(g))]
        vals.append((s["da"].sum() - s["db"].sum()) / s["n"].sum())
    point = (g["da"].sum() - g["db"].sum()) / g["n"].sum()
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--tickers", nargs="+", default=TICKERS)
    ap.add_argument("--intervals", nargs="+", default=INTERVALS)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore predictions.partial.csv")
    ap.add_argument("--history-days", type=int, default=None,
                    help="truncate training history to this many calendar days before the test window")
    ap.add_argument("--skip-v1", action="store_true", help="skip the slow v1 model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.skip_v1:
        os.environ["LEVELEDGE_BENCH_SKIP_V1"] = "1"
        global MODELS, SKIP_V1
        SKIP_V1, MODELS = True, ["gauss_rw", "v2_base", "v2"]
    print("Fetching data...", flush=True)
    data = {(t, iv): fetch_history(t, iv) for t in args.tickers for iv in args.intervals}
    last = max(d.index[-1] for d in data.values())
    sched = nyse_schedule(last - pd.Timedelta(days=60), last)
    # only full days whose 16:00 close is in the data
    days = [d.date() for d in sched.index if sched.loc[d, "market_close"].hour == 16
            and sched.loc[d, "market_close"] <= last + pd.Timedelta(minutes=60)][-args.days:]
    if args.history_days:
        cut = pd.Timestamp(days[0], tz=ET) - pd.Timedelta(days=args.history_days)
        data = {k: v[v.index >= cut] for k, v in data.items()}
        print(f"Training history truncated to start {cut:%Y-%m-%d}", flush=True)
    cases = [(t, iv, d, dt) for t in args.tickers for iv in args.intervals
             for d in days for dt in (DECISION_TIMES[:1] if args.quick else DECISION_TIMES)]
    print(f"{len(cases)} forecast cases x {len(STRIKES)} strikes over {days[0]}..{days[-1]}", flush=True)

    t0 = _time.time()
    partial = os.path.join(args.out, "predictions.partial.csv")
    rows = []
    if os.path.exists(partial) and not args.fresh:
        prev = pd.read_csv(partial)
        rows = prev.to_dict(orient="records")
        done = set(zip(prev["ticker"], prev["interval"], prev["day"], prev["decision"]))
        cases = [c for c in cases if (c[0], c[1], str(c[2]), c[3].strftime("%H:%M")) not in done]
        print(f"Resuming: {len(done)} cases already scored, {len(cases)} to go", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    pending = list(cases)
    attempts = 0
    while pending and attempts < 5:
        attempts += 1
        try:
            with ProcessPoolExecutor(args.procs, initializer=_init, initargs=(data,),
                                     max_tasks_per_child=10) as ex:
                futs = {ex.submit(run_case, c): c for c in pending}
                for i, fut in enumerate(as_completed(futs), 1):
                    rows.extend(fut.result())
                    pending.remove(futs[fut])
                    if i % 25 == 0:
                        print(f"  {i}/{len(futs)} cases  ({_time.time() - t0:.0f}s)", flush=True)
                        pd.DataFrame(rows).to_csv(partial, index=False)
        except BrokenProcessPool:
            print(f"  worker died; restarting pool with {len(pending)} cases left", flush=True)
            pd.DataFrame(rows).to_csv(partial, index=False)
    if pending:
        print(f"WARNING: {len(pending)} cases never completed", flush=True)
    df = pd.DataFrame(rows).dropna(subset=MODELS)
    df.to_csv(os.path.join(args.out, "predictions.csv"), index=False)

    overall = score(df)
    report = {"cases": len(cases), "forecasts": len(df), "days": [str(days[0]), str(days[-1])],
              "overall": overall.round(4).to_dict(orient="index")}
    print("\n=== Overall ===\n", overall.round(4).to_string())
    for key in ["interval", "ticker", "strike_pct", "decision"]:
        tab = df.groupby(key).apply(lambda g: score(g)["brier"]).round(4)
        report[f"brier_by_{key}"] = tab.to_dict(orient="index")
        print(f"\n--- Brier by {key} ---\n", tab.to_string())
    for a, b in [(a, b) for a, b in [("v2", "v1"), ("v2", "gauss_rw"), ("v2", "v2_base")] if b in MODELS]:
        pt, lo, hi = bootstrap_diff(df, a, b)
        report[f"brier_diff_{a}_minus_{b}"] = dict(point=pt, ci95=[lo, hi])
        print(f"Brier({a}) - Brier({b}) = {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    speed = df.drop_duplicates(["ticker", "interval", "day", "decision"])[["sec_v1", "sec_v2"]].mean()
    report["seconds_per_case_8_strikes"] = speed.round(2).to_dict()
    print("\nSeconds per case (8 strikes, 1 thread):", speed.round(2).to_dict())
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {args.out}/summary.json and predictions.csv ({_time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
