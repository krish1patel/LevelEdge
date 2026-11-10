#!/usr/bin/env python3
"""
Does LevelEdge beat the market? Out-of-sample test against Kalshi prices.

Question
--------
Kalshi's hourly above/below contracts (S&P 500, Nasdaq-100, Bitcoin) ask the
exact question LevelEdge answers, and their price is the market's probability.
A forecast is only tradeable if it beats that price after fees and spread.

Protocol
--------
For every settled hourly event and decision times 60, 30 and 15 minutes before
it resolves:

* LevelEdge sees only 15-minute bars that had closed by the decision time and
  forecasts P(above K) for every strike near the money (selected by the spot
  price at T-60, never by the outcome).
* The market quote is the last minute-candle bid/ask at or before the
  decision time (at most 5 minutes old).
* The outcome is Kalshi's own settlement (``result``), so no proxy is involved.

Scores
------
1. Calibration: Brier and log loss of the market mid, the LevelEdge baseline,
   the full gated LevelEdge model, and an out-of-sample blend of market and
   model (logit pooling fitted only on earlier days). Brier differences carry a
   day-block bootstrap 95% CI.
2. Trading: buy YES when p - ask - fee >= theta, buy NO when bid - p - fee >= theta,
   one contract per signal, taker fills at the quoted price. Compared with
   a random picker that makes the same number of trades per day on random
   quoted contracts and sides, and with always buying the favourite.

Primary test, fixed before running: signal = full model, theta = 0.02,
fee rate = 0.07, all series pooled. Every other row is a secondary/sensitivity
result and should be read with multiple testing in mind.

Usage
-----
    python benchmarks/kalshi_edge.py              # collect (cached) + fit + score
    python benchmarks/kalshi_edge.py --score-only # re-score saved predictions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["LEVELEDGE_DISABLE_LOGGING"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge.constants import US_EASTERN as ET  # noqa: E402
from leveledge.data import fetch_history  # noqa: E402
from leveledge.kalshi import SERIES, KalshiClient, quote_at, taker_fee  # noqa: E402
from leveledge.model import ForecastConfig, LevelForecaster  # noqa: E402
from leveledge.data import clean_bars  # noqa: E402
from leveledge.sessions import compute_horizon, interval_minutes, is_crypto_ticker  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "kalshi")
OUT = os.path.join(ROOT, "benchmarks", "results", "kalshi")
SERIES_RUN = ["KXINXU", "KXNASDAQ100U", "KXBTCD"]
OFFSETS_MIN = [60, 30, 15]
BAND = {"^GSPC": 0.012, "^NDX": 0.015, "BTC-USD": 0.03}   # strikes within +-band of spot at T-60
WARMUP_DAYS = 7
INTERVAL = "15m"
MID_RANGE = (0.02, 0.98)
EPS = 1e-4

PRIMARY = {"signal": "p_v2", "theta": 0.02, "fee_rate": 0.07}


# ---------------------------------------------------------------------------
# stage 1: collect (network, cached)
# ---------------------------------------------------------------------------
def collect(series: str, client: KalshiClient) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(CACHE, exist_ok=True)
    spec = SERIES[series]
    bars_path = os.path.join(CACHE, f"bars_{spec.underlying.replace('^', '')}.parquet")
    if os.path.exists(bars_path):
        bars = pd.read_parquet(bars_path)
    else:
        bars = fetch_history(spec.underlying, INTERVAL)
        bars.to_parquet(bars_path)
    start = (bars.index[0] + pd.Timedelta(days=WARMUP_DAYS)).to_pydatetime()
    end = bars.index[-1].to_pydatetime()

    mk_path = os.path.join(CACHE, f"markets_{series}.parquet")
    if os.path.exists(mk_path):
        markets = pd.read_parquet(mk_path)
    else:
        markets = client.settled_markets(series, start, end + pd.Timedelta(minutes=15))
        markets.to_parquet(mk_path)
    markets = markets[markets["y"].notna()]

    cd_path = os.path.join(CACHE, f"candles_{series}.parquet")
    have = pd.read_parquet(cd_path) if os.path.exists(cd_path) else pd.DataFrame()
    done = set(have["event_ticker"].unique()) if not have.empty else set()
    imin = interval_minutes(INTERVAL)
    crypto = is_crypto_ticker(spec.underlying)
    new = []
    events = markets.groupby("event_ticker")
    todo = [e for e in events.groups if e not in done]
    print(f"[{series}] {markets['event_ticker'].nunique()} events, {len(todo)} to fetch", flush=True)
    for i, ev in enumerate(sorted(todo)):
        g = events.get_group(ev)
        T = g["close_time"].iloc[0]
        first = T - pd.Timedelta(minutes=max(OFFSETS_MIN))
        seen = bars[bars.index + pd.Timedelta(minutes=imin) <= first]
        if len(seen) < 100 or T > bars.index[-1] + pd.Timedelta(minutes=imin):
            continue
        spot = float(seen["Close"].iloc[-1])
        band = BAND[spec.underlying]
        sel = g[(g["strike"] >= spot * (1 - band)) & (g["strike"] <= spot * (1 + band))]
        if sel.empty:
            continue
        c = client.candles(sel["ticker"].tolist(), first - pd.Timedelta(minutes=6),
                           T - pd.Timedelta(minutes=min(OFFSETS_MIN) - 1))
        if not c.empty:
            c["event_ticker"] = ev
            new.append(c)
        if (i + 1) % 100 == 0:
            print(f"  [{series}] candles {i + 1}/{len(todo)}", flush=True)
            _flush(have, new, cd_path)
    candles = _flush(have, new, cd_path)
    return bars, markets, candles


def _flush(have, new, path):
    allc = pd.concat([have] + new, ignore_index=True) if new else have
    if not allc.empty:
        allc.to_parquet(path)
    return allc


# ---------------------------------------------------------------------------
# stage 2: model forecasts (CPU, parallel)
# ---------------------------------------------------------------------------
_BARS: dict = {}


def _init(bars_by_sym):
    global _BARS
    _BARS = bars_by_sym


def _fit_case(case):
    sym, T, d, strikes = case
    raw = _BARS[sym]
    crypto = is_crypto_ticker(sym)
    imin = interval_minutes(INTERVAL)
    try:
        bars = clean_bars(raw, INTERVAL, sym, d.to_pydatetime())
        hz = compute_horizon(bars.index[-1], T.to_pydatetime(), imin, crypto)
        if hz.effective_target != T:   # forecast must resolve exactly on the contract's time
            return None
        out = {}
        for name, ev in (("p_base", False), ("p_v2", True)):
            m = LevelForecaster(bars, imin, crypto, hz, ForecastConfig(n_jobs=1)).fit(evaluate=ev)
            out[name] = m.prob_above(strikes)
            if ev:
                out["xgb_on"] = float(m.weights[0] > 0)
        out["spot"] = float(bars["Close"].iloc[-1])
        return out
    except Exception as e:  # noqa: BLE001 - one bad case must not kill the run
        return {"error": f"{type(e).__name__}: {e}"}


def build_rows(series, bars, markets, candles, procs) -> pd.DataFrame:
    spec = SERIES[series]
    cases, meta = [], []
    mk = markets.set_index("ticker")
    for ev, c in candles.groupby("event_ticker"):
        T = mk.loc[c["ticker"].iloc[0], "close_time"]
        for off in OFFSETS_MIN:
            d = T - pd.Timedelta(minutes=off)
            q = quote_at(c, d)
            if q.empty:
                continue
            q = q.join(mk[["strike", "y"]], on="ticker")
            q["mid"] = (q["bid"] + q["ask"]) / 2
            q = q[(q["mid"] >= MID_RANGE[0]) & (q["mid"] <= MID_RANGE[1])]
            if q.empty:
                continue
            cases.append((spec.underlying, T, d, q["strike"].to_numpy()))
            meta.append((ev, T, d, off, q))
    print(f"[{series}] fitting {len(cases)} decision points", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=procs, initializer=_init, initargs=({spec.underlying: bars},)) as ex:
        for (ev, T, d, off, q), res in zip(meta, ex.map(_fit_case, cases, chunksize=4)):
            if res is None or "error" in res:
                continue
            q = q.copy()
            q["p_base"], q["p_v2"] = res["p_base"], res["p_v2"]
            q["xgb_on"], q["spot"] = res["xgb_on"], res["spot"]
            q["series"], q["event_ticker"], q["target"], q["decision"], q["offset_min"] = series, ev, T, d, off
            rows.append(q)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["day"] = out["target"].dt.strftime("%Y-%m-%d")
    return out


# ---------------------------------------------------------------------------
# stage 3: scoring
# ---------------------------------------------------------------------------
def _clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(y, p):
    p = _clip(p)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def day_bootstrap(df, stat, n=2000, seed=0):
    """95% CI of ``stat(sub_df)`` resampling whole days (outcomes within a day are correlated)."""
    rng = np.random.default_rng(seed)
    days = df["day"].unique()
    groups = {d: g for d, g in df.groupby("day")}
    vals = []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        vals.append(stat(pd.concat([groups[d] for d in pick], ignore_index=True)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def add_blend(df):
    """Out-of-sample logit pooling of market mid and model, refit each day on earlier days only."""
    df = df.sort_values("decision").copy()
    df["p_blend"] = np.nan
    df["blend_w_model"] = np.nan
    days = sorted(df["day"].unique())
    lm, lv = logit(_clip(df["mid"])), logit(_clip(df["p_v2"]))
    for i, d in enumerate(days):
        if i < 5:
            continue
        tr = df["day"].isin(days[:i]).to_numpy()
        te = (df["day"] == d).to_numpy()
        w = _fit_logit2(lm[tr], lv[tr], df["y"].to_numpy()[tr])
        df.loc[te, "p_blend"] = expit(w[0] * lm[te] + w[1] * lv[te])
        df.loc[te, "blend_w_model"] = w[1]
    return df


def _fit_logit2(a, b, y, iters=50):
    """Intercept-free logistic regression on two logit inputs (Newton)."""
    X = np.column_stack([a, b])
    w = np.array([1.0, 0.0])
    for _ in range(iters):
        p = expit(X @ w)
        g = X.T @ (p - y)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def trades(df, signal, theta, fee_rate):
    """Taker trades: YES at the ask, NO at (1 - bid). PnL per contract in dollars."""
    p = df[signal].to_numpy()
    ask, bid, y = df["ask"].to_numpy(), df["bid"].to_numpy(), df["y"].to_numpy()
    fee_yes, fee_no = taker_fee(ask, fee_rate), taker_fee(1 - bid, fee_rate)
    edge_yes = p - ask - fee_yes
    edge_no = (1 - p) - (1 - bid) - fee_no
    buy_yes = (edge_yes >= theta) & (edge_yes >= edge_no)
    buy_no = (edge_no >= theta) & ~buy_yes
    pnl = np.where(buy_yes, y - ask - fee_yes, np.where(buy_no, (1 - y) - (1 - bid) - fee_no, np.nan))
    t = df.loc[buy_yes | buy_no, ["day", "series", "offset_min", "event_ticker", "mid"]].copy()
    t["side"] = np.where(buy_yes, "yes", "no")[buy_yes | buy_no]
    t["edge"] = np.where(buy_yes, edge_yes, edge_no)[buy_yes | buy_no]
    t["pnl"] = pnl[buy_yes | buy_no]
    return t


def trade_stats(t, all_days, df=None, fee_rate=0.07, n_random=2000, seed=1):
    if t.empty:
        return {"trades": 0}
    daily = t.groupby("day")["pnl"].sum().reindex(all_days, fill_value=0.0)
    eq = daily.cumsum()
    dd = float((eq - eq.cummax()).min())
    sd = daily.std(ddof=1)
    out = {
        "trades": int(len(t)),
        "days_traded": int((t.groupby("day").size() > 0).sum()),
        "avg_pnl_cents": round(100 * t["pnl"].mean(), 3),
        "total_pnl_per_contract_usd": round(float(t["pnl"].sum()), 2),
        "hit_rate": round(float((t["pnl"] > 0).mean()), 4),
        "avg_edge_cents": round(100 * t["edge"].mean(), 3),
        "sharpe_daily_annualized": round(float(daily.mean() / sd * np.sqrt(252)), 2) if sd > 0 else None,
        "max_drawdown_usd": round(dd, 2),
    }
    tt = t.assign(day=t["day"])
    lo, hi = day_bootstrap(tt, lambda s: s["pnl"].mean())
    out["avg_pnl_cents_ci95"] = [round(100 * lo, 3), round(100 * hi, 3)]
    if df is not None:
        out["random_picker"] = random_picker(df, t, fee_rate, n_random, seed)
    return out


def random_picker(df, t, fee_rate, n, seed):
    """Same trades per day, random quoted contract and random side, taker fills."""
    rng = np.random.default_rng(seed)
    per_day = t.groupby("day").size()
    by_day = {d: g for d, g in df.groupby("day")}
    totals = np.zeros(n)
    for d, k in per_day.items():
        g = by_day[d]
        ask, bid, y = g["ask"].to_numpy(), g["bid"].to_numpy(), g["y"].to_numpy()
        pnl_yes = y - ask - taker_fee(ask, fee_rate)
        pnl_no = (1 - y) - (1 - bid) - taker_fee(1 - bid, fee_rate)
        idx = rng.integers(0, len(g), size=(n, k))
        side = rng.random((n, k)) < 0.5
        totals += np.where(side, pnl_yes[idx], pnl_no[idx]).sum(axis=1)
    strat = float(t["pnl"].sum())
    return {"mean_total_usd": round(float(totals.mean()), 2),
            "p5_p95_total_usd": [round(float(np.percentile(totals, 5)), 2), round(float(np.percentile(totals, 95)), 2)],
            "p_value_strategy_beats_random": round(float((totals >= strat).mean()), 4)}


def calibration_block(df, cols):
    out = {}
    for c in cols:
        d = df.dropna(subset=[c])
        out[c] = {"n": int(len(d)), "brier": round(brier(d["y"], d[c]), 5),
                  "log_loss": round(logloss(d["y"], d[c]), 5), "ece": round(ece(d["y"], d[c]), 4)}
    return out


def brier_diff(df, a, b):
    d = df.dropna(subset=[a, b])
    if d.empty:
        return None
    diff = brier(d["y"], d[a]) - brier(d["y"], d[b])
    lo, hi = day_bootstrap(d, lambda s: brier(s["y"], s[a]) - brier(s["y"], s[b]))
    return {"diff": round(diff, 5), "ci95": [round(lo, 5), round(hi, 5)], "n": int(len(d))}


def session_ok(df: pd.DataFrame) -> pd.Series:
    """Stocks: the model must have seen today's first 15-minute bar (decision >= 09:45 ET).

    Earlier decisions anchor on yesterday's close while the market already
    prices the overnight move, which is an unfair, untradeable comparison. The
    live product applies the same rule.
    """
    stock = df["series"] != "KXBTCD"
    mins = df["decision"].dt.hour * 60 + df["decision"].dt.minute
    return ~stock | (mins >= 9 * 60 + 45)


LIQUID_SPREAD = 0.05


def score(df: pd.DataFrame) -> dict:
    n_raw = len(df)
    df = df[session_ok(df)].copy()
    df = add_blend(df)
    all_days = sorted(df["day"].unique())
    liquid = df[(df["ask"] - df["bid"]) <= LIQUID_SPREAD]
    res = {
        "rows_before_session_filter": int(n_raw),
        "rows": int(len(df)), "events": int(df["event_ticker"].nunique()), "days": len(all_days),
        "period": [all_days[0], all_days[-1]],
        "by_series_rows": df.groupby("series").size().to_dict(),
        "primary_test": PRIMARY,
        "calibration": calibration_block(df, ["mid", "p_base", "p_v2", "p_blend"]),
        "brier_v2_minus_market": brier_diff(df, "p_v2", "mid"),
        "brier_base_minus_market": brier_diff(df, "p_base", "mid"),
        "brier_blend_minus_market": brier_diff(df, "p_blend", "mid"),
        "blend_model_weight_last": round(float(df["blend_w_model"].dropna().iloc[-1]), 4)
            if df["blend_w_model"].notna().any() else None,
        "xgb_active_share": round(float(df.groupby(["event_ticker", "offset_min"])["xgb_on"].first().mean()), 3),
        "avg_spread_cents": round(100 * float((df["ask"] - df["bid"]).mean()), 3),
    }
    res["liquid"] = {
        "max_spread": LIQUID_SPREAD, "rows": int(len(liquid)),
        "calibration": calibration_block(liquid, ["mid", "p_v2", "p_blend"]),
        "brier_v2_minus_market": brier_diff(liquid, "p_v2", "mid"),
        "brier_blend_minus_market": brier_diff(liquid, "p_blend", "mid"),
        "strategies": {
            f"{sig}|theta={th}|fee=0.07": trade_stats(
                trades(liquid.dropna(subset=[sig]), sig, th, 0.07), all_days,
                df=liquid.dropna(subset=[sig]) if th == 0.02 else None)
            for sig in ["p_v2", "p_blend"] for th in [0.0, 0.02, 0.05]},
    }
    res["calibration_by_series"] = {s: calibration_block(g, ["mid", "p_v2"]) for s, g in df.groupby("series")}
    res["calibration_by_series_x_offset"] = {
        f"{s}|{int(o)}": calibration_block(g, ["mid", "p_v2"]) for (s, o), g in df.groupby(["series", "offset_min"])}
    res["calibration_by_offset"] = {int(o): calibration_block(g, ["mid", "p_v2"]) for o, g in df.groupby("offset_min")}

    # trading
    strat = {}
    for sig in ["p_v2", "p_base", "p_blend"]:
        for theta in [0.0, 0.02, 0.05]:
            for fee in [0.07, 0.035]:
                d = df.dropna(subset=[sig])
                t = trades(d, sig, theta, fee)
                is_primary = (sig, theta, fee) == (PRIMARY["signal"], PRIMARY["theta"], PRIMARY["fee_rate"])
                strat[f"{sig}|theta={theta}|fee={fee}"] = trade_stats(
                    t, sorted(d["day"].unique()), df=d if is_primary or theta == 0.02 else None, fee_rate=fee)
    res["strategies"] = strat
    prim = trades(df, PRIMARY["signal"], PRIMARY["theta"], PRIMARY["fee_rate"])
    res["primary_by_series"] = {s: trade_stats(g, all_days) for s, g in prim.groupby("series")}
    res["primary_by_offset"] = {int(o): trade_stats(g, all_days) for o, g in prim.groupby("offset_min")}
    res["primary_by_side"] = {s: trade_stats(g, all_days) for s, g in prim.groupby("side")}
    res["primary_by_price"] = {
        str(b): trade_stats(g, all_days) for b, g in prim.groupby(pd.cut(prim["mid"], [0, 0.1, 0.3, 0.7, 0.9, 1.0]))}

    # reference strategies
    fav_yes = df["mid"] >= 0.5
    pnl_fav = np.where(fav_yes, df["y"] - df["ask"] - taker_fee(df["ask"]),
                       (1 - df["y"]) - (1 - df["bid"]) - taker_fee(1 - df["bid"]))
    res["reference_buy_favourite_avg_pnl_cents"] = round(100 * float(np.mean(pnl_fav)), 3)
    ls = df[df["ask"] <= 0.15]
    res["reference_sell_longshots_avg_pnl_cents"] = round(100 * float(np.mean(
        (1 - ls["y"]) - (1 - ls["bid"]) - taker_fee(1 - ls["bid"]))), 3) if len(ls) else None
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--series", nargs="*", default=SERIES_RUN)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "predictions.csv")
    t0 = _time.time()
    if not args.score_only:
        client = KalshiClient()
        frames = []
        for s in args.series:
            bars, markets, candles = collect(s, client)
            if candles.empty:
                print(f"[{s}] no candles; skipped")
                continue
            frames.append(build_rows(s, bars, markets, candles, args.procs))
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(rows_path, index=False)
    df = pd.read_csv(rows_path, parse_dates=["target", "decision"])
    df["target"] = pd.to_datetime(df["target"], utc=True).dt.tz_convert(ET)
    df["decision"] = pd.to_datetime(df["decision"], utc=True).dt.tz_convert(ET)
    res = score(df)
    res["seconds"] = round(_time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps({k: res[k] for k in ["rows", "events", "days", "period", "calibration",
                                          "brier_v2_minus_market", "brier_blend_minus_market"]}, indent=2, default=str))
    print("PRIMARY:", json.dumps(res["strategies"][f"{PRIMARY['signal']}|theta={PRIMARY['theta']}|fee={PRIMARY['fee_rate']}"], default=str))


if __name__ == "__main__":
    main()
