#!/usr/bin/env python3
"""
Market-making test: does quoting around LevelEdge's fair value make money on Kalshi?

The taker test (kalshi_edge.py) found no edge after paying the spread. The one
place the model beat the quoted midpoint was on wide-spread strikes, so this
asks whether a maker can earn that spread instead of paying it.

Strategy (per contract and decision time, model fair value p strictly inside the spread)
-------------------------------------------------------------------------------------
* rest a YES bid at floor(p - m) if it improves the bid, and a YES offer at
  ceil(p + m) if it improves the offer (m = margin);
* the order lives for W minutes (5, 15, or until the market closes);
* fill rule, from Kalshi's real trade tape: the bid is filled if any taker sold
  YES at or below its price while it was live; the offer if any taker bought
  YES at or above it. Being the best price, the order would have been hit first.
  One contract per order, filled at the order's own price;
* PnL settles on Kalshi's result, less a maker fee.

Control: the identical strategy with the market midpoint as fair value. If the
model adds information, quoting around it must beat quoting around the mid.

Usage: python benchmarks/kalshi_maker.py     (needs predictions.csv from kalshi_edge.py)
"""
from __future__ import annotations

import json
import os
import sys
import warnings

os.environ["LEVELEDGE_DISABLE_LOGGING"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import kalshi_edge as K  # noqa: E402
from leveledge.constants import US_EASTERN as ET  # noqa: E402
from leveledge.kalshi import KalshiClient, taker_fee  # noqa: E402

TRADES_CACHE = os.path.join(K.CACHE, "trades.parquet")
MARGINS = [0.01, 0.02, 0.03]
WINDOWS = [5, 15, None]               # minutes; None = until close
MAKER_FEE_RATES = [0.0175, 0.0]       # conservative first; some series charge no maker fee


def load_rows() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(K.OUT, "predictions.csv"))
    for c in ("target", "decision"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(ET)
    df = df[K.session_ok(df)]
    df = df[(df["p_v2"] > df["bid"] + 0.01) & (df["p_v2"] < df["ask"] - 0.01)]
    return df.reset_index(drop=True)


def fetch_trades(rows: pd.DataFrame, client: KalshiClient) -> pd.DataFrame:
    have = pd.read_parquet(TRADES_CACHE) if os.path.exists(TRADES_CACHE) else pd.DataFrame()
    done = set(have["ticker"].unique()) if not have.empty else set()
    spans = rows.groupby("ticker").agg(start=("decision", "min"), end=("target", "first"))
    todo = [t for t in spans.index if t not in done]
    print(f"trades: {len(spans)} tickers, {len(todo)} to fetch", flush=True)
    new = []
    for i, t in enumerate(todo):
        s, e = spans.loc[t, "start"], spans.loc[t, "end"]
        got = client._paginate("/markets/trades", "trades", {
            "ticker": t, "limit": 1000, "min_ts": int(s.timestamp()), "max_ts": int(e.timestamp())})
        new.append(pd.DataFrame([{
            "ticker": t, "ts": g.get("created_time"), "yes_price": float(g.get("yes_price_dollars") or 0),
            "count": float(g.get("count_fp") or 0), "taker_side": g.get("taker_side")} for g in got])
            if got else pd.DataFrame({"ticker": [t], "ts": [None], "yes_price": [np.nan], "count": [0.0],
                                      "taker_side": [None]}))
        if (i + 1) % 500 == 0:
            print(f"  trades {i + 1}/{len(todo)}", flush=True)
            have = _save(have, new)
            new = []
    return _save(have, new)


def _save(have, new):
    out = pd.concat([have] + new, ignore_index=True) if new else have
    if not out.empty:
        out.to_parquet(TRADES_CACHE)
    return out


def simulate(rows: pd.DataFrame, trades: pd.DataFrame, fair_col: str, m: float, window, fee_rate: float):
    tr = trades.dropna(subset=["ts"]).copy()
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True, format="ISO8601").dt.tz_convert(ET)
    by_t = {t: g for t, g in tr.groupby("ticker")}
    out = []
    for r in rows.itertuples(index=False):
        fair = getattr(r, fair_col)
        g = by_t.get(r.ticker)
        end = r.target if window is None else min(r.target, r.decision + pd.Timedelta(minutes=window))
        live = g[(g["ts"] > r.decision) & (g["ts"] <= end)] if g is not None else None
        bid_px = np.floor((fair - m) * 100 + 1e-9) / 100
        ask_px = np.ceil((fair + m) * 100 - 1e-9) / 100
        if r.bid < bid_px < r.ask:
            filled = live is not None and ((live["taker_side"] == "no") & (live["yes_price"] <= bid_px + 1e-9)).any()
            out.append((r.day, r.series, "bid", bool(filled),
                        (r.y - bid_px - taker_fee(bid_px, fee_rate)) if filled else 0.0))
        if r.bid < ask_px < r.ask:
            filled = live is not None and ((live["taker_side"] == "yes") & (live["yes_price"] >= ask_px - 1e-9)).any()
            out.append((r.day, r.series, "offer", bool(filled),
                        (ask_px - r.y - taker_fee(1 - ask_px, fee_rate)) if filled else 0.0))
    return pd.DataFrame(out, columns=["day", "series", "side", "filled", "pnl"])


def stats(o: pd.DataFrame, all_days):
    f = o[o["filled"]]
    if f.empty:
        return {"orders": int(len(o)), "fills": 0}
    daily = f.groupby("day")["pnl"].sum().reindex(all_days, fill_value=0.0)
    sd = daily.std(ddof=1)
    lo, hi = K.day_bootstrap(f, lambda s: s["pnl"].mean())
    return {"orders": int(len(o)), "fills": int(len(f)), "fill_rate": round(len(f) / len(o), 3),
            "avg_pnl_cents": round(100 * f["pnl"].mean(), 3), "avg_pnl_cents_ci95": [round(100 * lo, 3), round(100 * hi, 3)],
            "total_usd": round(float(f["pnl"].sum()), 2),
            "sharpe_daily_annualized": round(float(daily.mean() / sd * np.sqrt(252)), 2) if sd > 0 else None,
            "by_series_avg_pnl_cents": {s: round(100 * g["pnl"].mean(), 3) for s, g in f.groupby("series")}}


def main():
    rows = load_rows()
    print(f"{len(rows)} quote rows with model fair value inside the spread", flush=True)
    trades = fetch_trades(rows, KalshiClient())
    all_days = sorted(rows["day"].unique())
    res = {"rows": int(len(rows)), "tickers": int(rows["ticker"].nunique()), "days": len(all_days), "results": {}}
    for fee in MAKER_FEE_RATES:
        for m in MARGINS:
            for w in WINDOWS:
                key = f"margin={m}|window={w or 'close'}|maker_fee={fee}"
                res["results"][key] = {
                    "model": stats(simulate(rows, trades, "p_v2", m, w, fee), all_days),
                    "control_mid": stats(simulate(rows, trades, "mid", m, w, fee), all_days)}
                r = res["results"][key]
                print(key, "| model", r["model"].get("fills"), r["model"].get("avg_pnl_cents"), r["model"].get("avg_pnl_cents_ci95"),
                      "| mid", r["control_mid"].get("fills"), r["control_mid"].get("avg_pnl_cents"), r["control_mid"].get("avg_pnl_cents_ci95"), flush=True)
    with open(os.path.join(K.OUT, "maker_summary.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
