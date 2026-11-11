#!/usr/bin/env python3
"""
Day-trading strategies from published papers, on 10 years of 5-minute bars.

orb_qqq   Zarattini, Barbon & Aziz (2023/2024), 5-minute opening range breakout on QQQ:
          first 5-minute candle up -> long, down -> short, flat (within one cent) -> no trade.
          Enter at the 9:35 open; stop at the first candle's low (long) or high (short);
          target 10x the stop distance; otherwise exit at the session's last close.
          Size: a stop-out loses 1% of current equity, capped at 4x leverage. Compounding.
intramom  Gao, Han, Li & Zhou (JFE 2018), market intraday momentum on SPY: the return
          from the previous close to 10:00 sets the side for the last half-hour
          (15:30 -> close), long if positive, short if negative, 1x equity.

Data: Alpaca consolidated (SIP) 5-minute bars, regular session only (NYSE calendar,
half-days included), split/dividend adjusted. Fills: market entries at the bar open;
stops at the stop price, or the bar open if the price gaps through; when one bar
touches both stop and target, the stop counts first. Costs: slippage of 0, 1 or 2 bp
per side plus $0.0005/share commission (the paper's assumption).

Out of sample: the ORB paper's sample ends in February 2023, so 2023-03 onward is
after publication. The intraday-momentum paper used 1993-2013, so all of 2016+ is.

Usage: python benchmarks/intraday_research.py
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
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from leveledge import portfolio as P  # noqa: E402
from leveledge.broker import credentials  # noqa: E402
from leveledge.constants import US_EASTERN as ET  # noqa: E402
from leveledge.sessions import nyse_schedule  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "intraday")
OUT = os.path.join(ROOT, "benchmarks", "results", "intraday")
START, PUB_END = "2016-01-01", "2023-02-28"
COMMISSION = 0.0005


def fetch_bars(sym: str) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{sym}_5min.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    k, s = credentials()
    H = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}
    rows, token = [], None
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    while True:
        params = {"symbols": sym, "timeframe": "5Min", "start": f"{START}T00:00:00Z", "end": end,
                  "feed": "sip", "adjustment": "all", "limit": 10000, "page_token": token}
        for attempt in range(5):
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H, params=params, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(2 + 3 * attempt)
        r.raise_for_status()
        j = r.json()
        rows += (j.get("bars") or {}).get(sym, [])
        token = j.get("next_page_token")
        print(f"  {sym}: {len(rows):,} bars", flush=True)
        if not token:
            break
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}).set_index("t")
    sched = nyse_schedule(df.index.min().normalize().tz_localize(None), df.index.max().normalize().tz_localize(None))
    day = df.index.normalize().tz_localize(None)
    o = sched["market_open"].reindex(day).to_numpy()
    c = sched["market_close"].reindex(day).to_numpy()
    keep = (df.index >= pd.DatetimeIndex(o)) & (df.index < pd.DatetimeIndex(c))     # regular session, bar starts
    df = df[keep][["open", "high", "low", "close", "volume"]]
    df.to_parquet(path)
    return df


def orb(bars: pd.DataFrame, slip_bp: float, equity0: float = 100_000.0) -> tuple[pd.Series, pd.DataFrame]:
    eq = equity0
    daily, trades = {}, []
    s = slip_bp / 1e4
    for day, g in bars.groupby(bars.index.normalize()):
        if len(g) < 3:
            continue
        first, rest = g.iloc[0], g.iloc[1:]
        if abs(first["close"] - first["open"]) < 0.01:
            daily[day] = 0.0
            continue
        side = 1 if first["close"] > first["open"] else -1
        entry = float(rest.iloc[0]["open"])
        stop = float(first["low"] if side == 1 else first["high"])
        R = abs(entry - stop)
        if R <= 0 or (side == 1 and stop >= entry) or (side == -1 and stop <= entry):
            daily[day] = 0.0
            continue
        target = entry + side * 10 * R
        shares = min(0.01 * eq / R, 4 * eq / entry)
        exit_px, how = float(rest.iloc[-1]["close"]), "close"
        for _, b in rest.iterrows():
            hit_stop = b["low"] <= stop if side == 1 else b["high"] >= stop
            hit_tgt = b["high"] >= target if side == 1 else b["low"] <= target
            if hit_stop:                                  # stop first when both touch (conservative)
                gap = b["open"] < stop if side == 1 else b["open"] > stop
                exit_px, how = (float(b["open"]) if gap else stop), "stop"
                break
            if hit_tgt:
                exit_px, how = target, "target"
                break
        e_fill = entry * (1 + side * s)
        x_fill = exit_px * (1 - side * s)
        pnl = side * (x_fill - e_fill) * shares - 2 * COMMISSION * shares
        r = pnl / eq
        trades.append({"day": day, "side": side, "entry": entry, "stop": stop, "exit": exit_px, "how": how,
                       "R_multiple": side * (exit_px - entry) / R, "ret": r, "pnl": pnl, "notional": shares * entry,
                       "leverage": shares * entry / eq})
        eq *= 1 + r
        daily[day] = r
    return pd.Series(daily).sort_index(), pd.DataFrame(trades)


def intraday_momentum(bars: pd.DataFrame, daily_close: pd.Series, slip_bp: float) -> tuple[pd.Series, pd.DataFrame]:
    s = slip_bp / 1e4
    out, trades = {}, []
    prev_close = daily_close.shift(1)
    for day, g in bars.groupby(bars.index.normalize()):
        pc = prev_close.get(day)
        t = g.index
        at10 = g[t.strftime("%H:%M") == "09:55"]
        last_half = g[t >= t[-1] - pd.Timedelta(minutes=25)]       # final 6 bars: the last half-hour
        if pc is None or np.isnan(pc) or at10.empty or len(last_half) < 6 or t[-1].strftime("%H:%M") != "15:55":
            continue                                                 # skip half-days (no 15:30-16:00 window)
        r1 = float(at10.iloc[0]["close"]) / pc - 1
        side = 1 if r1 > 0 else -1
        entry, exit_ = float(last_half.iloc[0]["open"]), float(last_half.iloc[-1]["close"])
        r = side * (exit_ * (1 - side * s) / (entry * (1 + side * s)) - 1) - 2 * COMMISSION / entry
        out[day] = r
        trades.append({"day": day, "side": side, "signal": r1, "ret": r})
    return pd.Series(out).sort_index(), pd.DataFrame(trades)


def trade_stats(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {}
    wins, losses = tr[tr["ret"] > 0], tr[tr["ret"] <= 0]
    out = {"trades": int(len(tr)), "win_rate": round(float(len(wins) / len(tr)), 3),
           "avg_ret_bp": round(float(tr["ret"].mean() * 1e4), 2),
           "avg_win_bp": round(float(wins["ret"].mean() * 1e4), 2) if len(wins) else None,
           "avg_loss_bp": round(float(losses["ret"].mean() * 1e4), 2) if len(losses) else None,
           "profit_factor": round(float(wins["ret"].sum() / -losses["ret"].sum()), 3) if len(losses) and losses["ret"].sum() < 0 else None,
           "worst_trade_bp": round(float(tr["ret"].min() * 1e4), 1),
           "longest_losing_streak": int(max((len(list(g)) for k, g in __import__("itertools").groupby(tr["ret"] <= 0) if k), default=0))}
    if "how" in tr:
        out["exits"] = tr["how"].value_counts().to_dict()
        out["avg_leverage"] = round(float(tr["leverage"].mean()), 2)
    return out


def summarize(daily: pd.Series, tr: pd.DataFrame, cash: pd.Series, sl=slice(None)) -> dict:
    d = daily.loc[sl]
    idx = pd.DatetimeIndex(d.index).tz_localize(None)
    r = pd.Series(d.to_numpy(), index=idx)
    c = cash.reindex(idx).fillna(0.0) * 0                    # intraday: capital sits idle overnight; no cash credit
    m = P.metrics(r, c)
    t = tr[(pd.DatetimeIndex(tr["day"]).tz_localize(None) >= idx[0]) & (pd.DatetimeIndex(tr["day"]).tz_localize(None) <= idx[-1])] if len(tr) else tr
    return {**m, **{f"trade_{k}": v for k, v in trade_stats(t).items()}}


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    qqq, spy = fetch_bars("QQQ"), fetch_bars("SPY")
    print(f"QQQ {len(qqq):,} bars {qqq.index[0]} .. {qqq.index[-1]}; SPY {len(spy):,}", flush=True)
    px = P.load_prices(["QQQ", "SPY"], start="2015-01-01", cache_dir=os.path.join(ROOT, "benchmarks", "cache", "etf"))
    cash = (px[P.TBILL]["Close"] / 100 / 252)
    spy_daily_close = spy.groupby(spy.index.normalize())["close"].last()

    res = {"period": [str(qqq.index[0].date()), str(qqq.index[-1].date())], "paper_sample_end": PUB_END, "orb": {}, "intramom": {}}
    for bp in (0.0, 1.0, 2.0):
        d, tr = orb(qqq, bp)
        res["orb"][f"{bp:g}bp"] = {"full": summarize(d, tr, cash), "paper_period": summarize(d, tr, cash, slice(None, PUB_END)),
                                    "after_publication": summarize(d, tr, cash, slice("2023-03-01", None))}
        if bp == 1.0:
            (1 + d).cumprod().rename("orb_1bp").to_frame().to_csv(os.path.join(OUT, "orb_equity.csv"))
            tr.to_csv(os.path.join(OUT, "orb_trades.csv"), index=False)
        d2, tr2 = intraday_momentum(spy, spy_daily_close, bp)
        res["intramom"][f"{bp:g}bp"] = {"full": summarize(d2, tr2, cash), "2016_2020": summarize(d2, tr2, cash, slice(None, "2020-12-31")),
                                         "2021_on": summarize(d2, tr2, cash, slice("2021-01-01", None))}
    q = qqq.groupby(qqq.index.normalize())["close"].last()
    qd = q.pct_change().dropna()
    res["qqq_buy_hold"] = {"full": P.metrics(pd.Series(qd.to_numpy(), index=pd.DatetimeIndex(qd.index).tz_localize(None)), cash * 0),
                           "after_publication": P.metrics(pd.Series(qd.loc["2023-03-01":].to_numpy(), index=pd.DatetimeIndex(qd.loc["2023-03-01":].index).tz_localize(None)), cash * 0)}
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)

    keys = ["cagr", "vol", "sharpe", "max_drawdown", "trade_trades", "trade_win_rate", "trade_avg_ret_bp", "trade_profit_factor"]
    for strat in ("orb", "intramom"):
        print(f"=== {strat}")
        for bp, parts in res[strat].items():
            for part, m in parts.items():
                print(f"  {bp:>4s} {part:18s}", {k: m.get(k) for k in keys})
    print("QQQ buy & hold:", {k: {kk: v[kk] for kk in ('cagr', 'sharpe', 'max_drawdown')} for k, v in res["qqq_buy_hold"].items()})
    print("ORB 1bp exits/leverage:", {k: res["orb"]["1bp"]["full"].get(k) for k in ("trade_exits", "trade_avg_leverage", "trade_longest_losing_streak", "trade_worst_trade_bp")})


if __name__ == "__main__":
    main()
