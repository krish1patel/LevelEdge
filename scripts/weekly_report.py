#!/usr/bin/env python3
"""
Weekly report for the paper book, written as Markdown to stdout.

    python scripts/weekly_report.py [--trade-record paper_trade.json]

Covers: this run's trades, account equity and change over the week and since the
live start, drawdown from peak, positions vs targets, and the live account
against the book's own rules, the shadow DAA book, 60/40 and SPY over the same days.
With a trade record whose action is "hold", it prints nothing (no report on
non-rebalance days).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import pandas as pd  # noqa: E402

from leveledge import live  # noqa: E402
from leveledge.broker import AlpacaBroker, credentials  # noqa: E402
from leveledge.constants import US_EASTERN  # noqa: E402


def pct(x: float, d: int = 2) -> str:
    return f"{'+' if x > 0 else '−' if x < 0 else ''}{abs(x) * 100:.{d}f}%"


def money(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x):,.2f}"


def account_curve(b: AlpacaBroker) -> pd.Series:
    h = b.portfolio_history(period="1A", timeframe="1D")
    s = pd.Series(h.get("equity") or [], index=pd.to_datetime(h.get("timestamp") or [], unit="s"), dtype=float)
    s.index = s.index.tz_localize("UTC").tz_convert(US_EASTERN).normalize().tz_localize(None)
    s = s[s > 0]
    return s[s.index >= pd.Timestamp(live.LIVE_START)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trade-record", default=None)
    args = ap.parse_args()
    rec = None
    if args.trade_record and os.path.exists(args.trade_record):
        with open(args.trade_record) as f:
            rec = json.load(f)
        if str(rec.get("action", "")).startswith("hold"):
            return 0

    b = AlpacaBroker(*credentials())
    acct = b.account()
    equity, last = float(acct["equity"]), float(acct.get("last_equity") or acct["equity"])
    curve = account_curve(b)
    curve.loc[pd.Timestamp(datetime.now(tz=US_EASTERN).date())] = equity
    curve = curve[~curve.index.duplicated(keep="last")].sort_index()
    start_eq = float(curve.iloc[0]) if len(curve) else equity
    week_ago = curve[curve.index <= curve.index[-1] - pd.Timedelta(days=7)]
    week_base = float(week_ago.iloc[-1]) if len(week_ago) else start_eq
    peak = float(curve.max()) if len(curve) else equity

    lines = [f"## Paper book — week of {datetime.now(tz=US_EASTERN):%a %b %d, %Y}", ""]
    lines += ["| | |", "|---|---|",
              f"| Equity | {money(equity)} |",
              f"| Today | {money(equity - last)} ({pct(equity / last - 1)}) |",
              f"| Past week | {pct(equity / week_base - 1)} |",
              f"| Since {live.LIVE_START} | {pct(equity / start_eq - 1)} |",
              f"| Below peak | {pct(equity / peak - 1)} |",
              f"| Cash (negative = margin) | {money(float(acct['cash']))} |", ""]

    if rec and rec.get("orders"):
        lines += [f"### Trades this run ({rec.get('action', '')})", "", "| Order | Size | Reason |", "|---|---|---|"]
        for o in rec["orders"]:
            size = money(o["notional"]) if o.get("notional") is not None else f"{o.get('qty')} sh"
            lines.append(f"| {o['side'].title()} {o['symbol']} | {size} | {o.get('reason', '')} |")
        errs = [s for s in rec.get("submitted", []) if "error" in s]
        if errs:
            lines += ["", f"**{len(errs)} order(s) failed:** " + "; ".join(f"{e['symbol']}: {e['error']}" for e in errs)]
        lines.append("")
    elif rec:
        lines += [f"No trades this run: {rec.get('action', '')} (every ETF was within the {live.BAND:.0%} band).", ""]

    # live account vs the rules and benchmarks over the same days
    try:
        tr = live.tracking_since(live.LIVE_START)
        if len(tr) >= 2 and len(curve) >= 2:
            acct_ret = equity / start_eq - 1
            rows = [("Paper account (actual)", acct_ret), ("Book rules (model)", tr["book_rules"].iloc[-1] - 1),
                    ("Shadow DAA book", tr["shadow_daa"].iloc[-1] - 1), ("60/40", tr["sixty_forty"].iloc[-1] - 1),
                    ("SPY", tr["spy_buy_hold"].iloc[-1] - 1)]
            lines += [f"### Since {live.LIVE_START}", "", "| | Return |", "|---|---|"] + [f"| {n} | {pct(r)} |" for n, r in rows]
            lines += ["", "Actual vs model shows execution slippage; the shadow book is the untraded DAA candidate.", ""]
    except Exception as e:  # noqa: BLE001 - the report must still go out
        lines += [f"_Benchmark comparison unavailable: {e}_", ""]

    pos = sorted(b.positions(), key=lambda p: -float(p["market_value"]))
    targets = (rec or {}).get("targets") or {}
    if pos:
        lines += ["### Positions", "", "| ETF | Weight | Target | Value | Unrealized |", "|---|---|---|---|---|"]
        for p in pos:
            mv = float(p["market_value"])
            tgt = targets.get(p["symbol"])
            lines.append(f"| {p['symbol']} | {mv / equity:.1%} | {'' if tgt is None else f'{tgt:.1%}'} | {money(mv)} | {money(float(p['unrealized_pl']))} |")
        lines.append("")
    t = live.current_targets()
    lines += [f"Next rebalance: {t['next_rebalance']} after the close. Strategy `{live.STRATEGY_ID}`, paper account."]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
