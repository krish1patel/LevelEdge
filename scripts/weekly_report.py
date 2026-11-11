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
from leveledge.broker import AlpacaBroker, credentials, margin_interest, slippage_bp  # noqa: E402
from leveledge.constants import US_EASTERN  # noqa: E402


def pct(x: float, d: int = 2) -> str:
    return f"{'+' if x > 0 else '−' if x < 0 else ''}{abs(x) * 100:.{d}f}%"


def money(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x):,.2f}"


def account_curve(b: AlpacaBroker) -> pd.Series:
    h = b.portfolio_history(period="1A", timeframe="1D")
    s = pd.Series(h.get("equity") or [], index=pd.to_datetime(h.get("timestamp") or [], unit="s"), dtype=float)
    s.index = s.index.tz_localize("UTC").tz_convert(US_EASTERN).normalize().tz_localize(None)
    return s[s > 0]          # the account's own history (a reset clears it), so the first point is the start


def tbill_rate() -> float:
    """Latest 13-week T-bill yield (annual, as a fraction); 4% if unavailable."""
    try:
        import yfinance as yf
        h = yf.Ticker("^IRX").history(period="10d")
        return float(h["Close"].dropna().iloc[-1]) / 100
    except Exception:  # noqa: BLE001
        return 0.04


def execution_section(b: AlpacaBroker, days: int = 8) -> list[str]:
    """Slippage of recent fills against that day's official open."""
    import yfinance as yf

    since = pd.Timestamp.now(tz=US_EASTERN).normalize() - pd.Timedelta(days=days)
    fills = [o for o in b.orders(status="closed", limit=200)
             if o.get("filled_at") and o.get("filled_avg_price")
             and str(o.get("client_order_id", "")).startswith("core6040_")          # the strategy's own orders only
             and pd.Timestamp(o["filled_at"]).tz_convert(US_EASTERN) >= since]
    if not fills:
        return []
    rows, cost, notional = [], 0.0, 0.0
    opens: dict = {}
    for o in fills:
        t = pd.Timestamp(o["filled_at"]).tz_convert(US_EASTERN)
        sym, day = o["symbol"], t.normalize().tz_localize(None)
        if sym not in opens:
            h = yf.Ticker(sym).history(start=(since - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), interval="1d", auto_adjust=False)
            h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
            opens[sym] = h["Open"]
        ref = opens[sym].get(day)
        fill = float(o["filled_avg_price"])
        qty = float(o.get("filled_qty") or 0)
        bp = slippage_bp(o["side"], fill, float(ref)) if ref is not None else float("nan")
        at_open = t.hour == 9 and t.minute <= 35
        if bp == bp:
            cost += bp / 1e4 * fill * qty
            notional += fill * qty
        rows.append(f"| {t:%a %b %d %H:%M} | {o['side'].title()} {sym} | {fill:,.2f} | "
                    f"{'' if ref is None else f'{float(ref):,.2f}'} | {'' if bp != bp else f'{bp:+.1f}'} | {'at the open' if at_open else 'intraday'} |")
    out = ["### Execution", "", "| Filled | Order | Fill | Day's open | Slippage (bp) | When |", "|---|---|---|---|---|---|"] + rows
    if notional:
        out += ["", f"Fill cost against the open: {money(cost)} on {money(notional)} traded ({cost / notional * 1e4:+.1f} bp). "
                "Only fills at the open are a fair test; intraday fills also include the day's price move."]
    return out + [""]


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
              f"| Since {curve.index[0]:%Y-%m-%d} ({money(start_eq)}) | {pct(equity / start_eq - 1)} |",
              f"| Below peak | {pct(equity / peak - 1)} |",
              f"| Cash (negative = margin) | {money(float(acct['cash']))} |"]

    # the paper account charges no margin interest; a real one would
    borrowed = max(0.0, -float(acct["cash"]))
    days = max(1, (curve.index[-1] - curve.index[0]).days)
    tbill = tbill_rate()
    interest = margin_interest(borrowed, days, tbill, 0.02)
    if borrowed > 0:
        lines += [f"| Est. margin interest so far (T-bill {tbill:.2%} + 2%, not charged on paper) | −{money(interest)} |",
                  f"| Since start, net of that interest | {pct((equity - interest) / start_eq - 1)} |"]
    lines.append("")
    lines += execution_section(b)

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
        tr = live.tracking_since(curve.index[0].strftime("%Y-%m-%d"))
        if len(tr) >= 2 and len(curve) >= 2:
            acct_ret = equity / start_eq - 1
            rows = [("Paper account (actual, now)", acct_ret), (f"Book rules (model, to the {tr.index[-1]:%b %d} close)", tr["book_rules"].iloc[-1] - 1),
                    ("Shadow DAA book", tr["shadow_daa"].iloc[-1] - 1), ("60/40", tr["sixty_forty"].iloc[-1] - 1),
                    ("SPY", tr["spy_buy_hold"].iloc[-1] - 1)]
            lines += [f"### Since {curve.index[0]:%Y-%m-%d}", "", "| | Return |", "|---|---|"] + [f"| {n} | {pct(r)} |" for n, r in rows]
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
    near = sorted(t["assets"], key=lambda a: abs(a["flip_move"]))[:4]
    lines += ["### Nearest trend flips", "", "| ETF | Trend | Flips at | Move needed |", "|---|---|---|---|"]
    lines += [f"| {a['symbol']} | {'on: sells below' if a['trend_on'] else 'off: buys above'} | ${a['flip_price']:,.2f} | {pct(a['flip_move'], 1)} |"
              for a in near]
    lines += ["", f"Next rebalance: {t['next_rebalance']} after the close. Strategy `{live.STRATEGY_ID}`, paper account."]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
