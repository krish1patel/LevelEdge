#!/usr/bin/env python3
"""
One step of the LevelEdge paper-trading loop (run after the close on weekdays).

1. Compute target weights from the latest completed close (leveledge.live).
2. Only on the last trading day of the week (or with --force): read the
   Alpaca paper account, plan orders (no borrowing, 1% no-trade band), and
   submit them with --execute. Day market orders placed after the close fill
   at the next open, matching the backtest.
3. Print a JSON record of what was decided and why.

Without --execute, or without Alpaca keys, nothing is sent: the plan is
printed against the real account (or a hypothetical $100,000 one).

Env: ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY (paper keys from alpaca.markets).
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
    load_dotenv()
except ImportError:
    pass

from leveledge import live, safety  # noqa: E402
from leveledge.broker import AlpacaBroker, credentials, plan_orders  # noqa: E402
from leveledge.constants import US_EASTERN  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="submit orders to the Alpaca paper account")
    ap.add_argument("--force", action="store_true", help="rebalance even if today is not the weekly rebalance day")
    ap.add_argument("--band", type=float, default=live.BAND)
    args = ap.parse_args()

    t = live.current_targets()
    record = {"run_at": datetime.now(tz=US_EASTERN).isoformat(), "strategy": t["strategy"],
              "as_of_close": t["as_of_close"], "targets": t["weights"], "cash_weight": t["cash_weight"],
              "rebalance_day": t["rebalance_today"], "orders": [], "submitted": []}
    if not (t["rebalance_today"] or args.force):
        record["action"] = "hold: not the weekly rebalance day"
        print(json.dumps(record, indent=1))
        return 0

    creds = credentials()
    prices = {a["symbol"]: a["last_close"] for a in t["assets"]}
    if not creds:
        orders = plan_orders(t["weights"], 100_000.0, 100_000.0, {}, prices, {}, band=args.band, max_gross=live.MAX_GROSS)
        record.update(action="plan only: no Alpaca keys (hypothetical $100,000 account)",
                      orders=[o.as_dict() for o in orders])
        print(json.dumps(record, indent=1))
        return 0

    broker = AlpacaBroker(*creds)
    acct = broker.account()
    equity, cash = float(acct["equity"]), float(acct["cash"])
    raw_positions = broker.positions()
    positions = {p["symbol"]: {"market_value": float(p["market_value"]), "qty": float(p["qty"])}
                 for p in raw_positions}
    alpaca_prices = {p["symbol"]: float(p["current_price"]) for p in raw_positions if p.get("current_price")}
    fractionable = {s: bool(broker.asset(s).get("fractionable", False)) for s in t["weights"]}
    if float(acct.get("multiplier", 1)) < live.LEVERAGE:
        record.update(action=f"refused: the account's margin multiplier ({acct.get('multiplier')}) is below the book's {live.LEVERAGE}x")
        print(json.dumps(record, indent=1))
        return 1
    orders = plan_orders(t["weights"], equity, cash, positions, prices, fractionable, band=args.band,
                         max_gross=live.MAX_GROSS)
    record.update(account={"equity": equity, "cash": cash, "paper": broker.is_paper},
                  positions=positions, orders=[o.as_dict() for o in orders])
    prefix = f"{live.STRATEGY_ID}-{t['as_of_close']}-"
    checks = safety.pretrade_checks(t, acct, broker.is_paper, positions, alpaca_prices, orders, equity, live.MAX_GROSS,
                                    [str(o.get("client_order_id", "")) for o in broker.orders(status="all", limit=200)],
                                    prefix, datetime.now(tz=US_EASTERN))
    record["safety"] = safety.summarize(checks)
    if not record["safety"]["passed"]:
        record["action"] = "halted: safety check failed (" + ", ".join(record["safety"]["failed"]) + "); no orders sent"
        print(json.dumps(record, indent=1))
        return 1
    if not args.execute:
        record["action"] = "dry run: orders planned, not sent (use --execute)"
        print(json.dumps(record, indent=1))
        return 0

    broker.cancel_open_orders()          # never stack this week's orders on last week's leftovers
    for i, o in enumerate(orders):
        try:
            res = broker.submit(o, client_order_id=f"{prefix}{i}")
            record["submitted"].append({"symbol": o.symbol, "side": o.side, "id": res.get("id"), "status": res.get("status")})
        except Exception as e:  # noqa: BLE001 - record and keep going; the next run re-plans from actual positions
            record["submitted"].append({"symbol": o.symbol, "side": o.side, "error": str(e)[:200]})
    record["action"] = f"submitted {sum('id' in s for s in record['submitted'])} of {len(orders)} orders"
    print(json.dumps(record, indent=1))
    return 0 if all("id" in s for s in record["submitted"]) else 1


if __name__ == "__main__":
    sys.exit(main())
