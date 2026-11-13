#!/usr/bin/env python3
"""
Print the exact orders that bring a real brokerage account (Fidelity) to the
LevelEdge book's current targets: whole shares, sells first, no margin.

Usage:
  python scripts/mirror_ticket.py --positions Portfolio_Positions_Nov-13-2026.csv [--account Z12345678]
  python scripts/mirror_ticket.py --cash 25000          # a new account holding only cash

Fidelity: Accounts & Trade -> Portfolio -> Positions -> Download.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from leveledge import live, mirror  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--positions", help="Fidelity positions CSV export")
    src.add_argument("--cash", type=float, help="value of an account that holds only cash")
    ap.add_argument("--account", help="account number or name, when the export holds several accounts")
    ap.add_argument("--band", type=float, default=live.BAND)
    args = ap.parse_args()

    if args.positions:
        with open(args.positions, encoding="utf-8-sig") as f:
            positions, cash = mirror.parse_fidelity_positions(f.read(), args.account)
    else:
        positions, cash = {}, args.cash
    t = live.current_targets()
    prices = {a["symbol"]: a["last_close"] for a in t["assets"]}
    missing = [s for s in positions if s not in prices]
    if missing:
        print(f"Holdings outside the book ({', '.join(missing)}) are left alone; the book is sized on cash plus its own ETFs.\n")
    # value the book's ETFs at the same closes the orders are planned on
    orders, summary = mirror.ticket(t["weights"], prices,
                                    {s: {"qty": p["qty"], "market_value": p["qty"] * prices[s]} for s, p in positions.items() if s in prices},
                                    cash, args.band)
    print(mirror.format_ticket(orders, summary, t["weights"], prices, t["as_of_close"], t["rebalance_today"],
                               t["next_rebalance"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
