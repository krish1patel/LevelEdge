"""
Mirror the paper book in a real brokerage account (built for Fidelity).

The paper account's own orders only fit an account of the same size and the same
holdings. A real account drifts apart (dividends, deposits, different fills), so the
ticket is planned from the real account's own positions: the same targets, the same
no-trade band, whole shares, sells before buys, never borrowing.

Fidelity: Positions -> Download (CSV). Money-market sweep rows (symbols ending in
"**", e.g. SPAXX**) and "Pending Activity" count as cash.
"""
from __future__ import annotations

import csv
import io

from .broker import Order, plan_orders


def _num(x) -> float:
    s = str(x or "").strip().replace("$", "").replace(",", "").replace("+", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_fidelity_positions(text: str, account: str | None = None) -> tuple[dict[str, dict], float]:
    """Positions {symbol: {"qty", "market_value"}} and cash from a Fidelity positions CSV.

    ``account`` filters by account number or name when the file holds several accounts.
    """
    lines = text.lstrip("﻿").splitlines()
    start = next((i for i, ln in enumerate(lines) if "Symbol" in ln and "Quantity" in ln), None)
    if start is None:
        raise ValueError("not a Fidelity positions file: no header with Symbol and Quantity")
    positions: dict[str, dict] = {}
    cash = 0.0
    accounts = set()
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        sym = str(row.get("Symbol") or "").strip()
        if not sym or row.get("Current Value") is None:
            continue                                         # footer disclaimers, blank lines
        acct_ids = {str(row.get("Account Number") or "").strip(), str(row.get("Account Name") or "").strip()}
        accounts.add(str(row.get("Account Number") or "").strip())
        if account and account not in acct_ids:
            continue
        value = _num(row.get("Current Value"))
        if sym.endswith("**") or sym.lower().startswith("pending"):
            cash += value
            continue
        p = positions.setdefault(sym.upper(), {"qty": 0.0, "market_value": 0.0})
        p["qty"] += _num(row.get("Quantity"))
        p["market_value"] += value
    if not account and len(accounts - {""}) > 1:
        raise ValueError(f"the file holds {len(accounts - {''})} accounts; pick one with --account")
    return positions, cash


def ticket(targets: dict[str, float], prices: dict[str, float], positions: dict[str, dict], cash: float,
           band: float) -> tuple[list[Order], dict]:
    """Whole-share orders that move a no-borrowing account toward ``targets``."""
    equity = cash + sum(p["market_value"] for p in positions.values())
    orders = plan_orders(targets, equity, cash, positions, prices, {s: False for s in set(targets) | set(positions)},
                         band=band, max_gross=1.0)
    after = {s: p["qty"] for s, p in positions.items()}
    flow = 0.0
    for o in orders:
        q = o.qty or 0.0
        after[o.symbol] = after.get(o.symbol, 0.0) + (q if o.side == "buy" else -q)
        flow += (-1 if o.side == "buy" else 1) * q * prices.get(o.symbol, 0.0)
    held = {s: q * prices.get(s, 0.0) for s, q in after.items() if q > 1e-9}
    summary = {"equity": equity, "cash_before": cash, "cash_after": cash + flow,
               "weights_after": {s: v / equity for s, v in held.items()} if equity > 0 else {}}
    return orders, summary


def format_ticket(orders: list[Order], summary: dict, targets: dict[str, float], prices: dict[str, float],
                  as_of: str, rebalance_today: bool, next_rebalance: str) -> str:
    out = [f"# Orders to mirror the LevelEdge book (targets from the {as_of} close)", ""]
    if not rebalance_today:
        out += [f"Note: the paper account trades only on the weekly rebalance day (next: {next_rebalance}). "
                "This ticket uses the latest close anyway.", ""]
    out += [f"Money in the book: ${summary['equity']:,.2f} (cash ${summary['cash_before']:,.2f} plus the book's ETFs).", ""]
    if not orders:
        out += ["No trades: every ETF is within the no-trade band of its target."]
    else:
        out += ["Enter the sells first, then the buys. Market orders, day, whole shares; placed outside market "
                "hours they fill at the next open, like the paper account's.", "",
                "| # | Action | Shares | Symbol | Est. price | Est. amount |", "|---|---|---|---|---|---|"]
        for i, o in enumerate(orders, 1):
            q = o.qty or 0.0
            px = prices.get(o.symbol, 0.0)
            qs = f"{q:,.0f}" if float(q).is_integer() else f"{q:,.3f}"
            out.append(f"| {i} | {o.side.upper()} | {qs} | {o.symbol} | ${px:,.2f} | ${q * px:,.2f} |")
        out += ["", f"Cash after the orders: about ${summary['cash_after']:,.2f} (never negative: no margin)."]
    rows = sorted(set(targets) | set(summary["weights_after"]), key=lambda s: -targets.get(s, 0.0))
    out += ["", "| ETF | Target | After orders |", "|---|---|---|"]
    out += [f"| {s} | {targets.get(s, 0.0):.1%} | {summary['weights_after'].get(s, 0.0):.1%} |"
            for s in rows if targets.get(s, 0.0) > 0 or summary["weights_after"].get(s, 0.0) > 0]
    out += ["", "Prices are the last close; fills at the open will differ a little. Under the band the book "
            "leaves small drift alone, the same rule the paper account follows."]
    return "\n".join(out)


__all__ = ["parse_fidelity_positions", "ticket", "format_ticket"]
