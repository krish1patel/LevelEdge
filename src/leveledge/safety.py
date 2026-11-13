"""
Pre-trade safety checks for the paper book. Any failed check halts the run before an
order is sent; the workflow then posts an alert. Each check catches a failure that
would otherwise trade silently on bad inputs:

    account_ok       account active, trading not blocked, and a paper account
    data_fresh       targets come from the last completed NYSE session (no stale prices)
    moves_plausible  no ETF moved more than 15% in a day (bad data, or a day for a human)
    prices_agree     the yfinance close matches Alpaca's price for every held ETF (within 3%)
    order_sizes      no single order above 90% of equity; total traded below 2x equity
    gross_ok         planned holdings stay within the book's gross ceiling
    not_duplicate    this decision's orders were not already sent (re-run protection)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import pandas as pd

MAX_DAY_MOVE = 0.15
MAX_PRICE_GAP = 0.03
MAX_ORDER_SHARE = 0.90
MAX_TURNOVER = 2.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def last_completed_session(now: datetime) -> pd.Timestamp:
    from .sessions import nyse_schedule

    today = pd.Timestamp(now.date())
    sched = nyse_schedule(today - timedelta(days=10), today)
    done = sched[sched["market_close"] + pd.Timedelta(minutes=15) <= pd.Timestamp(now)]
    return pd.Timestamp(done.index[-1]).normalize()


def pretrade_checks(targets: dict, account: dict, is_paper: bool, positions: dict, alpaca_prices: dict,
                    orders: list, equity: float, max_gross: float, recent_client_ids: list[str],
                    client_prefix: str, now: datetime) -> list[Check]:
    out: list[Check] = []

    blocked = [k for k in ("trading_blocked", "account_blocked", "trade_suspended_by_user") if account.get(k)]
    status = str(account.get("status", "ACTIVE")).upper()
    out.append(Check("account_ok", is_paper and status == "ACTIVE" and not blocked,
                     f"paper={is_paper}, status={status}" + (f", blocked: {', '.join(blocked)}" if blocked else "")))

    want = last_completed_session(now)
    got = pd.Timestamp(targets["as_of_close"])
    out.append(Check("data_fresh", got == want, f"targets from the {got.date()} close; last completed session {want.date()}"))

    big = {a["symbol"]: a.get("ret_1d") for a in targets["assets"]
           if a.get("ret_1d") is not None and abs(a["ret_1d"]) > MAX_DAY_MOVE}
    out.append(Check("moves_plausible", not big,
                     "largest one-day move within ±15%" if not big else "implausible moves: " + ", ".join(f"{s} {v:+.1%}" for s, v in big.items())))

    closes = {a["symbol"]: a["last_close"] for a in targets["assets"]}
    gaps = {s: alpaca_prices[s] / closes[s] - 1 for s in alpaca_prices if s in closes and closes[s] > 0}
    bad = {s: g for s, g in gaps.items() if abs(g) > MAX_PRICE_GAP}
    out.append(Check("prices_agree", not bad,
                     (f"{len(gaps)} held ETFs within ±3% of Alpaca" if gaps else "no held ETFs to compare") if not bad
                     else "price sources disagree: " + ", ".join(f"{s} {g:+.1%}" for s, g in bad.items())))

    notional = []
    for o in orders:
        n = o.notional if getattr(o, "notional", None) is not None else (o.qty or 0) * closes.get(o.symbol, 0.0)
        notional.append(abs(float(n)))
    biggest = max(notional, default=0.0)
    total = sum(notional)
    out.append(Check("order_sizes", biggest <= MAX_ORDER_SHARE * equity and total <= MAX_TURNOVER * equity,
                     f"{len(orders)} orders, largest ${biggest:,.0f} ({biggest / equity:.0%} of equity), total ${total:,.0f} ({total / equity:.0%})"))

    after = {s: float(p.get("market_value", 0.0)) for s, p in positions.items()}
    for o, n in zip(orders, notional):
        after[o.symbol] = after.get(o.symbol, 0.0) + (n if o.side == "buy" else -n)
    gross = sum(max(v, 0.0) for v in after.values()) / equity if equity > 0 else 0.0
    out.append(Check("gross_ok", gross <= max_gross + 1e-6, f"holdings after orders {gross:.2f}x equity (ceiling {max_gross:.2f}x)"))

    dup = [c for c in recent_client_ids if c.startswith(client_prefix)]
    out.append(Check("not_duplicate", not dup, "no orders yet for this decision" if not dup else f"{len(dup)} orders already sent for this decision"))
    return out


def summarize(checks: list[Check]) -> dict:
    failed = [c for c in checks if not c.ok]
    return {"passed": not failed, "failed": [c.name for c in failed], "checks": [asdict(c) for c in checks]}
