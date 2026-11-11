"""
Alpaca brokerage: paper trading by default, never live unless explicitly allowed.

Safety rules enforced here, not just documented:
* the client refuses any non-paper endpoint unless LEVELEDGE_ALLOW_LIVE_TRADING=1;
* :func:`plan_orders` never takes total long exposure above ``max_gross`` times
  equity (1.0 = no borrowing; the book uses 1.5, under Alpaca's 2x overnight limit);
* trades smaller than the band are skipped to keep turnover and costs down.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import requests

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class LiveTradingRefused(RuntimeError):
    pass


@dataclass
class Order:
    symbol: str
    side: str                 # "buy" | "sell"
    notional: float | None    # dollars (fractionable assets)
    qty: float | None         # shares (whole-share assets, or full exits)
    reason: str

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


KEY_NAMES = ("ALPACA_API_KEY_ID", "APCA_API_KEY_ID", "ALPACA_API_KEY")
SECRET_NAMES = ("ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY")


def credentials() -> tuple[str, str] | None:
    """Alpaca keys from the environment, accepting the common naming styles."""
    key = next((os.environ[n] for n in KEY_NAMES if os.environ.get(n)), None)
    secret = next((os.environ[n] for n in SECRET_NAMES if os.environ.get(n)), None)
    return (key, secret) if key and secret else None


class AlpacaBroker:
    def __init__(self, key: str, secret: str, base_url: str | None = None, timeout: float = 20.0):
        self.base_url = (base_url or os.environ.get("ALPACA_BASE_URL") or PAPER_URL).rstrip("/")
        if "paper-api" not in self.base_url and os.environ.get("LEVELEDGE_ALLOW_LIVE_TRADING") != "1":
            raise LiveTradingRefused(
                f"{self.base_url} is not Alpaca's paper endpoint. LevelEdge trades paper money only "
                "unless LEVELEDGE_ALLOW_LIVE_TRADING=1 is set deliberately.")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.timeout = timeout

    @property
    def is_paper(self) -> bool:
        return "paper-api" in self.base_url

    def _req(self, method: str, path: str, **kw):
        r = self.s.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    # ---- reads ---------------------------------------------------------------
    def account(self) -> dict:
        return self._req("GET", "/v2/account")

    def clock(self) -> dict:
        return self._req("GET", "/v2/clock")

    def positions(self) -> list[dict]:
        return self._req("GET", "/v2/positions")

    def asset(self, symbol: str) -> dict:
        return self._req("GET", f"/v2/assets/{symbol}")

    def orders(self, status: str = "all", limit: int = 200) -> list[dict]:
        return self._req("GET", "/v2/orders", params={"status": status, "limit": limit, "direction": "desc"})

    def portfolio_history(self, period: str = "1A", timeframe: str = "1D") -> dict:
        return self._req("GET", "/v2/account/portfolio/history", params={"period": period, "timeframe": timeframe})

    # ---- writes --------------------------------------------------------------
    def cancel_open_orders(self) -> None:
        self._req("DELETE", "/v2/orders")

    def submit(self, o: Order, client_order_id: str | None = None) -> dict:
        body = {"symbol": o.symbol, "side": o.side, "type": "market", "time_in_force": "day"}
        if o.notional is not None:
            body["notional"] = f"{o.notional:.2f}"
        else:
            body["qty"] = str(o.qty)
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._req("POST", "/v2/orders", json=body)


def plan_orders(targets: dict[str, float], equity: float, cash: float, positions: dict[str, dict],
                prices: dict[str, float], fractionable: dict[str, bool], band: float = 0.01,
                min_notional: float = 1.0, max_gross: float = 1.0) -> list[Order]:
    """Orders that move the account toward ``targets`` (weights of equity).

    positions : symbol -> {"market_value": $, "qty": shares}
    band      : skip a symbol if its weight is within ``band`` of target (unless exiting fully)
    Sells come first; buys are scaled down so total long exposure never exceeds
    ``max_gross`` x equity (with max_gross = 1 they never exceed cash plus sale proceeds).
    """
    if equity <= 0:
        return []
    sells, buys = [], []
    symbols = sorted(set(targets) | set(positions))
    for s in symbols:
        tgt = max(0.0, targets.get(s, 0.0)) * equity
        pos = positions.get(s, {})
        cur = float(pos.get("market_value", 0.0))
        diff = tgt - cur
        if tgt == 0 and cur > 0:
            sells.append(Order(s, "sell", None, float(pos.get("qty", 0.0)), "exit: target is zero"))
            continue
        if abs(diff) < band * equity or abs(diff) < min_notional:
            continue
        px = prices.get(s)
        if diff < 0:
            if fractionable.get(s, True):
                sells.append(Order(s, "sell", round(-diff, 2), None, f"trim to {tgt / equity:.1%}"))
            elif px:
                q = math.floor(-diff / px)
                if q > 0:
                    sells.append(Order(s, "sell", None, float(q), f"trim to {tgt / equity:.1%}"))
        else:
            buys.append((s, diff, px))
    proceeds = sum(o.notional if o.notional is not None else (o.qty or 0) * prices.get(o.symbol, 0.0) for o in sells)
    long_after_sells = sum(float(p.get("market_value", 0.0)) for p in positions.values()) - proceeds
    budget = max(0.0, max_gross * equity - long_after_sells) * 0.995   # buffer for price moves before the fill
    if max_gross <= 1.0:
        budget = min(budget, max(0.0, cash + proceeds) * 0.995)
    want = sum(d for _, d, _ in buys)
    scale = min(1.0, budget / want) if want > 0 else 1.0
    note = "" if scale == 1 else " (cash-limited)"
    planned_buys = []
    for s, d, px in buys:
        amt = d * scale
        if amt < min_notional:
            continue
        if fractionable.get(s, True):
            planned_buys.append(Order(s, "buy", round(amt, 2), None, f"add to {targets[s]:.1%}{note}"))
        elif px and math.floor(amt / px) > 0:
            planned_buys.append(Order(s, "buy", None, float(math.floor(amt / px)), f"add to {targets[s]:.1%}{note}"))
    return sells + planned_buys


def slippage_bp(side: str, fill: float, reference: float) -> float:
    """Execution cost of a fill against a reference price, in basis points.

    Positive means the fill was worse than the reference (paid more on a buy,
    received less on a sell); negative means it was better.
    """
    if not reference or reference <= 0:
        return float("nan")
    sign = 1.0 if side == "buy" else -1.0
    return sign * (fill - reference) / reference * 1e4


def margin_interest(borrowed: float, days: float, tbill_rate: float, spread: float = 0.02) -> float:
    """Estimated margin interest in dollars: borrowed x (T-bill + spread) x days / 365."""
    return max(0.0, borrowed) * (tbill_rate + spread) * max(0.0, days) / 365.0


__all__ = ["AlpacaBroker", "Order", "plan_orders", "credentials", "LiveTradingRefused", "PAPER_URL",
           "slippage_bp", "margin_interest"]
