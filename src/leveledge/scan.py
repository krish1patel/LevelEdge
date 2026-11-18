"""
Daily market scan: the v2 forecaster run over ~100 major US stocks and funds after each close,
with each result turned into plain-English insights.

For every ticker, ``analyze()`` fits the forecaster on its full daily history (walk-forward
validated, as on the dashboard) for the close 5 trading days ahead, and adds context from the
price history: trend against the 200-day average, how unusual recent swings are against the past
five years, distance from the 52-week high and recent returns. ``insights()`` turns those numbers
into short sentences; ``notable()`` ranks the most noteworthy across the universe.

The scan runs on AWS (EventBridge Scheduler -> dispatcher -> SQS -> parallel Lambda workers ->
DynamoDB; see app/worker.py and deploy/aws/scan.sh), or locally with scripts/market_scan.py.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .constants import US_EASTERN

HORIZON_DAYS = 5
SCHEMA = 1

# (ticker, plain name, group)
UNIVERSE: list[tuple[str, str, str]] = [
    ("SPY", "S&P 500 fund", "Funds"), ("QQQ", "Nasdaq-100 fund", "Funds"), ("IWM", "Small US companies fund", "Funds"),
    ("DIA", "Dow Jones fund", "Funds"), ("EFA", "Developed markets fund", "Funds"), ("EEM", "Emerging markets fund", "Funds"),
    ("TLT", "Long-term Treasury bond fund", "Funds"), ("IEF", "Mid-term Treasury bond fund", "Funds"), ("GLD", "Gold fund", "Funds"),
    ("USO", "Oil fund", "Funds"), ("XLE", "Energy sector fund", "Funds"), ("XLF", "Financials sector fund", "Funds"),
    ("AAPL", "Apple", "Technology"), ("MSFT", "Microsoft", "Technology"), ("NVDA", "Nvidia", "Technology"), ("AVGO", "Broadcom", "Technology"),
    ("ORCL", "Oracle", "Technology"), ("CRM", "Salesforce", "Technology"), ("ADBE", "Adobe", "Technology"), ("AMD", "AMD", "Technology"),
    ("CSCO", "Cisco", "Technology"), ("ACN", "Accenture", "Technology"), ("IBM", "IBM", "Technology"), ("INTU", "Intuit", "Technology"),
    ("QCOM", "Qualcomm", "Technology"), ("TXN", "Texas Instruments", "Technology"), ("AMAT", "Applied Materials", "Technology"),
    ("MU", "Micron", "Technology"), ("INTC", "Intel", "Technology"), ("PANW", "Palo Alto Networks", "Technology"), ("ADI", "Analog Devices", "Technology"),
    ("GOOGL", "Alphabet", "Communication"), ("META", "Meta", "Communication"), ("NFLX", "Netflix", "Communication"),
    ("DIS", "Disney", "Communication"), ("CMCSA", "Comcast", "Communication"), ("VZ", "Verizon", "Communication"), ("T", "AT&T", "Communication"),
    ("AMZN", "Amazon", "Consumer"), ("TSLA", "Tesla", "Consumer"), ("HD", "Home Depot", "Consumer"), ("MCD", "McDonald's", "Consumer"),
    ("NKE", "Nike", "Consumer"), ("SBUX", "Starbucks", "Consumer"), ("LOW", "Lowe's", "Consumer"), ("BKNG", "Booking Holdings", "Consumer"),
    ("TJX", "TJX", "Consumer"), ("UBER", "Uber", "Consumer"), ("WMT", "Walmart", "Staples"), ("COST", "Costco", "Staples"),
    ("PG", "Procter & Gamble", "Staples"), ("KO", "Coca-Cola", "Staples"), ("PEP", "PepsiCo", "Staples"), ("PM", "Philip Morris", "Staples"),
    ("MO", "Altria", "Staples"), ("JPM", "JPMorgan Chase", "Financials"), ("BAC", "Bank of America", "Financials"), ("WFC", "Wells Fargo", "Financials"),
    ("GS", "Goldman Sachs", "Financials"), ("MS", "Morgan Stanley", "Financials"), ("C", "Citigroup", "Financials"), ("BLK", "BlackRock", "Financials"),
    ("SCHW", "Charles Schwab", "Financials"), ("AXP", "American Express", "Financials"), ("V", "Visa", "Financials"), ("MA", "Mastercard", "Financials"),
    ("SPGI", "S&P Global", "Financials"), ("BRK-B", "Berkshire Hathaway", "Financials"), ("LLY", "Eli Lilly", "Health"), ("UNH", "UnitedHealth", "Health"),
    ("JNJ", "Johnson & Johnson", "Health"), ("MRK", "Merck", "Health"), ("ABBV", "AbbVie", "Health"), ("PFE", "Pfizer", "Health"),
    ("TMO", "Thermo Fisher", "Health"), ("ABT", "Abbott", "Health"), ("DHR", "Danaher", "Health"), ("AMGN", "Amgen", "Health"),
    ("ISRG", "Intuitive Surgical", "Health"), ("GILD", "Gilead", "Health"), ("VRTX", "Vertex", "Health"), ("BMY", "Bristol-Myers Squibb", "Health"),
    ("CVS", "CVS Health", "Health"), ("XOM", "Exxon Mobil", "Energy"), ("CVX", "Chevron", "Energy"), ("COP", "ConocoPhillips", "Energy"),
    ("CAT", "Caterpillar", "Industrials"), ("GE", "GE Aerospace", "Industrials"), ("HON", "Honeywell", "Industrials"), ("UNP", "Union Pacific", "Industrials"),
    ("BA", "Boeing", "Industrials"), ("RTX", "RTX", "Industrials"), ("LMT", "Lockheed Martin", "Industrials"), ("DE", "Deere", "Industrials"),
    ("UPS", "UPS", "Industrials"), ("LIN", "Linde", "Materials"), ("NEE", "NextEra Energy", "Utilities"), ("SO", "Southern Company", "Utilities"),
    ("DUK", "Duke Energy", "Utilities"), ("PLD", "Prologis", "Real estate"),
]
NAMES = {s: (n, g) for s, n, g in UNIVERSE}


def week_target(now: datetime | None = None) -> datetime:
    """The NYSE close HORIZON_DAYS sessions after the last completed one."""
    from .sessions import nyse_schedule

    now = now or datetime.now(tz=US_EASTERN)
    sched = nyse_schedule(now.date(), (pd.Timestamp(now.date()) + pd.Timedelta(days=21)).date())
    future = [t.to_pydatetime() for t in sched["market_close"] if t.to_pydatetime() > now]
    return future[HORIZON_DAYS - 1]


def context(close: pd.Series) -> dict:
    """Trend, unusual-volatility and drawdown context from daily closes (no model)."""
    c = close.dropna().astype(float)
    lr = np.log(c).diff().dropna()
    vol21 = lr.rolling(21).std() * np.sqrt(252)
    hist = vol21.dropna().iloc[-1260:]                       # about five years
    now_vol = float(vol21.iloc[-1])
    sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float("nan")
    hi52 = float(c.iloc[-252:].max())
    ret = lambda n: float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None   # noqa: E731
    return {
        "ret_1d": ret(1), "ret_1m": ret(21), "ret_12m": ret(252),
        "vol_now": now_vol, "vol_pct": float((hist < now_vol).mean()) if len(hist) else None,
        "sma200": sma200, "vs_sma200": float(c.iloc[-1] / sma200 - 1) if np.isfinite(sma200) else None,
        "from_high": float(c.iloc[-1] / hi52 - 1),
    }


def analyze(ticker: str, now: datetime | None = None, bars: pd.DataFrame | None = None) -> dict:
    """Forecast one ticker a week ahead and return the numbers and insights for the scan."""
    from .data import fetch_history
    from .predictor import forecast

    now = now or datetime.now(tz=US_EASTERN)
    bars = bars if bars is not None else fetch_history(ticker, "1d")
    target = week_target(now)
    m = forecast(ticker, target, "1d", data=bars)
    spot = float(m.current_price)
    q = m.quantiles((0.1, 0.5, 0.9))
    s = m.summary()
    name, group = NAMES.get(ticker, (ticker, "Other"))
    out = {
        "schema": SCHEMA, "ticker": ticker, "name": name, "group": group,
        "as_of": pd.Timestamp(bars.index[-1]).strftime("%Y-%m-%d"), "target": target.isoformat(),
        "price": round(spot, 4),
        "range80": [round(q[0.1], 4), round(q[0.9], 4)], "median": round(q[0.5], 4),
        "swing": round((q[0.9] - q[0.1]) / 2 / spot, 5),
        "p_up": round(float(m.prob_above([spot])[0]), 4),
        "coverage80": None if m.interval_coverage() is None else round(float(m.interval_coverage()), 4),
        "model_weight": round(float(s["weights"]["xgb"]), 4), "history_rows": int(s["rows"]),
        **{k: (None if v is None or not np.isfinite(v) else round(float(v), 5)) for k, v in context(bars["Close"]).items()},
    }
    out["insights"] = insights(out)
    return out


def pct(x: float, d: int = 0) -> str:
    return f"{abs(x) * 100:.{d}f}%"


def insights(r: dict) -> list[dict]:
    """Plain-English notes, each with a weight used to rank the most noteworthy across the scan."""
    notes: list[dict] = []

    def add(kind: str, weight: float, text: str):
        notes.append({"kind": kind, "weight": round(weight, 3), "text": text})

    vp, swing = r.get("vol_pct"), r["swing"]
    if vp is not None and vp >= 0.85:
        rare = "the biggest in at least five years" if vp >= 0.995 else f"bigger than {pct(vp)} of the past five years"
        add("volatile", vp, f"Swinging more than usual: its recent moves are {rare}. Expect about ±{pct(swing, 1)} over the next week.")
    elif vp is not None and vp <= 0.10:
        add("calm", 1 - vp, f"Unusually calm: recent moves are smaller than {pct(1 - vp)} of the past five years.")
    vs = r.get("vs_sma200")
    if vs is not None and abs(vs) <= 0.02:
        add("trend_edge", 0.8 - abs(vs) * 10, f"Within {pct(abs(vs), 1)} of its 200-day average, so its long-term trend could flip "
            f"{'down' if vs >= 0 else 'up'} soon.")
    fh = r.get("from_high")
    if fh is not None and fh <= -0.25:
        add("drawdown", min(1.0, -fh), f"{pct(fh)} below its 52-week high.")
    m1 = r.get("ret_1m")
    if m1 is not None and abs(m1) >= 0.15:
        add("big_month", min(1.0, abs(m1) * 2), f"{'Up' if m1 > 0 else 'Down'} {pct(m1)} in the past month.")
    p = r["p_up"]
    if p >= 0.6 or p <= 0.4:
        add("lean", abs(p - 0.5) * 2, f"The model leans {'up' if p > 0.5 else 'down'}: a {pct(p)} chance it's higher in a week "
            f"(most weeks it says 50–55%).")
    cov = r.get("coverage80")
    if cov is not None and cov < 0.7:
        add("caution", 0.3, f"Treat with care: in testing, its 8-in-10 ranges held only {pct(cov)} of the time.")
    if not notes:
        add("steady", 0.0, f"Nothing unusual. Expect about ±{pct(swing, 1)} over the next week.")
    return sorted(notes, key=lambda n: -n["weight"])


def notable(rows: list[dict], top: int = 8, per_kind: int = 3) -> list[dict]:
    """The most noteworthy insights across the scan: at most one per ticker and ``per_kind`` of each kind,
    so a day when many names are volatile still shows trend flips, big drops and model leans."""
    cands = sorted(({"ticker": r["ticker"], "name": r["name"], **n} for r in rows for n in r["insights"] if n["kind"] not in ("steady", "caution")),
                   key=lambda n: -n["weight"])
    out, seen, kinds = [], set(), {}
    for n in cands:
        if n["ticker"] in seen or kinds.get(n["kind"], 0) >= per_kind:
            continue
        out.append(n)
        seen.add(n["ticker"])
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
        if len(out) == top:
            break
    return out


def summary(rows: list[dict]) -> dict:
    """Universe-wide counts for the scan's key points."""
    ok = [r for r in rows if "swing" in r]
    up = [r for r in ok if (r.get("vs_sma200") or 0) > 0]
    return {
        "analyzed": len(ok), "in_uptrend": len(up),
        "volatile": sum(1 for r in ok if (r.get("vol_pct") or 0) >= 0.85),
        "median_swing": float(np.median([r["swing"] for r in ok])) if ok else None,
        "biggest_swing": max(ok, key=lambda r: r["swing"])["ticker"] if ok else None,
    }


__all__ = ["UNIVERSE", "NAMES", "HORIZON_DAYS", "week_target", "context", "analyze", "insights", "notable", "summary"]
