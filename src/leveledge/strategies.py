"""
Published allocation rules, implemented as their papers specify (no tuning).

* :func:`daa_g12` Keller & Keuning (2018) Defensive Asset Allocation, G12 universe.
* :func:`gtaa5`   Faber (2007) 10-month moving-average timing, five asset classes.

Both return month-end decision weights (rows = decision dates), to be traded at
the next open by :func:`leveledge.portfolio.backtest`.
"""
from __future__ import annotations

import pandas as pd

from leveledge import portfolio as P

DAA_RISKY = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD"]
DAA_CANARY = ["VWO", "BND"]
DAA_CASH = ["SHY", "IEF", "LQD"]
DAA_TOP = 6
DAA_BREADTH = 2
GTAA_ASSETS = ["SPY", "EFA", "IEF", "GSG", "VNQ"]
DAA_SYMBOLS = sorted(set(DAA_RISKY + DAA_CANARY + DAA_CASH))


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return P.rebalance_dates(idx, "monthly")


def mom_13612w(closes: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Keller's 13612W momentum: (12 r1 + 4 r3 + 2 r6 + r12) / 4 on month-end prices."""
    m = closes.loc[dates]
    r = lambda k: m / m.shift(k) - 1
    return (12 * r(1) + 4 * r(3) + 2 * r(6) + r(12)) / 4


def daa_g12(closes: pd.DataFrame) -> pd.DataFrame:
    """DAA-G12: top 6 risky by 13612W; cash share = (canaries with momentum <= 0) / 2,
    invested in the best of SHY / IEF / LQD."""
    me = month_ends(closes.index)
    mom = mom_13612w(closes, me)
    rows = {}
    for d in me:
        s = mom.loc[d]
        if s[DAA_SYMBOLS].isna().any():
            continue
        cf = min(1.0, int((s[DAA_CANARY] <= 0).sum()) / DAA_BREADTH)
        w = pd.Series(0.0, index=closes.columns)
        top = s[DAA_RISKY].sort_values(ascending=False).index[:DAA_TOP]
        w[top] += (1 - cf) / DAA_TOP
        if cf > 0:
            w[s[DAA_CASH].idxmax()] += cf
        rows[d] = w
    return pd.DataFrame(rows).T


def gtaa5(closes: pd.DataFrame) -> pd.DataFrame:
    """Faber GTAA: 20% per asset class while its month-end price is above its 10-month average."""
    me = month_ends(closes.index)
    m = closes.loc[me, GTAA_ASSETS]
    sma = m.rolling(10).mean()
    rows = {}
    for d in me:
        if sma.loc[d].isna().any():
            continue
        w = pd.Series(0.0, index=closes.columns)
        on = m.loc[d] > sma.loc[d]
        w[on[on].index] = 0.2
        rows[d] = w
    return pd.DataFrame(rows).T


__all__ = ["daa_g12", "gtaa5", "mom_13612w", "month_ends", "DAA_SYMBOLS", "DAA_RISKY", "DAA_CANARY", "DAA_CASH", "GTAA_ASSETS"]
