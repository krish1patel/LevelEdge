import numpy as np
import pandas as pd

from leveledge import strategies as S


def _closes(trends: dict[str, float], days: int = 420) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=days)
    t = np.arange(days)
    return pd.DataFrame({s: 100 * np.exp(g * t / 252) for s, g in trends.items()}, index=idx)


def _all(up: float, canary: float) -> dict[str, float]:
    d = {s: up for s in S.DAA_SYMBOLS}
    for c in S.DAA_CANARY:
        d[c] = canary
    d["SHY"], d["IEF"], d["LQD"] = 0.01, 0.03, 0.02
    return d


def test_daa_fully_invested_when_canaries_positive():
    w = S.daa_g12(_closes(_all(0.2, 0.1))).iloc[-1]
    assert abs(w.sum() - 1) < 1e-9
    risky = w[S.DAA_RISKY]
    assert (risky > 0).sum() == S.DAA_TOP


def test_daa_all_cash_when_both_canaries_negative():
    w = S.daa_g12(_closes(_all(0.2, -0.3))).iloc[-1]
    assert abs(w.sum() - 1) < 1e-9
    assert w["IEF"] == 1.0                 # best cash asset by momentum
    assert w[[s for s in S.DAA_RISKY if s != "LQD"]].sum() == 0


def test_daa_half_cash_with_one_bad_canary():
    d = _all(0.2, 0.1)
    d["BND"] = -0.3
    w = S.daa_g12(_closes(d)).iloc[-1]
    cash_part = w[S.DAA_CASH].max()
    assert abs(cash_part - 0.5) < 0.5 / S.DAA_TOP + 1e-9   # half to cash (LQD may also be a risky pick)
    assert abs(w.sum() - 1) < 1e-9


def test_gtaa_holds_only_assets_above_their_average():
    d = {s: 0.2 for s in S.GTAA_ASSETS}
    d["GSG"] = -0.4
    w = S.gtaa5(_closes(d)).iloc[-1]
    assert w["GSG"] == 0 and abs(w.sum() - 0.8) < 1e-9
