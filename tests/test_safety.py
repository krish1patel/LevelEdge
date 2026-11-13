from datetime import datetime

import pandas as pd

from leveledge import outlook, safety
from leveledge.broker import Order
from leveledge.constants import US_EASTERN

NOW = datetime(2026, 11, 13, 16, 40, tzinfo=US_EASTERN)       # Friday after the close


def _targets(as_of="2026-11-13", ret_1d=0.004):
    return {"as_of_close": as_of, "assets": [{"symbol": "SPY", "last_close": 750.0, "ret_1d": ret_1d},
                                             {"symbol": "IEF", "last_close": 90.0, "ret_1d": -0.002}]}


def _run(**kw):
    args = dict(targets=_targets(), account={"status": "ACTIVE"}, is_paper=True,
                positions={"SPY": {"market_value": 60_000.0}, "IEF": {"market_value": 40_000.0}},
                alpaca_prices={"SPY": 750.4, "IEF": 89.95},
                orders=[Order("SPY", "buy", 2_000.0, None, "drift")], equity=100_000.0, max_gross=1.6,
                recent_client_ids=["core6040_trend30_x150_v2-2026-11-06-0"],
                client_prefix="core6040_trend30_x150_v2-2026-11-13-", now=NOW)
    args.update(kw)
    return {c.name: c for c in safety.pretrade_checks(**args)}


def test_all_pass_on_normal_inputs():
    checks = _run()
    assert all(c.ok for c in checks.values()), {k: c.detail for k, c in checks.items() if not c.ok}


def test_stale_data_halts():
    assert not _run(targets=_targets(as_of="2026-11-12"))["data_fresh"].ok


def test_last_session_before_the_close_is_yesterday():
    assert safety.last_completed_session(datetime(2026, 11, 13, 14, 0, tzinfo=US_EASTERN)) == pd.Timestamp("2026-11-12")


def test_implausible_move_and_price_disagreement_halt():
    assert not _run(targets=_targets(ret_1d=-0.30))["moves_plausible"].ok
    assert not _run(alpaca_prices={"SPY": 700.0})["prices_agree"].ok


def test_blocked_or_live_account_halts():
    assert not _run(account={"status": "ACTIVE", "trading_blocked": True})["account_ok"].ok
    assert not _run(is_paper=False)["account_ok"].ok


def test_oversized_orders_and_gross_halt():
    assert not _run(orders=[Order("SPY", "buy", 95_000.0, None, "x")])["order_sizes"].ok
    assert not _run(orders=[Order("SPY", "buy", 70_000.0, None, "x")])["gross_ok"].ok


def test_duplicate_run_halts():
    assert not _run(recent_client_ids=["core6040_trend30_x150_v2-2026-11-13-0"])["not_duplicate"].ok


def test_forward_scoring_counts_band_hits():
    q = {"q025": -0.05, "q050": -0.04, "q100": -0.03, "q250": -0.01, "q500": 0.0, "q750": 0.01,
         "q900": 0.03, "q950": 0.04, "q975": 0.05}
    entries = [{"as_of_close": "2026-11-02", "spot": 100.0,
                "horizons": [{"h": 5, "target_date": "2026-11-09", "ret": q, "p_up": 0.55, "p_down5": 0.02},
                             {"h": 21, "target_date": "2026-12-01", "ret": q, "p_up": 0.6, "p_down5": 0.1}]},
               {"as_of_close": "2026-11-03", "spot": 100.0,
                "horizons": [{"h": 5, "target_date": "2026-11-10", "ret": q, "p_up": 0.55, "p_down5": 0.02}]}]
    closes = pd.Series([102.0, 95.0], index=pd.to_datetime(["2026-11-09", "2026-11-10"]))
    r = outlook.score_forward(entries, closes)
    h5 = r["horizons"]["5"]
    assert h5["n"] == 2 and h5["cover80"] == 0.5 and h5["cover95"] == 0.5      # +2% inside, -5% on the edge
    assert r["horizons"]["21"]["n"] == 0 and r["next_result"] == "2026-12-01"


def test_target_dates_skip_weekends_and_holidays():
    d = outlook.target_dates("2026-11-20", (5,))      # Thanksgiving week: Nov 26 closed
    assert d[5] == "2026-11-30"
