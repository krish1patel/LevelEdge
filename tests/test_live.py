from datetime import datetime

import pandas as pd
import pytest

from leveledge import live
from leveledge.broker import AlpacaBroker, LiveTradingRefused, plan_orders
from leveledge.constants import US_EASTERN as ET


def test_book_combines_core_and_sleeve_at_leverage():
    trend = pd.Series({"SPY": 0.2, "IEF": 0.1, "GLD": 0.25, "TLT": 0.0})
    book = live.book_from_trend(trend)
    L = live.LEVERAGE
    assert book["SPY"] == pytest.approx(L * (0.7 * 0.6 + 0.3 * 0.2))
    assert book["IEF"] == pytest.approx(L * (0.7 * 0.4 + 0.3 * 0.1))
    assert book["GLD"] == pytest.approx(L * 0.3 * 0.25)
    assert book.sum() <= live.MAX_GROSS + 1e-12


def test_rebalance_day_is_last_trading_day_of_week():
    assert live.is_rebalance_day(pd.Timestamp("2026-11-06"))        # Friday
    assert not live.is_rebalance_day(pd.Timestamp("2026-11-05"))    # Thursday
    assert live.is_rebalance_day(pd.Timestamp("2026-04-02"))        # Thursday before Good Friday 2026


def test_partial_daily_bar_is_dropped_before_close():
    idx = pd.DatetimeIndex(["2026-11-06", "2026-11-09"])
    prices = {"SPY": pd.DataFrame({"Close": [1.0, 2.0]}, index=idx)}
    during = live.completed_daily(prices, datetime(2026, 11, 9, 13, 0, tzinfo=ET))
    after = live.completed_daily(prices, datetime(2026, 11, 9, 16, 30, tzinfo=ET))
    assert len(during["SPY"]) == 1 and len(after["SPY"]) == 2


def test_broker_refuses_live_endpoint(monkeypatch):
    monkeypatch.delenv("LEVELEDGE_ALLOW_LIVE_TRADING", raising=False)
    with pytest.raises(LiveTradingRefused):
        AlpacaBroker("k", "s", base_url="https://api.alpaca.markets")
    assert AlpacaBroker("k", "s").is_paper


def test_credentials_accept_both_naming_styles(monkeypatch):
    from leveledge.broker import KEY_NAMES, SECRET_NAMES, credentials
    for n in KEY_NAMES + SECRET_NAMES:
        monkeypatch.delenv(n, raising=False)
    assert credentials() is None
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    assert credentials() == ("k", "s")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k2")
    assert credentials()[0] == "k2"


def test_plan_never_borrows():
    orders = plan_orders({"SPY": 0.6, "IEF": 0.4}, equity=10_000, cash=1_000,
                         positions={"GLD": {"market_value": 2_000, "qty": 10}},
                         prices={"SPY": 600, "IEF": 95, "GLD": 200}, fractionable={})
    spend = sum(o.notional for o in orders if o.side == "buy")
    assert spend <= 1_000 + 2_000 + 1e-6               # cash + sale proceeds only
    assert orders[0].side == "sell" and orders[0].symbol == "GLD"


def test_plan_skips_small_drift_and_exits_fully():
    orders = plan_orders({"SPY": 0.6, "IEF": 0.4}, equity=10_000, cash=0,
                         positions={"SPY": {"market_value": 6_050, "qty": 10}, "IEF": {"market_value": 3_950, "qty": 40},
                                    "TLT": {"market_value": 1, "qty": 0.01}},
                         prices={"SPY": 605, "IEF": 98.75, "TLT": 100}, fractionable={})
    assert [(o.symbol, o.side, o.qty) for o in orders] == [("TLT", "sell", 0.01)]


def test_plan_borrows_only_up_to_max_gross():
    orders = plan_orders({"SPY": 0.9, "IEF": 0.6}, equity=10_000, cash=10_000, positions={},
                         prices={"SPY": 600, "IEF": 95}, fractionable={}, max_gross=1.5)
    spend = sum(o.notional for o in orders if o.side == "buy")
    assert 10_000 < spend <= 15_000 + 1e-6            # uses margin, but never beyond 1.5x equity
    capped = plan_orders({"SPY": 1.5, "IEF": 0.6}, equity=10_000, cash=10_000, positions={},
                         prices={"SPY": 600, "IEF": 95}, fractionable={}, max_gross=1.6)
    assert sum(o.notional for o in capped if o.side == "buy") <= 16_000 + 1e-6


def test_plan_for_a_100_dollar_account_never_borrows_and_respects_the_band():
    targets = {"SPY": 0.449, "IEF": 0.28, "VNQ": 0.033, "EFA": 0.032, "DBC": 0.03, "IWM": 0.026, "EEM": 0.021, "QQQ": 0.019}
    orders = plan_orders(targets, equity=100.0, cash=100.0, positions={}, prices={}, fractionable={}, band=live.BAND,
                         max_gross=live.MAX_GROSS)
    bought = {o.symbol for o in orders}
    assert bought == {s for s, w in targets.items() if w >= live.BAND}   # QQQ at 1.9% waits until it clears the 2% band
    assert sum(o.notional for o in orders) <= 100.0
    assert all(o.notional >= 1.0 for o in orders)                         # Alpaca's fractional minimum


def test_plan_whole_shares_when_not_fractionable():
    orders = plan_orders({"DBC": 0.2}, equity=10_000, cash=10_000, positions={}, prices={"DBC": 23.0},
                         fractionable={"DBC": False})
    assert orders[0].qty == 86.0 and orders[0].notional is None
