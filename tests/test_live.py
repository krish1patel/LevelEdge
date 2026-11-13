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


def test_plan_whole_shares_when_not_fractionable():
    orders = plan_orders({"DBC": 0.2}, equity=10_000, cash=10_000, positions={}, prices={"DBC": 23.0},
                         fractionable={"DBC": False})
    assert orders[0].qty == 86.0 and orders[0].notional is None


def test_slippage_sign_convention():
    from leveledge.broker import slippage_bp
    assert slippage_bp("buy", 100.10, 100.0) == pytest.approx(10.0)     # paid up: a cost
    assert slippage_bp("sell", 99.90, 100.0) == pytest.approx(10.0)     # sold low: a cost
    assert slippage_bp("buy", 99.95, 100.0) == pytest.approx(-5.0)      # better than the open


def test_margin_interest_estimate():
    from leveledge.broker import margin_interest
    assert margin_interest(36_500, 10, 0.04, 0.02) == pytest.approx(60.0)
    assert margin_interest(-5, 10, 0.04) == 0.0


def test_trend_flip_price_matches_the_signal():
    import numpy as np
    from leveledge import portfolio as P
    idx = pd.bdate_range("2024-01-01", periods=300)
    closes = pd.DataFrame({"A": np.linspace(100, 110, 300)}, index=idx)
    cash = pd.Series(0.0001, index=idx)
    flip = live.trend_flip_prices(closes, cash, 252)["A"]
    sig = P.momentum_signal(closes, cash, 252)["A"].iloc[-1]
    assert sig == 1.0 and closes["A"].iloc[-1] > flip                # on, and price is above the flip level
    bumped = closes.copy()
    bumped.iloc[-1, 0] = flip * 0.999                                  # just below the flip price
    assert P.momentum_signal(bumped, cash, 252)["A"].iloc[-1] == 0.0


def test_plan_repays_margin_with_whole_share_sells():
    # 1.3x invested, cash -30k; a no-borrowing book trims every overweight ETF, even inside the band
    pos = {"SPY": {"market_value": 78_000, "qty": 104}, "IEF": {"market_value": 52_000, "qty": 520},
           "QQQ": {"market_value": 1_800, "qty": 3}}
    orders = plan_orders({"SPY": 0.588, "IEF": 0.4, "QQQ": 0.012}, equity=100_000, cash=-31_800, positions=pos,
                         prices={"SPY": 750, "IEF": 100, "QQQ": 600}, fractionable={"SPY": False, "IEF": False, "QQQ": False},
                         band=0.02, max_gross=1.0)
    assert all(o.side == "sell" and float(o.qty).is_integer() for o in orders)
    assert {o.symbol for o in orders} == {"SPY", "IEF", "QQQ"}       # QQQ's $600 drift is inside the band
    proceeds = sum(o.qty * {"SPY": 750, "IEF": 100, "QQQ": 600}[o.symbol] for o in orders)
    assert proceeds >= 31_800


def test_fidelity_positions_and_ticket():
    from leveledge import mirror
    csv_text = ("﻿Account Number,Account Name,Symbol,Description,Quantity,Last Price,Last Price Change,Current Value,Type\n"
                "Z1,Individual,SPAXX**,MONEY MARKET,,,,$5000.00,Cash\n"
                "Z1,Individual,SPY,SPDR S&P500,10,$750.00,,\"$7,500.00\",Cash\n"
                "Z1,Individual,Pending Activity,,,,,-$100.00,\n\n"
                "\"The data and information in this spreadsheet is provided to you solely for your use\"\n")
    pos, cash = mirror.parse_fidelity_positions(csv_text)
    assert pos == {"SPY": {"qty": 10.0, "market_value": 7500.0}} and cash == 4900.0
    orders, summ = mirror.ticket({"SPY": 0.6, "IEF": 0.4}, {"SPY": 750.0, "IEF": 100.0}, pos, cash, band=0.02)
    assert all(float(o.qty).is_integer() for o in orders)
    assert summ["cash_after"] >= 0
    assert abs(summ["weights_after"]["IEF"] - 0.4) < 0.02         # whole shares and a 0.5% cash buffer


def test_fidelity_file_with_two_accounts_needs_a_choice():
    import pytest
    from leveledge import mirror
    txt = ("Account Number,Account Name,Symbol,Quantity,Current Value\n"
           "Z1,Individual,SPY,1,$750\nZ2,Roth IRA,SPY,2,$1500\n")
    with pytest.raises(ValueError):
        mirror.parse_fidelity_positions(txt)
    assert mirror.parse_fidelity_positions(txt, "Roth IRA")[0]["SPY"]["qty"] == 2.0
