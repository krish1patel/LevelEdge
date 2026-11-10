from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from leveledge.constants import US_EASTERN as ET
from leveledge.edge import evaluate_quotes, kelly_single, size_event
from leveledge.kalshi import markets_frame, quote_at, taker_fee


def test_taker_fee_peaks_at_half():
    assert taker_fee(0.5) == pytest.approx(0.0175)
    assert taker_fee(0.0) == 0 and taker_fee(1.0) == 0
    assert taker_fee(0.1) == pytest.approx(taker_fee(0.9))


def test_edge_signs_and_costs():
    # model 0.70, market 0.60/0.62: YES is cheap
    q = evaluate_quotes([0.70], [0.60], [0.62], fee_rate=0.07).iloc[0]
    assert q.side == "yes"
    assert q.cost == pytest.approx(0.62 + 0.07 * 0.62 * 0.38)
    assert q.edge == pytest.approx(0.70 - q.cost)
    # model 0.40, market 0.60/0.62: NO costs 1 - 0.60 = 0.40 and wins 60%
    q = evaluate_quotes([0.40], [0.60], [0.62]).iloc[0]
    assert q.side == "no"
    assert q.cost == pytest.approx(0.40 + 0.07 * 0.40 * 0.60)
    assert q.edge == pytest.approx(0.60 - q.cost)


def test_no_edge_when_model_agrees_with_mid():
    q = evaluate_quotes([0.61], [0.60], [0.62]).iloc[0]
    assert q.edge < 0          # spread + fee make agreeing with the market a loser


def test_kelly_single_basic():
    assert kelly_single(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_single(0.4, 0.5) == 0.0


def test_size_event_respects_cap_and_ignores_small_edges():
    strikes = np.array([100, 101, 102, 103])
    p = np.array([0.9, 0.6, 0.3, 0.1])
    sides = np.array(["yes", "yes", "no", "no"])
    costs = np.array([0.80, 0.45, 0.55, 0.88])
    edges = np.array([0.10, 0.15, 0.15, 0.01])
    s = size_event(strikes, p, sides, costs, edges, kelly_mult=0.25, event_cap=0.05, min_edge=0.02)
    assert s.sum() <= 0.05 + 1e-9
    assert s[3] == 0                     # below min_edge
    assert (s >= 0).all() and s[:3].sum() > 0


def test_size_event_nothing_when_no_edge():
    s = size_event([1, 2], [0.5, 0.4], ["yes", "no"], [0.6, 0.7], [-0.1, -0.05])
    assert (s == 0).all()


def test_markets_frame_keeps_single_strike_above_contracts():
    raw = [
        {"ticker": "A-T1", "event_ticker": "A", "strike_type": "greater_or_equal", "floor_strike": 100,
         "cap_strike": None, "close_time": "2026-11-09T21:00:00Z", "open_time": "2026-11-09T14:00:00Z",
         "result": "yes"},
        {"ticker": "A-B1", "event_ticker": "A", "strike_type": "between", "floor_strike": 100,
         "cap_strike": 105, "close_time": "2026-11-09T21:00:00Z", "open_time": "2026-11-09T14:00:00Z",
         "result": "no"},
    ]
    df = markets_frame(raw)
    assert df["ticker"].tolist() == ["A-T1"]
    assert df["y"].iloc[0] == 1
    assert df["close_time"].iloc[0] == pd.Timestamp(datetime(2026, 11, 9, 16, 0), tz=ET)


def test_quote_at_never_looks_ahead():
    ts = pd.date_range("2026-11-09 15:00", periods=5, freq="1min", tz=ET)
    c = pd.DataFrame({"ticker": "X", "ts": ts, "bid": [0.40, 0.41, 0.42, 0.43, 0.44],
                      "ask": [0.42, 0.43, 0.44, 0.45, 0.46]})
    q = quote_at(c, ts[2])
    assert q["bid"].iloc[0] == 0.42          # candle ending exactly at t, not later
    assert quote_at(c, ts[0] - pd.Timedelta(minutes=1)).empty


def test_quote_at_drops_one_sided_and_stale():
    ts = pd.date_range("2026-11-09 15:00", periods=2, freq="1min", tz=ET)
    c = pd.DataFrame({"ticker": ["X", "Y"], "ts": ts, "bid": [0.0, 0.5], "ask": [0.0, 0.52]})
    assert quote_at(c, ts[1])["ticker"].tolist() == ["Y"]
    assert quote_at(c, ts[1] + pd.Timedelta(minutes=10)).empty
