import numpy as np
import pandas as pd
import pytest

from leveledge import portfolio as P


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=900)
    cols = ["A", "B", "C"]
    lr = rng.normal(0.0003, 0.01, size=(len(idx), 3))
    closes = pd.DataFrame(100 * np.exp(np.cumsum(lr, axis=0)), index=idx, columns=cols)
    opens = closes.shift(1).fillna(100.0) * np.exp(rng.normal(0, 0.002, size=closes.shape))
    cash = pd.Series(0.0001, index=idx)
    return opens, closes, cash


def test_momentum_signal_uses_only_past(market):
    _, closes, cash = market
    s1 = P.momentum_signal(closes, cash, 252)
    future = closes.copy()
    future.iloc[600:] *= 5.0                       # change the future only
    s2 = P.momentum_signal(future, cash, 252)
    pd.testing.assert_frame_equal(s1.iloc[:600], s2.iloc[:600])


def test_backtest_decision_affects_only_later_days(market):
    opens, closes, cash = market
    d = closes.index[400]
    dec_a = pd.DataFrame([[0.5, 0.0, 0.0]], index=[d], columns=closes.columns)
    dec_b = pd.DataFrame([[0.0, 0.5, 0.0]], index=[d], columns=closes.columns)
    a = P.backtest(opens, closes, cash, dec_a, cost_bps=0)
    b = P.backtest(opens, closes, cash, dec_b, cost_bps=0)
    assert a.loc[d, "ret"] == b.loc[d, "ret"]      # decision day itself is unaffected
    assert a["ret"].iloc[1] != b["ret"].iloc[1]    # the trade happens at the next open


def test_backtest_flat_book_earns_cash(market):
    opens, closes, cash = market
    dec = pd.DataFrame([[0.0, 0.0, 0.0]], index=[closes.index[10]], columns=closes.columns)
    bt = P.backtest(opens, closes, cash, dec)
    assert np.allclose(bt["ret"].iloc[2:], cash.iloc[12:].to_numpy())


def test_costs_reduce_returns(market):
    opens, closes, cash = market
    cfg = P.StrategyConfig(universe=list(closes.columns), lookback=126)
    w = P.strategy_weights(opens, closes, cash, cfg)
    r0 = P.backtest(opens, closes, cash, w, cost_bps=0)["ret"]
    r5 = P.backtest(opens, closes, cash, w, cost_bps=5)["ret"]
    assert r5.sum() < r0.sum()


def test_target_weights_respect_caps():
    sig = pd.Series([1.0, 1.0, 1.0, 0.0], index=list("ABCD"))
    vol = pd.Series([0.001, 0.001, 0.02, 0.01], index=list("ABCD"))   # very low vol tempts leverage
    corr = pd.DataFrame(np.eye(4), index=list("ABCD"), columns=list("ABCD"))
    w = P.target_weights(sig, vol, corr, P.StrategyConfig(max_weight=0.25, max_gross=1.0))
    assert (w <= 0.25 + 1e-12).all() and w.sum() <= 1.0 + 1e-12
    assert w["D"] == 0.0 and (w >= 0).all()


def test_qlike_prefers_correct_variance():
    rng = np.random.default_rng(0)
    true_var = rng.uniform(0.5, 2.0, 5000)
    realized = true_var * rng.chisquare(5, 5000) / 5
    assert P.qlike(realized, true_var) < P.qlike(realized, np.full(5000, true_var.mean()))
