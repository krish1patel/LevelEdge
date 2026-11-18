import numpy as np
import pandas as pd

from leveledge import integrity as I
from leveledge import portfolio as P
from leveledge.news import LAG_DAYS, available, sentiment_signals


def _ctx(n=1400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    cols = list(P.UNIVERSE)
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, len(cols))), axis=0)), index=idx, columns=cols)
    o = c.shift(1).fillna(c.iloc[0]) * (1 + rng.normal(0, 0.001, c.shape))
    return I.Ctx(o, c, pd.Series(0.0001, index=idx))


def _clean(ctx, window=20):
    c = ctx.closes["SPY"]
    return I._spy_rule(ctx, (c > c.shift(window)).astype(float).where(c.shift(window).notna()))


def _leaky(ctx, window=5):
    c = ctx.closes["SPY"]
    return I._spy_rule(ctx, (c.shift(-window) > c).astype(float).fillna(0.0))        # peeks at next week's price


def _register(monkeypatch):
    for sid, fn in (("t_clean", _clean), ("t_leaky", _leaky)):
        monkeypatch.setitem(I.STRATEGIES, sid, {"name": sid, "decide": fn, "params": {}, "grid": [{"window": w} for w in (10, 20, 40)],
                                                "benchmark": "spy", "universe": "etf"})
    monkeypatch.setattr(I, "SHUFFLE_CHUNK", 5)
    monkeypatch.setattr(I, "SHUFFLES", 10)
    monkeypatch.setattr(I, "DEV_END", "2017-12-31")
    monkeypatch.setattr(I, "HOLD_START", "2018-01-01")


def _evaluate(sid, ctx):
    return I.aggregate(sid, {t["id"]: I.run_task(t, ctx) for t in I.plan(sid)})


def test_lookahead_is_caught_and_caps_the_score(monkeypatch):
    _register(monkeypatch)
    ctx = _ctx()
    leaky, clean = _evaluate("t_leaky", ctx), _evaluate("t_clean", ctx)
    assert leaky["checks"]["lookahead"]["status"] == "fail"
    assert leaky["score"] <= 20 and leaky["grade"] == "F"
    assert any("saw the future" in f for f in leaky["findings"])
    assert clean["checks"]["lookahead"]["status"] == "pass"


def test_plan_splits_work_into_independent_tasks(monkeypatch):
    _register(monkeypatch)
    kinds = [t["kind"] for t in I.plan("t_clean")]
    assert kinds.count("base") == 1 and kinds.count("grid") == 3 and kinds.count("shuffle") == 2
    assert len({t["id"] for t in I.plan("t_clean")}) == len(kinds)                    # unique ids (idempotent storage keys)


def test_reversed_variant_ranking_fails_stability(monkeypatch):
    _register(monkeypatch)
    ctx = _ctx(seed=3)
    res = {t["id"]: I.run_task(t, ctx) for t in I.plan("t_clean")}
    for i, (d, h) in enumerate(((1.0, 0.1), (0.5, 0.5), (0.1, 1.0))):                 # best before = worst after
        res[f"t_clean:grid:{i}"]["sharpe"] = {"full": 0.5, "dev": d, "holdout": h}
    rep = I.aggregate("t_clean", res)
    assert rep["checks"]["stability"]["status"] == "fail"
    assert any("tended to do worst after" in f for f in rep["findings"])


def test_weights_sum_to_100():
    assert sum(I.WEIGHTS.values()) == 100


def test_news_sentiment_is_only_used_after_publication():
    days = pd.bdate_range("2020-01-01", periods=60)
    raw = pd.Series(np.arange(80, dtype=float), index=pd.date_range("2019-12-20", periods=80))
    known = available(raw, days)
    for d in days[10:]:
        assert raw.index[raw == known.loc[d]][0] <= d - pd.Timedelta(days=LAG_DAYS)
    sig = sentiment_signals(raw, days)
    assert set(sig) == {"level", "momentum", "level_raw", "change_21d"}
