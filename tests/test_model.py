from datetime import datetime

import numpy as np
import pytest

from leveledge.constants import US_EASTERN as ET
from leveledge.model import ForecastConfig, LevelForecaster, fit_stacker
from leveledge.sessions import compute_horizon


@pytest.fixture(scope="module")
def fitted(rth_bars):
    h = compute_horizon(rth_bars.index[-1], datetime(2026, 11, 2, 16, 0, tzinfo=ET), 15, False)
    return LevelForecaster(rth_bars, 15, False, h, ForecastConfig(n_jobs=2)).fit()


def test_curve_is_monotone(fitted):
    c = fitted.curve()
    assert np.all(np.diff(c["p_above"]) <= 1e-12)
    assert c["p_above"].iloc[0] > 0.9 and c["p_above"].iloc[-1] < 0.1


def test_levels_query_monotone(fitted):
    S = fitted.current_price
    p = fitted.prob_above([S * x for x in np.linspace(0.95, 1.05, 41)])
    assert np.all(np.diff(p) <= 1e-9)


def test_random_walk_is_calibrated(fitted):
    """On a driftless random walk the forecast should be near 50% at the spot
    and the out-of-fold 80% range should cover roughly 80% of outcomes."""
    p_at_spot = fitted.prob_above([fitted.current_price])[0]
    assert 0.38 < p_at_spot < 0.62
    cov = fitted.interval_coverage()
    assert 0.72 < cov < 0.88
    m = fitted.metrics
    assert m["blend"]["log_loss"] <= m["baseline"]["log_loss"] + 0.01


def test_quantiles_ordered(fitted):
    q = fitted.quantiles()
    vals = [q[k] for k in sorted(q)]
    assert vals == sorted(vals)


def test_level_backtest_reports_skill(fitted):
    lb = fitted.level_backtest(fitted.current_price * 1.005)
    assert set(["blend", "baseline", "climatology", "brier_skill"]) <= set(lb)


def test_stacker_has_no_drift_term():
    rng = np.random.default_rng(1)
    lb = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(lb - 1.0)))).astype(float)  # biased sample
    a, b, c = fit_stacker(lb + rng.normal(0, 1, 4000), lb, y)
    assert c == 0.0 and a >= 0 and b >= 0


def test_classifier_must_pass_holdout_gate(fitted):
    """On a pure random walk there is nothing to learn: the gate should keep the
    classifier out, leaving the calibrated baseline."""
    g = fitted.gate
    assert g is not None and set(g) >= {"passed", "holdout_log_loss_blend", "holdout_log_loss_baseline"}
    if not g["passed"]:
        assert fitted.weights == (0.0, 1.0, 0.0)
        assert fitted.model is None
