import numpy as np
import pandas as pd
from scipy.stats import norm

from leveledge import regime as R


def _two_regime(T=5000, seed=1):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, int)
    for t in range(1, T):
        stay = 0.99 if s[t - 1] == 0 else 0.96
        s[t] = s[t - 1] if rng.random() < stay else 1 - s[t - 1]
    x = np.where(s == 0, rng.normal(0.05, 0.7, T), rng.normal(-0.1, 2.0, T))
    return x, s


def test_hmm_recovers_states_and_filter_is_causal():
    x, s = _two_regime()
    fit = R.hmm2_fit(x)
    assert fit["sd"][1] > fit["sd"][0]
    assert abs(fit["sd"][0] - 0.7) < 0.1 and abs(fit["sd"][1] - 2.0) < 0.25
    p = R.hmm2_filter(fit, x)
    assert ((p > 0.5) == s).mean() > 0.9
    # filtered probabilities up to t do not change when later data is appended
    assert np.allclose(R.hmm2_filter(fit, x[:3000]), p[:3000])


def test_gjr_recovers_leverage_effect():
    rng = np.random.default_rng(2)
    T, e, s2 = 6000, np.zeros(6000), 0.5
    for t in range(T):
        e[t] = np.sqrt(s2) * rng.standard_t(7) / np.sqrt(7 / 5)
        s2 = 0.02 + (0.02 + 0.12 * (e[t] < 0)) * e[t] ** 2 + 0.88 * s2
    fit = R.gjr_fit(e)
    assert 0.06 < fit["gamma"] < 0.2 and 0.8 < fit["beta"] < 0.95
    paths = R.gjr_simulate(fit, fit["next_s2"], 21, n=4000)
    assert paths.shape == (4000, 21)
    assert np.all(np.diff(paths.std(0)) > -0.05)          # uncertainty grows with the horizon


def test_jump_model_online_state_uses_past_only():
    x, s = _two_regime(seed=3)
    X = R.jm_features(pd.Series(x / 100)).to_numpy()
    Z = (X - X.mean(0)) / X.std(0)
    c, states = R.jm_fit(Z, 50.0)
    assert np.count_nonzero(np.diff(states)) < 200          # the penalty keeps regimes persistent
    on = R.jm_online(Z, c, 50.0)
    assert np.array_equal(R.jm_online(Z[:2000], c, 50.0), on[:2000])


def test_crps_matches_closed_form_for_normal():
    y = np.random.default_rng(0).normal(size=2000)
    lv = np.round(np.arange(0.005, 1, 0.005), 3)
    crps = R.crps_from_quantiles(y, np.tile(norm.ppf(lv), (len(y), 1)), lv).mean()
    assert abs(crps - 1 / np.sqrt(np.pi)) < 0.02      # expected CRPS of a correct N(0,1) forecast = 0.5642


def test_daily_variance_positive():
    df = pd.DataFrame({"Open": [100, 101, 99.0], "High": [102, 103, 101.0], "Low": [99, 100, 97.0], "Close": [101, 100, 100.0]})
    v = R.daily_variance(df)
    assert (v > 0).all()
