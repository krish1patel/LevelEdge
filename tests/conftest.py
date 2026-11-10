import os

import numpy as np
import pandas as pd
import pytest

os.environ["LEVELEDGE_DISABLE_LOGGING"] = "1"

from leveledge.constants import US_EASTERN
from leveledge.sessions import nyse_schedule


def make_rth_bars(start="2026-08-03", end="2026-10-30", minutes=15, sigma_day=0.012, seed=0):
    """Synthetic random-walk regular-session bars on the real NYSE calendar."""
    rng = np.random.default_rng(seed)
    sched = nyse_schedule(start, end)
    idx = []
    for _, row in sched.iterrows():
        t = row["market_open"]
        while t < row["market_close"]:
            idx.append(t)
            t += pd.Timedelta(minutes=minutes)
    idx = pd.DatetimeIndex(idx).tz_convert(US_EASTERN)
    per_bar = sigma_day / np.sqrt(390 / minutes)
    r = rng.standard_t(5, len(idx)) * per_bar * np.sqrt(3 / 5)
    close = 500 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, per_bar / 2, len(idx))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, per_bar / 2, len(idx))))
    vol = rng.lognormal(12, 0.4, len(idx))
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol}, index=idx)


@pytest.fixture(scope="session")
def rth_bars():
    return make_rth_bars()
