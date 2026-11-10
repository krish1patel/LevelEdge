from datetime import datetime

import pytest

from leveledge import Predictor, forecast
from leveledge.constants import US_EASTERN as ET

AS_OF = datetime(2026, 10, 30, 12, 0, tzinfo=ET)
TARGET = datetime(2026, 10, 30, 16, 0, tzinfo=ET)


def test_legacy_api_roundtrip(rth_bars):
    p = Predictor("SPY", TARGET, "15m", 500.0, end_datetime=AS_OF, data=rth_bars)
    assert p.candles_ahead == 16
    p.train_xgb()
    prob = p.predict_xgb()
    assert 0 <= prob <= 1
    auc, ps, pr = p.get_xgb_model_metrics()
    assert auc is None or 0 <= auc <= 1
    rec = p.log_record(prob)
    assert rec["model_type"] == "leveledge_v2_stacked" and rec["candles_ahead"] == 16


def test_predictors_share_one_fit(rth_bars):
    m1 = forecast("SPY", TARGET, "15m", as_of=AS_OF, data=rth_bars)
    m2 = forecast("SPY", TARGET, "15m", as_of=AS_OF, data=rth_bars)
    assert m1 is m2


def test_rejects_bad_inputs(rth_bars):
    with pytest.raises(ValueError):
        Predictor("SPY", TARGET, "7m", 500.0, end_datetime=AS_OF, data=rth_bars)
    with pytest.raises(ValueError):
        Predictor("SPY", AS_OF, "15m", 500.0, end_datetime=TARGET, data=rth_bars)
