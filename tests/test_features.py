import numpy as np

from leveledge.features import build_features, feature_columns


def test_no_lookahead(rth_bars):
    """Features at row i must not change when later bars are appended."""
    full = build_features(rth_bars, 15, False)
    cut = len(rth_bars) - 137
    part = build_features(rth_bars.iloc[:cut], 15, False)
    cols = [c for c in feature_columns(full) if c != "diurnal_factor"] + ["_sigma"]
    a, b = full[cols].iloc[:cut].to_numpy(float), part[cols].to_numpy(float)
    assert np.allclose(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9), rtol=1e-9, atol=1e-12)


def test_features_are_scale_free(rth_bars):
    """Multiplying every price by 3 must not change any model input."""
    scaled = rth_bars.copy()
    scaled[["Open", "High", "Low", "Close"]] *= 3
    a = build_features(rth_bars, 15, False)
    b = build_features(scaled, 15, False)
    cols = feature_columns(a)
    assert np.allclose(np.nan_to_num(a[cols].to_numpy(float), nan=-9),
                       np.nan_to_num(b[cols].to_numpy(float), nan=-9), rtol=1e-7, atol=1e-9)
