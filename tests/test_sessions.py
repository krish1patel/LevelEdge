from datetime import datetime

import pandas as pd
import pytest

from leveledge.constants import US_EASTERN as ET
from leveledge.sessions import bar_ends, compute_horizon, completed_bars, regular_session_only

T = lambda s: pd.Timestamp(s, tz=ET)


@pytest.mark.parametrize("last_start,target,imin,crypto,bars,eff,crossed", [
    ("2026-11-04 09:45", "2026-11-04 16:00", 15, False, 24, "2026-11-04 16:00", 0),
    # day after Thanksgiving: 13:00 early close
    ("2026-11-27 10:00", "2026-11-27 16:00", 60, False, 2, "2026-11-27 13:00", 0),
    # Friday close -> Monday 10:00 crosses one session close
    ("2026-11-06 15:45", "2026-11-09 10:00", 15, False, 2, "2026-11-09 10:00", 1),
    # 90m bars: 11:00, 12:30, 14:00, 15:30 -> last bar is the 30-minute 15:30-16:00 stub
    ("2026-11-09 09:30", "2026-11-09 16:00", 90, False, 4, "2026-11-09 16:00", 0),
    # target between bar closes resolves on the earlier close
    ("2026-11-04 09:45", "2026-11-04 15:50", 15, False, 23, "2026-11-04 15:45", 0),
    ("2026-11-06 00:00", "2026-11-13 16:00", 1440, False, 5, "2026-11-13 16:00", 5),
    ("2026-11-09 09:00", "2026-11-09 16:00", 60, True, 6, "2026-11-09 16:00", 0),
])
def test_horizon(last_start, target, imin, crypto, bars, eff, crossed):
    h = compute_horizon(T(last_start), T(target).to_pydatetime(), imin, crypto)
    assert (h.bars, h.effective_target, h.sessions_crossed) == (bars, T(eff), crossed)


def test_horizon_rejects_target_before_next_close():
    with pytest.raises(ValueError):
        compute_horizon(T("2026-11-04 09:45"), T("2026-11-04 10:10").to_pydatetime(), 15, False)


def test_last_hourly_bar_ends_at_close():
    ends = bar_ends(pd.DatetimeIndex([T("2026-11-04 15:30")]), 60, False)
    assert ends[0] == T("2026-11-04 16:00")


def test_incomplete_bar_dropped():
    idx = pd.DatetimeIndex([T("2026-11-04 09:30"), T("2026-11-04 09:45"), T("2026-11-04 10:00")])
    df = pd.DataFrame({"Close": [1, 2, 3]}, index=idx)
    out = completed_bars(df, 15, False, datetime(2026, 11, 4, 10, 7, tzinfo=ET))
    assert list(out.index) == list(idx[:2])


def test_prepost_bars_removed():
    idx = pd.DatetimeIndex([T("2026-11-04 08:00"), T("2026-11-04 09:30"), T("2026-11-04 16:00")])
    df = pd.DataFrame({"Close": [1, 2, 3]}, index=idx)
    assert list(regular_session_only(df, 15).index) == [idx[1]]
