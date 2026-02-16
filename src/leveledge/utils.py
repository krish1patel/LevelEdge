from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_MARKET_TZ = "US/Eastern"


def ensure_future_market_datetime(target_datetime: datetime, tz_name: str = DEFAULT_MARKET_TZ) -> datetime:
    """Return a timezone-aware datetime in ``tz_name`` that is in the future.

    If ``target_datetime`` is naive, it is interpreted in ``tz_name``. If it is aware, it
    is converted to ``tz_name``. The function raises ``ValueError`` when the resulting
    timestamp is in the past (relative to ``tz_name`` now).
    """

    tz = ZoneInfo(tz_name)
    if target_datetime.tzinfo is None:
        normalized_dt = target_datetime.replace(tzinfo=tz)
    else:
        normalized_dt = target_datetime.astimezone(tz)

    now = datetime.now(tz)
    if normalized_dt <= now:
        raise ValueError(
            f"Target datetime ({normalized_dt.isoformat()}) must be in the future relative to {tz_name} (current {now.isoformat()})."
        )

    return normalized_dt
