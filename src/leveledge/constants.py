"""Shared constants used throughout the LevelEdge helpers."""

from datetime import timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    US_EASTERN = ZoneInfo("US/Eastern")
except ZoneInfoNotFoundError:
    US_EASTERN = timezone(timedelta(hours=-5))

ALLOWED_INTERVALS: list[str] = [
    "1m",
    "2m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "90m",
    "1d",
]
