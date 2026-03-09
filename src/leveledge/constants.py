"""Shared constants used throughout the LevelEdge helpers."""

from zoneinfo import ZoneInfo

US_EASTERN = ZoneInfo("US/Eastern")

ALLOWED_INTERVALS: list[str] = [
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "1h",
    "90m",
    "1d",
]
