"""Time helpers shared across the application."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

_TZ8 = timezone(timedelta(hours=8))


def now_tz8_iso() -> str:
    """Return the current time in UTC+8 as an ISO 8601 string."""
    return datetime.now(_TZ8).isoformat()
