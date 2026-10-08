"""Local wall time of the room, from the injected clock.

Several things must agree on what "Thursday at three" or "18:00" means: the
tariff schedule, the date a model is told when it resolves a request, and the
timestamps the supervisor reads. ``datetime.fromtimestamp`` answers in the
*machine's* zone, so the same seed gave a different tariff band -- and a
different E5 result -- on a laptop in IST than on an Orin provisioned in UTC.

Local time is therefore epoch seconds plus the configured site offset
(``site.utc_offset_h``), and it is naive: tool arguments are naive ISO-8601
local times (section 5.7.6), and comparing an aware value with a naive one
raises rather than answering.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

SECONDS_PER_HOUR = 3600.0


def local_time(epoch_s: float, utc_offset_h: float) -> datetime:
    """The room's local wall time at ``epoch_s``, without a zone attached."""
    moment = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    shifted = moment + timedelta(seconds=utc_offset_h * SECONDS_PER_HOUR)
    return shifted.replace(tzinfo=None)


def epoch_of(local: datetime, utc_offset_h: float) -> float:
    """Epoch seconds of a naive local wall time; the inverse of :func:`local_time`.

    :raises ValueError: if ``local`` carries a zone. An aware value already
        names its own instant, and silently re-reading it at the site offset
        would move it.
    """
    if local.tzinfo is not None:
        raise ValueError(f"expected a naive local time, got {local.isoformat()}")
    as_utc = local.replace(tzinfo=timezone.utc)
    return as_utc.timestamp() - utc_offset_h * SECONDS_PER_HOUR
