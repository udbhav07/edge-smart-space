"""Local wall time of the room, from the injected clock.

Three things need to agree on what "Thursday at three" means: the tariff
schedule, the prompt that tells a model today's date, and the providers that
store and check the times a model writes. If any of them used the machine's
own zone and another used configuration, a node provisioned in UTC would put
the design review five and a half hours away from where it was asked for.

So local time is always epoch seconds plus the configured site offset, and it
is naive: tool arguments are naive ISO-8601 local times (section 5.7.6), and
comparing an aware value with a naive one raises rather than answering.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

SECONDS_PER_HOUR = 3600.0


def local_time(epoch_s: float, utc_offset_h: float) -> datetime:
    """The room's local wall time at ``epoch_s``, without a zone attached."""
    moment = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    shifted = moment + timedelta(seconds=utc_offset_h * SECONDS_PER_HOUR)
    return shifted.replace(tzinfo=None)
