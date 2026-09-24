"""The electricity tariff, as a schedule and as a retained topic (FR-16).

FR-16 shifts the comfort band while the tariff is at peak. The shift itself is
the supervisor's to propose, gated like any other setpoint; what this module
provides is the fact it reasons from -- which band is in force and when it
next changes -- on ``space/context/tariff``, retained, so a supervisor that
starts mid-peak knows it is mid-peak (section 5.7.2, ``get_tariff_state``).

**Time of day comes from the injected clock, not from the machine's zone.**
Local time is epoch seconds plus a configured UTC offset. A timezone database
would make the answer depend on how the Orin was provisioned, and a simulated
day would stop being reproducible from its seed; a fixed offset is honest for
a room that does not move and has no daylight saving.

It runs in the control process because that is the one process that must
stay up. A tariff published by something that had crashed would sit retained
and wrong, and ``next_transition_ts`` in the past is the only way anyone would
notice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.common import topics
from src.common.clock import Clock
from src.common.config import TariffConfig
from src.common.mqtt_client import Blackboard
from src.common.schemas import TariffBand, TariffState

LOGGER = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class _Boundary:
    """One edge of a peak window: the moment the band becomes ``band``."""

    ts: float
    band: TariffBand


class TariffSchedule:
    """Which band is in force at any instant, from the configured windows."""

    def __init__(self, config: TariffConfig) -> None:
        self._config = config

    @property
    def offset_c(self) -> float:
        return self._config.peak_offset_c

    def _local_midnight(self, ts: float) -> float:
        """Epoch seconds of the local midnight that starts ``ts``'s day."""
        offset_s = self._config.utc_offset_h * SECONDS_PER_HOUR
        return ts - ((ts + offset_s) % SECONDS_PER_DAY)

    def _boundaries_around(self, ts: float) -> list[_Boundary]:
        """Every window edge from yesterday to tomorrow, in order.

        Three days, because the band in force at 00:30 may have been set by a
        window that closed at 22:00 the day before, and the next change may be
        tomorrow evening.
        """
        today = self._local_midnight(ts)
        edges: list[_Boundary] = []
        for day in (today - SECONDS_PER_DAY, today, today + SECONDS_PER_DAY):
            for window in self._config.peak_windows:
                edges.append(
                    _Boundary(day + window.start_h * SECONDS_PER_HOUR, TariffBand.PEAK)
                )
                edges.append(
                    _Boundary(day + window.end_h * SECONDS_PER_HOUR, TariffBand.NORMAL)
                )
        return sorted(edges, key=lambda edge: edge.ts)

    def state_at(self, ts: float) -> TariffState:
        """The band in force at ``ts``, since when, and until when."""
        edges = self._boundaries_around(ts)
        if not edges:
            # No peak windows: always normal. The state still has to say when
            # it will next be looked at, so it names the next local midnight.
            midnight = self._local_midnight(ts)
            return TariffState(
                ts=ts,
                band=TariffBand.NORMAL,
                since_ts=midnight,
                next_transition_ts=midnight + SECONDS_PER_DAY,
                offset_c=self.offset_c,
            )

        past = [edge for edge in edges if edge.ts <= ts]
        future = [edge for edge in edges if edge.ts > ts]
        current = past[-1]
        return TariffState(
            ts=ts,
            band=current.band,
            since_ts=current.ts,
            next_transition_ts=future[0].ts,
            offset_c=self.offset_c,
        )


class TariffPublisher:
    """Keeps ``space/context/tariff`` true.

    Publishes on the first tick and on every change of band, and not on the
    ticks between: the topic is retained, so a republish every 5 s would say
    nothing new and bury the transitions in the recording.
    """

    def __init__(
        self, schedule: TariffSchedule, clock: Clock, blackboard: Blackboard
    ) -> None:
        self._schedule = schedule
        self._clock = clock
        self._blackboard = blackboard
        self._published: TariffState | None = None

    @property
    def current(self) -> TariffState | None:
        """The state last published, if any."""
        return self._published

    def tick(self) -> TariffState | None:
        """Publish if the band has changed since the last publication.

        :returns: the state published, or None when nothing changed.
        """
        state = self._schedule.state_at(self._clock.now())
        if self._published is not None and (
            state.band is self._published.band
            and state.since_ts == self._published.since_ts
        ):
            return None
        self._blackboard.publish(topics.CONTEXT_TARIFF, state)
        self._published = state
        LOGGER.info(
            "tariff is %s until %.0f (offset %.1f C while peak)",
            state.band.value,
            state.next_transition_ts,
            state.offset_c,
        )
        return state
