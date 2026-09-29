"""The electricity tariff as a schedule (FR-16).

Peak pricing is a published timetable, not a measurement, so the band is a
pure function of local time and configuration. Keeping it here, with no
blackboard and no state, lets the control service apply the offset and the
supervisor read the band from one definition -- and lets a test ask what the
band is at any moment without waiting for it.

Local time is derived from the injected clock's epoch seconds, never read
directly (section 2 of the coding rules).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.common.config import TariffConfig
from src.common.schemas import TariffBand, TariffState


class TariffSchedule:
    """Which band applies when, and when it next changes."""

    def __init__(self, config: TariffConfig) -> None:
        self._config = config

    @property
    def peak_offset_c(self) -> float:
        return self._config.peak_offset_c

    def band_at(self, ts: float) -> TariffBand:
        hour = datetime.fromtimestamp(ts).hour
        if self._config.peak_start_hour <= hour < self._config.peak_end_hour:
            return TariffBand.PEAK
        return TariffBand.NORMAL

    def next_transition_ts(self, ts: float) -> float:
        """Epoch seconds at which the band next changes."""
        moment = datetime.fromtimestamp(ts)
        day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        for offset_days in (0, 1):
            for hour in sorted((self._config.peak_start_hour, self._config.peak_end_hour)):
                candidate = day + timedelta(days=offset_days, hours=hour)
                if candidate > moment:
                    return candidate.timestamp()
        raise AssertionError("a daily schedule always changes within two days")

    def state(self, ts: float) -> TariffState:
        return TariffState(
            ts=ts, band=self.band_at(ts), next_transition_ts=self.next_transition_ts(ts)
        )
