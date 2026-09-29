"""The thermostat this project is arguing against (DESIGN.md section 8.2).

It is deliberately not a straw man, and the construction is the argument. It
runs *the same control law* as the real system -- the same
:class:`RegulatoryController`, the same deadband, the same dwell timer, the
same setpoint -- by handing it a mode of NORMAL and a prediction equal to the
measurement. Nothing about its tracking is handicapped.

What it does not have is the two contributions:

* **No model.** It has no expectation of what the room should be doing, so
  there is nothing to fall back on when the sensor stops telling the truth,
  and nothing to compare a reading against.
* **No fault layer.** It never changes mode, never degrades, never holds. Its
  only concession to a broken sensor is rejecting a physically impossible
  reading, which is what a real thermostat does and which section 8.2 grants
  it explicitly.

That last point is where E5 is decided. Given a sensor frozen at a plausible
value, this controller keeps acting on the frozen number indefinitely -- it
cannot tell a still room from a dead sensor, because telling them apart
requires exactly the expectation it does not have. The comparison is therefore
attributable: any difference in comfort under fault comes from the model and
the detector bank, not from a better-tuned loop.
"""

from __future__ import annotations

import logging

from src.common.clock import Clock
from src.common.config import Bounds, ControllerConfig
from src.common.schemas import Command, Mode, SensorReading
from src.control.regulatory import RegulatoryController

LOGGER = logging.getLogger(__name__)

#: The only mode it knows. A thermostat has no degradation states, so it runs
#: as though everything were always fine -- which, under a fault, is the
#: mistake being measured.
_ALWAYS_NORMAL = Mode.NORMAL


class BaselineThermostat:
    """Fixed-deadband control on the raw reading, with no model behind it."""

    def __init__(
        self,
        config: ControllerConfig,
        clock: Clock,
        actuator_id: str,
        limits: Bounds,
    ) -> None:
        self._controller = RegulatoryController(
            config=config, clock=clock, actuator_id=actuator_id
        )
        self._limits = limits
        self._setpoint_c = config.default_setpoint_c
        self._measured_c: float | None = None
        self._rejected = 0

    @property
    def setpoint_c(self) -> float:
        return self._setpoint_c

    @setpoint_c.setter
    def setpoint_c(self, value: float) -> None:
        """A schedule may move the target. Both systems get the same one."""
        self._setpoint_c = value

    @property
    def rejected_readings(self) -> int:
        """Readings discarded as physically impossible."""
        return self._rejected

    @property
    def believed_temperature_c(self) -> float | None:
        """What it thinks the room is. Not necessarily what the room is."""
        return self._measured_c

    def observe(self, reading: SensorReading) -> None:
        """Take a reading at face value, unless it is impossible.

        Out-of-range rejection is the whole of its fault handling, and it is
        granted deliberately (section 8.2). Everything else -- a sensor that
        has frozen at a plausible value, or gone silent, or drifted -- is
        indistinguishable from a quiet room to a controller with no model, so
        the last believed temperature simply stands.
        """
        if not self._limits.contains(reading.value):
            self._rejected += 1
            LOGGER.debug(
                "baseline rejected %.2f from %s", reading.value, reading.sensor_id
            )
            return
        self._measured_c = reading.value

    def tick(self) -> Command | None:
        """One control cycle.

        :returns: the command, or None before any reading has arrived. The
            real system holds off for the same reason: there is nothing to
            control on yet.
        """
        if self._measured_c is None:
            return None
        return self._controller.tick(
            measured_c=self._measured_c,
            # No model: the best guess at the room is the last reading, so the
            # substitution the real system performs in DEGRADED_SENSOR is a
            # no-op here. That absence is the thing being measured.
            predicted_c=self._measured_c,
            setpoint_c=self._setpoint_c,
            mode=_ALWAYS_NORMAL,
        )
