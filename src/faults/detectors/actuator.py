"""D5: the air conditioner is not actually cooling (FR-24).

The second detector that only exists because the model exists -- or more
precisely, because an *expectation* exists. R-02 says the IR path is
open-loop: commands go out and nothing comes back, so ``AckStatus.UNKNOWN`` is
the normal, correct answer to "did that land?". A system that treated a
missing acknowledgement as success would believe it was cooling a room it was
not touching, indefinitely.

So the test is not "was the command acknowledged" but "did the room respond".
The only honest evidence that an actuator works is the plant moving.

**Sustained cooling is tracked by state, not by counting commands.** After a
COOL the controller emits MAINTAIN on every tick while the compressor stays on
-- MAINTAIN means "keep doing what you are doing" -- so counting COOL messages
would see one command and conclude cooling had stopped. Only COOL and OFF
change the state; MAINTAIN and HOLD leave it alone. This is the same
distinction the validator draws for dwell and rate limits.

**The window is long because the room is slow.** Ten minutes, against a
thermal time constant measured in tens of minutes. An actuator fault is found
on the order of ten minutes and NFR-02 deliberately does not promise better.

**Where this can be wrong.** The test asks whether the room cooled, and a room
can fail to cool for reasons that are not the air conditioner's fault: a door
left open, a heat load nobody modelled, or an ambient high enough that the
plant is at capacity. On a 45-degree afternoon a working unit may hold the
room level rather than cool it, and this detector would call that a fault.
That is the trade R-04 settles with measured data during bring-up, by raising
the window or lowering the required cooling; it is stated here because it is a
limit of the method rather than a bug in it.

**It was more than a 45-degree afternoon.** A working unit cooling toward its
capacity slows asymptotically, and in the shipped simulation every healthy
run reached that regime near the setpoint and was declared an actuator fault.
So the ambient gap now decides which question is asked. Well below ambient
(``capacity_gap_c`` or more), a dead unit would warm visibly, so holding level
is evidence the unit works and only warming is a fault. Near ambient a working
unit has headroom to spare, so the room must still cool. With no ambient
reading the original test applies.

**The power meter is the deciding witness** (FR-23, FR-24, S-02). A room
that did not cool while the unit was told to is evidence against *something*,
and the temperature alone cannot say what: a dead unit and a thermometer
drifting upward look the same from the reading. The meter on the unit's supply
can. Through the window:

* drawing **running** power, the compressor was demonstrably working, and
  the question becomes whether the room really warmed. Humidity answers it:
  at constant vapour, a room that truly warms by a degree loses about 3 %RH.
  If humidity fell with the reading, the room warmed while the unit ran --
  lost refrigerant, the unit's fault. If it did not, the reading rose on its
  own: the warming is reported as a *plant contradiction*, which the bank
  hands to D4 on the indoor sensor. Live, before this, a 0.01 C/s drift was
  blamed on a healthy unit that was put on hold.
* drawing **standby** power while told to cool, the unit is not running, and
  that is the fault whatever the warming bound says.
* no meter, or a window that is neither, falls back to the test above.

Without humidity readings in the window, a running unit beside a warming
reading falls back to the passive-warming test above: the meter alone cannot
tell lost refrigerant from a drifting thermometer.

**Cooling is fitted, not differenced.** Two single readings, one at each end
of the window, carry the sensor's noise twice -- about 0.2 C against a 0.3 C
threshold -- and a room warming 0.2 C under a rising solar load read as 0.5.
The change over the window is the slope of a least-squares line through every
reading in it, which cuts that noise to about 0.05 C and needs no tunable.
"""

from __future__ import annotations

import math
from collections import deque

from src.common.clock import Clock
from src.common.config import ActuatorDetectorConfig
from src.common.schemas import CommandKind, DetectorId, SensorReading
from src.faults.detectors.base import Finding, Judgment

#: Commands that change whether the compressor is running. MAINTAIN and HOLD
#: assert the status quo and must not restart the evaluation window.
_STATE_CHANGING = frozenset({CommandKind.COOL, CommandKind.OFF})

#: Upper bound on readings held for one window. A window is emptied every time
#: it is judged, so this is a guard against a stalled clock, not a limit that
#: normal operation approaches (about 120 readings per 600 s window at 5 s).
_MAX_WINDOW_READINGS = 4096

#: Readings needed before a line is fitted rather than the ends differenced.
_READINGS_FOR_A_FIT = 3

#: Share of the window's meter readings that must agree before the meter
#: decides: a single start-up spike or a sample in the dead time must not.
_METER_AGREEMENT = 0.8


class ActuatorResponseDetector:
    """Watches whether sustained cooling actually cools the room.

    Holds the temperature at which the current cooling period began and the
    moment it began. Both reset whenever cooling stops, because a window that
    straddled an OFF would compare temperatures from two different regimes.
    """

    def __init__(
        self,
        subject: str,
        config: ActuatorDetectorConfig,
        clock: Clock,
        sample_period_s: float = 5.0,
        running_power_w: float | None = None,
    ) -> None:
        if not subject:
            raise ValueError("a detector must name the subject it watches")
        self._subject = subject
        self._config = config
        self._clock = clock
        self._sample_period_s = sample_period_s
        self._cooling = False
        self._started_s: float | None = None
        self._start_temperature_c: float | None = None
        self._latest_temperature_c: float | None = None
        self._ambient_c: float | None = None
        self._coupling: float | None = None
        self._window: deque[tuple[float, float]] = deque(maxlen=_MAX_WINDOW_READINGS)
        #: Draw above which the unit counts as running; None without a meter.
        self._running_power_w = running_power_w
        self._power: deque[float] = deque(maxlen=_MAX_WINDOW_READINGS)
        self._humidity: deque[tuple[float, float]] = deque(maxlen=_MAX_WINDOW_READINGS)
        self._contradiction: dict[str, float] | None = None

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def cooling(self) -> bool:
        """Whether the compressor is believed to be running."""
        return self._cooling

    def observe_command(self, kind: CommandKind) -> None:
        """Note a command, if it changes whether the plant is being cooled."""
        if kind not in _STATE_CHANGING:
            return
        if kind is CommandKind.COOL:
            self._begin_cooling()
        else:
            self._stop_cooling()

    def observe_reading(self, reading: SensorReading) -> None:
        """Note the room temperature.

        Every reading offered is accepted. The bank hands this detector the
        indoor temperature and nothing else, and naming a sensor id here would
        put a second source of truth beside the one in configuration.

        The reading's value is what matters and its timestamp is not used:
        elapsed time comes from the injected clock, for the same reason D1
        keys on arrival. A node with an unset clock would otherwise make the
        window look either instantaneous or fifty years long.
        """
        self._latest_temperature_c = reading.value
        if not self._cooling:
            return
        if self._start_temperature_c is None:
            self._start_temperature_c = reading.value
            self._started_s = self._clock.monotonic()
        self._window.append((self._clock.monotonic(), reading.value))

    def observe_power(self, watts: float) -> None:
        """Note the unit's electrical draw, while it is being told to cool."""
        if self._running_power_w is None or not self._cooling or self._started_s is None:
            return
        self._power.append(watts)

    def observe_humidity(self, pct: float) -> None:
        """Note indoor relative humidity: the room's second witness (S-02)."""
        if not self._cooling or self._started_s is None:
            return
        self._humidity.append((self._clock.monotonic(), pct))

    @property
    def plant_contradiction(self) -> dict[str, float] | None:
        """Evidence that the indoor reading, not the unit, is wrong (S-02).

        Set by :meth:`evaluate` when a full window found the room warming
        while the meter showed the compressor running throughout; None
        otherwise.
        """
        return self._contradiction

    def observe_coupling(self, a2: float) -> None:
        """Note the identified ambient coupling, a2, per sample (FR-23, FR-24).

        It sets how fast a room with a dead unit can warm, which is what tells
        a failed unit from a sensor drifting upward. Ignored unless physical.
        """
        if 0.0 < a2 < 1.0:
            self._coupling = a2

    def observe_ambient(self, ambient_c: float) -> None:
        """Note the outdoor temperature, which decides whether the unit may be
        at capacity. Not reset with the window: it is a condition, not
        evidence gathered during one."""
        self._ambient_c = ambient_c

    def reset(self) -> None:
        """Abandon the current evaluation window.

        Called when the fault retires, so the next verdict is measured from a
        fresh window rather than from temperatures recorded before whatever
        repair happened.
        """
        self._started_s = None
        self._start_temperature_c = None
        self._window.clear()
        self._power.clear()
        self._humidity.clear()
        self._contradiction = None

    def evaluate(self) -> Finding:
        """Judge the actuator on the current cooling period.

        UNKNOWN whenever there is nothing to judge: the plant is not being
        cooled, it has not been cooled for long enough to expect a measurable
        change, or no temperature has arrived to measure it with. An actuator
        nobody has asked to do anything has not been shown to work, and
        reporting it CLEAR would claim evidence that was never gathered --
        while reporting it FAULTED on a window with no temperatures in it
        would blame the air conditioner for a broken sensor.
        """
        if not self._cooling or self._started_s is None:
            return self._finding(Judgment.UNKNOWN, cooled_c=0.0, elapsed_s=0.0)

        if self._start_temperature_c is None or self._latest_temperature_c is None:
            return self._finding(Judgment.UNKNOWN, cooled_c=0.0, elapsed_s=0.0)

        elapsed_s = self._clock.monotonic() - self._started_s
        if elapsed_s < self._config.evaluation_window_s:
            return self._finding(Judgment.UNKNOWN, cooled_c=0.0, elapsed_s=elapsed_s)

        cooled_c = self._cooling_achieved_c()
        if cooled_c >= self._required_cooling_c():
            # It works. Re-anchor, so the next window is a fresh test rather
            # than a verdict that stands on one success forever.
            self._anchor_window()
            return self._finding(Judgment.CLEAR, cooled_c, elapsed_s)

        meter = self._meter_verdict()
        self._contradiction = None
        warmed_c = -cooled_c
        if meter == "running" and warmed_c >= self._config.min_cooling_c:
            humidity_change = self._humidity_change_pct()
            if humidity_change is not None:
                expected_fall = warmed_c * self._config.humidity_per_degree_pct
                if -humidity_change >= self._config.humidity_agreement * expected_fall:
                    # Humidity fell with it: the room truly warmed while the
                    # compressor ran. The unit draws power and cools nothing.
                    return self._finding(Judgment.FAULTED, cooled_c, elapsed_s)
                # The reading rose and the air did not: the reading is wrong,
                # and the unit did its part. D4 is handed the evidence.
                self._contradiction = {
                    "warmed_c": warmed_c,
                    "unit_running_w": self._median_power_w(),
                    "humidity_change_pct": humidity_change,
                    "expected_humidity_fall_pct": expected_fall,
                    "elapsed_s": elapsed_s,
                }
                return self._finding(Judgment.UNKNOWN, cooled_c, elapsed_s)
        if meter == "idle":
            # Told to cool, drawing standby: not running, whatever the reading.
            return self._finding(Judgment.FAULTED, cooled_c, elapsed_s)

        bound_c = self._passive_warming_bound_c(elapsed_s)
        if bound_c is not None and -cooled_c > bound_c:
            # Faster than any room with a dead unit could warm: the unit
            # cannot explain the reading, so the reading is the suspect, and
            # judging it is D4's job. Blaming the unit here was how a drifting
            # sensor ended up diagnosed as a failed air conditioner.
            return self._finding(Judgment.UNKNOWN, cooled_c, elapsed_s)
        return self._finding(Judgment.FAULTED, cooled_c, elapsed_s)

    def _humidity_change_pct(self) -> float | None:
        """Relative humidity's fitted change over the window; None if too few."""
        if len(self._humidity) < _READINGS_FOR_A_FIT:
            return None
        return _fitted_change(list(self._humidity))

    def _meter_verdict(self) -> str | None:
        """"running", "idle", or None when there is no meter or it is split."""
        if self._running_power_w is None or len(self._power) < _READINGS_FOR_A_FIT:
            return None
        running = sum(1 for watts in self._power if watts >= self._running_power_w)
        if running >= _METER_AGREEMENT * len(self._power):
            return "running"
        if len(self._power) - running >= _METER_AGREEMENT * len(self._power):
            return "idle"
        return None

    def _median_power_w(self) -> float:
        ordered = sorted(self._power)
        return ordered[len(ordered) // 2] if ordered else 0.0

    def _passive_warming_bound_c(self, elapsed_s: float) -> float | None:
        """The most a room with a dead unit could have warmed over the window,
        with the configured allowance; None without the model or ambient."""
        if (
            self._coupling is None
            or self._ambient_c is None
            or self._start_temperature_c is None
        ):
            return None
        time_constant_s = -self._sample_period_s / math.log(1.0 - self._coupling)
        gap_c = max(0.0, self._ambient_c - self._start_temperature_c)
        passive_c = gap_c * (1.0 - math.exp(-elapsed_s / time_constant_s))
        return (
            self._config.passive_warming_factor * passive_c
            + self._config.passive_warming_slack_c
        )

    def _required_cooling_c(self) -> float:
        """How much the room must have cooled for the unit to count as working.

        Near ambient, ``min_cooling_c``. Far enough below it that the unit may
        be at capacity, the room need only not have warmed by that much.
        """
        if self._at_possible_capacity():
            return -self._config.min_cooling_c
        return self._config.min_cooling_c

    def _at_possible_capacity(self) -> bool:
        if self._ambient_c is None or self._latest_temperature_c is None:
            return False
        gap_c = self._ambient_c - self._latest_temperature_c
        return gap_c >= self._config.capacity_gap_c

    def _cooling_achieved_c(self) -> float:
        """How much the room fell over the window. Negative means it rose.

        Section 5.5 writes this test as ``|dT|`` below a threshold, which
        catches a room that did not move but not a room that got *warmer*
        while the compressor was supposedly running -- the more certain
        actuator fault of the two. Signed cooling subsumes both cases, so that
        is what is measured here and what section 5.5 now says.
        """
        if self._start_temperature_c is None or self._latest_temperature_c is None:
            return 0.0
        if len(self._window) < _READINGS_FOR_A_FIT:
            return self._start_temperature_c - self._latest_temperature_c
        return -self._fitted_change_c()

    def _fitted_change_c(self) -> float:
        """Change across the window along a least-squares line through it."""
        return _fitted_change(list(self._window))

    def _conditions(self, elapsed_s: float) -> dict[str, float]:
        """What the verdict was judged against, when it is known.

        The passive-warming bound decides whether a warming room is blamed on
        the unit or left to D4, and it rests on the identified coupling and
        the ambient. A finding that omitted them could not show why a drifting
        sensor was, or was not, called a failed air conditioner.
        """
        conditions: dict[str, float] = {}
        if self._ambient_c is not None:
            conditions["ambient_c"] = self._ambient_c
        if self._coupling is not None:
            conditions["coupling_a2"] = self._coupling
        bound_c = self._passive_warming_bound_c(elapsed_s)
        if bound_c is not None:
            conditions["passive_warming_bound_c"] = bound_c
        if self._power:
            conditions["unit_median_w"] = self._median_power_w()
        return conditions

    def _begin_cooling(self) -> None:
        if self._cooling:
            return
        self._cooling = True
        self._anchor_window()

    def _stop_cooling(self) -> None:
        self._cooling = False
        self._started_s = None
        self._start_temperature_c = None
        self._window.clear()
        self._power.clear()
        self._humidity.clear()
        self._contradiction = None

    def _anchor_window(self) -> None:
        """Start a fresh evaluation window from the temperature now known."""
        self._started_s = self._clock.monotonic()
        self._start_temperature_c = self._latest_temperature_c
        self._window.clear()
        self._power.clear()
        self._humidity.clear()
        if self._latest_temperature_c is not None:
            self._window.append((self._started_s, self._latest_temperature_c))

    def _finding(
        self, judgment: Judgment, cooled_c: float, elapsed_s: float
    ) -> Finding:
        return Finding(
            detector=DetectorId.D5_ACTUATOR_NO_RESPONSE,
            subject=self._subject,
            judgment=judgment,
            confidence=1.0 if judgment is Judgment.FAULTED else 0.0,
            evidence={**self._conditions(elapsed_s),
                "cooled_c": cooled_c,
                "min_cooling_c": self._config.min_cooling_c,
                "elapsed_s": elapsed_s,
                "window_s": self._config.evaluation_window_s,
                "start_temperature_c": (
                    self._start_temperature_c
                    if self._start_temperature_c is not None
                    else 0.0
                ),
                "latest_temperature_c": (
                    self._latest_temperature_c
                    if self._latest_temperature_c is not None
                    else 0.0
                ),
                "required_cooling_c": self._required_cooling_c(),
            },
        )


def _fitted_change(points: list[tuple[float, float]]) -> float:
    """Change across the points along a least-squares line through them."""
    times = [t for t, _ in points]
    values = [value for _, value in points]
    mean_t = sum(times) / len(times)
    mean_value = sum(values) / len(values)
    spread = sum((t - mean_t) ** 2 for t in times)
    if spread == 0.0:
        return values[-1] - values[0]
    slope = sum((t - mean_t) * (v - mean_value) for t, v in zip(times, values)) / spread
    return slope * (times[-1] - times[0])
