"""Regulatory controller: deterministic setpoint tracking.

No learned or stochastic components (FR-10). This layer runs at 5 s cadence
and is the only thing in the system that must be real-time; it keeps running
at its nominal cadence when the reasoning layer is slow, wrong, or entirely
absent, holding the last validated setpoint (FR-11, FR-47).

**Intent is re-asserted, not assumed to have landed.** R-02 makes the IR path
open-loop: a command can be lost and nothing says so. A controller that sends
COOL or OFF once and then reports MAINTAIN forever has no way back from that
loss -- measured, a dropped OFF left the compressor running while the
controller believed it had stopped, and the room reached 18 C with the
controller sitting in MAINTAIN because neither of its transition conditions
could fire again. Periodically re-sending the intended state is what a real
IR integration does, and it is what makes a lost command recoverable rather
than permanent.

The adapted thermal model informs this controller but does not replace it.
Feedback authority stays in a conventional deadband law with dwell-time
protection, and the model contributes one thing: a prediction to control on
when the temperature sensor cannot be trusted (FR-27). That substitution is
the concrete payoff of having a model at all — closed-loop control continues
on a prediction rather than collapsing to open loop.

Control law: DESIGN.md section 5.3.
"""

from __future__ import annotations

from src.common.clock import Clock
from src.common.config import ControllerConfig
from src.common.schemas import Command, CommandKind, Mode

#: Modes in which the controller must not drive the plant at all. It emits
#: HOLD rather than falling silent, so downstream state stays observable.
_HOLD_MODES = frozenset({Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD})

#: The mode in which the model's prediction stands in for the measurement.
_SUBSTITUTING_MODE = Mode.DEGRADED_SENSOR


class RegulatoryController:
    """Tracks a validated setpoint with a deadband and a dwell timer.

    Holds the compressor state and the time of its last transition. Dwell is
    enforced here and independently in the safety validator; this instance
    exists so the controller does not *ask* for a command it knows is
    illegal, and the validator's exists so a bug here cannot damage the
    compressor.
    """

    def __init__(
        self,
        config: ControllerConfig,
        clock: Clock,
        actuator_id: str,
    ) -> None:
        self._config = config
        self._clock = clock
        self._actuator_id = actuator_id
        self._compressor_on = False
        self._last_transition_ts: float | None = None
        # Seeded rather than left empty so the first tick reports MAINTAIN
        # like any other quiet tick. The plant is still put into a known state
        # within one interval of startup, and a real transition is asserted
        # the moment it happens.
        self._last_assert_ts = clock.now()

    @property
    def compressor_on(self) -> bool:
        """Whether the controller believes the compressor is running."""
        return self._compressor_on

    def effective_temperature_c(
        self, measured_c: float, predicted_c: float, mode: Mode
    ) -> float:
        """The temperature the control law acts on.

        In ``DEGRADED_SENSOR`` the model's prediction stands in for a
        measurement that is known to be untrustworthy (FR-27). In every other
        mode the measurement is used directly: a model is a worse source of
        truth than a working sensor.
        """
        if mode is _SUBSTITUTING_MODE:
            return predicted_c
        return measured_c

    def tick(
        self,
        measured_c: float,
        predicted_c: float,
        setpoint_c: float,
        mode: Mode,
    ) -> Command:
        """Produce one command. Called every regulatory period.

        Always returns a command. Silence would be indistinguishable from a
        crashed controller, and the blackboard is how the rest of the system
        knows this loop is alive.
        """
        now = self._clock.now()

        if mode in _HOLD_MODES:
            return self._command(now, CommandKind.HOLD)

        temperature_c = self.effective_temperature_c(measured_c, predicted_c, mode)
        error_c = temperature_c - setpoint_c

        if self._should_start_cooling(error_c, now):
            self._transition(to_on=True, now=now)
            return self._assert_state(now, setpoint_c)

        if self._should_stop_cooling(error_c):
            self._transition(to_on=False, now=now)
            return self._assert_state(now, setpoint_c)

        if self._reassert_due(now):
            return self._assert_state(now, setpoint_c)

        return self._command(now, CommandKind.MAINTAIN)

    def _reassert_due(self, now: float) -> bool:
        """Whether the intended state should be re-sent (R-02).

        Nothing observes that a command landed, so the state is re-sent on a
        timer rather than on evidence. The interval is config, because on
        hardware it trades against how often an IR blaster may reasonably fire.
        """
        return now - self._last_assert_ts >= self._config.reassert_interval_s

    def _assert_state(self, now: float, setpoint_c: float) -> Command:
        """Command the plant into the state this controller intends."""
        self._last_assert_ts = now
        if self._compressor_on:
            return self._command(now, CommandKind.COOL, setpoint_c)
        return self._command(now, CommandKind.OFF)

    def _should_start_cooling(self, error_c: float, now: float) -> bool:
        if self._compressor_on:
            return False
        if error_c <= self._config.deadband_c:
            return False
        return self._dwell_elapsed(now)

    def _should_stop_cooling(self, error_c: float) -> bool:
        return self._compressor_on and error_c < -self._config.deadband_c

    def _dwell_elapsed(self, now: float) -> bool:
        """True before the first transition: nothing to protect yet."""
        if self._last_transition_ts is None:
            return True
        return now - self._last_transition_ts >= self._config.min_off_s

    def _transition(self, to_on: bool, now: float) -> None:
        self._compressor_on = to_on
        self._last_transition_ts = now

    def _command(
        self, now: float, kind: CommandKind, setpoint_c: float | None = None
    ) -> Command:
        return Command(
            ts=now,
            actuator_id=self._actuator_id,
            kind=kind,
            setpoint_c=setpoint_c,
        )
