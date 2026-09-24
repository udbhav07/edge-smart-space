"""Simulated air conditioner.

Models the three things about a real unit that break naive control code:

* **Dead time.** A command does not change the room the instant it is sent.
  Until it takes effect the room behaves as though nothing was commanded,
  which is what makes D5's evaluation window long (DESIGN.md section 5.5).
* **Command loss.** An IR command can simply not arrive.
* **No acknowledgement.** With ``acknowledges`` false the unit reports
  ``AckStatus.UNKNOWN`` for every command, which is the honest model of an
  open-loop IR path (R-02). Code that reads UNKNOWN as success will
  misdiagnose D5, so the simulator refuses to pretend otherwise by default.

The unit is bang-bang: the deadband law in section 5.3 emits COOL or OFF,
never a modulated fraction. The identified coefficient a3 therefore sees a
binary drive, which is exactly the weak-excitation situation R-01 warns
about.

Time advances through ``apply_due_commands``, which the simulation loop calls
once per step. Reading ``cooling_fraction`` never changes anything: a getter
that silently advanced state would make the plant's behaviour depend on how
many times it happened to be queried.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from src.common.clock import Clock
from src.common.config import SimActuatorConfig
from src.common.injection import (
    ACTUATOR_SUPPORTED_FAULTS,
    NO_FAULT,
    FaultInjection,
    InjectedFault,
)
from src.common.schemas import AckStatus, CommandKind

#: Bang-bang drive levels, as a fraction of rated cooling power.
COOLING_ON = 1.0
COOLING_OFF = 0.0

#: Commands that can be in flight at once. Dead time divided by the minimum
#: command interval is under two in the shipped configuration; this is a
#: generous bound that still refuses to grow without limit.
MAX_IN_FLIGHT_COMMANDS = 16

#: MAINTAIN instructs the unit to keep doing what it is already doing, and is
#: the only command that changes nothing.
_NO_CHANGE_COMMANDS = frozenset({CommandKind.MAINTAIN})

#: Commands that stop the unit driving the room. HOLD is one of them: it is
#: what the controller emits in DEGRADED_ACTUATOR and SAFE_HOLD, and "hold a
#: safe state" (FR-28) cannot mean leaving a compressor running at full power
#: with nobody watching. Measured before this was fixed: a held system kept
#: cooling indefinitely and drove the room to 21.8 C, two degrees past the
#: setpoint it had stopped tracking.
_STOPPING_COMMANDS = frozenset({CommandKind.OFF, CommandKind.HOLD})


@dataclass(frozen=True)
class _PendingCommand:
    """A command that will take effect once dead time has elapsed."""

    effective_ts: float
    cooling_fraction: float


class SimulatedActuator:
    """The one real actuator's stand-in (DESIGN.md section 2.1).

    Randomness comes from an injected generator so a scenario replays
    identically from the same seed (FR-62).
    """

    def __init__(
        self,
        config: SimActuatorConfig,
        rng: random.Random,
        clock: Clock,
    ) -> None:
        self._config = config
        self._rng = rng
        self._clock = clock
        self._applied_fraction = COOLING_OFF
        self._pending: deque[_PendingCommand] = deque(maxlen=MAX_IN_FLIGHT_COMMANDS)
        self._failed = False
        self._last_command_ts: float | None = None
        self._injection = NO_FAULT

    @property
    def is_failed(self) -> bool:
        """Whether an actuator failure is being injected (drives D5)."""
        return self._failed

    @property
    def last_command_ts(self) -> float | None:
        """When a command was last accepted. None before the first one."""
        return self._last_command_ts

    @property
    def pending_count(self) -> int:
        """Commands sent but not yet in effect."""
        return len(self._pending)

    @property
    def injected_fault(self) -> InjectedFault:
        """What is currently being injected, for the operator's own audit."""
        return self._injection.kind

    def inject(self, injection: FaultInjection) -> None:
        """Break the unit (FR-31).

        :raises ValueError: if the fault has no meaning for an actuator. It
            has no range to leave and no value to freeze at, and an injection
            that appeared to work while doing nothing would turn a detection
            trial into a phantom missed detection.
        """
        if injection.kind not in ACTUATOR_SUPPORTED_FAULTS:
            supported = sorted(kind.value for kind in ACTUATOR_SUPPORTED_FAULTS)
            raise ValueError(
                f"{type(self).__name__} cannot inject {injection.kind.value}; "
                f"supported: {supported}"
            )
        self._injection = injection
        if injection.kind is InjectedFault.STUCK_OFF:
            # Stuck *off* means off. Ignoring new commands is not enough: a
            # unit broken while it happened to be running would keep running,
            # which is a different fault from the one asked for -- and it made
            # a dead air conditioner look better at cooling than a healthy one
            # when the detector was measured against it.
            self._pending.clear()
            self._applied_fraction = COOLING_OFF

    def clear(self) -> None:
        """Stop injecting. The unit returns to its nominal imperfection."""
        self._injection = NO_FAULT

    @property
    def cooling_fraction(self) -> float:
        """The drive the room receives, as of the last ``apply_due_commands``.

        Zero while a failure is injected, however the unit was commanded: the
        room stops responding but the commands keep being accepted, which is
        exactly the fault D5 exists to catch (FR-24).
        """
        if self._failed:
            return COOLING_OFF
        return self._applied_fraction

    def inject_failure(self, failed: bool) -> None:
        """Make the unit stop affecting the room while still accepting commands."""
        self._failed = failed

    def apply_due_commands(self) -> None:
        """Promote every command whose dead time has elapsed, in order.

        Called once per simulation step. Applying them in order rather than
        keeping only the newest means a command issued inside another's dead
        time still takes effect first, as a real unit would do.
        """
        now = self._clock.now()
        while self._pending and self._pending[0].effective_ts <= now:
            self._applied_fraction = self._pending.popleft().cooling_fraction

    def command(self, kind: CommandKind, setpoint_c: float | None = None) -> AckStatus:
        """Send a command and report what is known about its fate.

        :returns: ACKNOWLEDGED only if the unit has readback and the command
            arrived; FAILED if it was lost on a unit with readback; UNKNOWN
            whenever the unit has no readback at all.
        """
        self._last_command_ts = self._clock.now()

        if self._injection.kind is InjectedFault.STUCK_OFF:
            # The unit takes the command and does nothing with it. On an
            # open-loop path this is indistinguishable from working, which is
            # the whole reason D5 tests the room rather than the reply (R-02).
            return self._acknowledge(arrived=True)

        if kind in _NO_CHANGE_COMMANDS:
            return self._acknowledge(arrived=True)

        if self._rng.random() < self._config.command_loss_probability:
            return self._acknowledge(arrived=False)

        fraction = COOLING_OFF if kind in _STOPPING_COMMANDS else COOLING_ON
        self._pending.append(
            _PendingCommand(
                effective_ts=self._clock.now() + self._config.dead_time_s,
                cooling_fraction=fraction,
            )
        )
        return self._acknowledge(arrived=True)

    def _acknowledge(self, arrived: bool) -> AckStatus:
        if not self._config.acknowledges:
            return AckStatus.UNKNOWN
        return AckStatus.ACKNOWLEDGED if arrived else AckStatus.FAILED
