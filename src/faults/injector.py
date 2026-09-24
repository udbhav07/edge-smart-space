"""Asks Layer 1 to misbehave, so detection can be demonstrated (FR-31).

An examiner asks for a stuck sensor and it happens, with no code edit and no
restart. That is the whole requirement, and it is why this exists as a
component rather than as a test fixture: a fault that can only be produced
from a test has not been shown to be detectable in the running system.

The injector publishes and nothing else. It does not know whether a simulator
or an ESP32 adapter is listening, it does not confirm that anything took
effect, and it cannot: the adapter obeys, the detectors notice a reading that
is absent, frozen or implausible, and the evidence of success is the fault the
bank raises. An injector that reported its own success would be reporting
that a message was sent.

**A magnitude that would be ignored is refused.** Injecting DROPOUT with a
value, or STUCK_AT without one, is a request nobody can act on sensibly.
Accepted silently, both look exactly like a fault that failed to take effect,
and the demonstration becomes a missed detection nobody can explain.
"""

from __future__ import annotations

import logging

from src.common import topics
from src.common.clock import Clock
from src.common.injection import FAULTS_REQUIRING_MAGNITUDE, InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import InjectionCommand

LOGGER = logging.getLogger(__name__)

#: Who is asking, when nobody says. Injection is an operator action by
#: definition: nothing in the system injects a fault into itself.
DEFAULT_REQUESTER = "operator"


class FaultInjector:
    """Publishes injection commands to Layer 1."""

    def __init__(
        self,
        blackboard: Blackboard,
        clock: Clock,
        requester: str = DEFAULT_REQUESTER,
    ) -> None:
        self._blackboard = blackboard
        self._clock = clock
        self._requester = requester

    def inject(
        self,
        subject: str,
        kind: InjectedFault,
        magnitude: float | None = None,
    ) -> InjectionCommand:
        """Ask the adapter for ``subject`` to start producing this fault.

        :param magnitude: the frozen value for STUCK_AT, the reported value
            for OUT_OF_RANGE, degrees per second for DRIFT. Must be given for
            those and omitted for the rest.
        :returns: the command published, so a caller can report exactly what
            it asked for rather than what it meant to ask for.
        :raises ValueError: if the magnitude is missing where it carries the
            fault's whole content, or supplied where it would be ignored.
        """
        needs_magnitude = kind in FAULTS_REQUIRING_MAGNITUDE
        if needs_magnitude and magnitude is None:
            raise ValueError(
                f"{kind.value} is defined by its magnitude; say how hard"
            )
        if not needs_magnitude and magnitude is not None:
            raise ValueError(
                f"{kind.value} ignores a magnitude, so {magnitude!r} would "
                f"have no effect; omit it"
            )

        command = InjectionCommand(
            ts=self._clock.now(),
            subject=subject,
            kind=kind,
            magnitude=magnitude if magnitude is not None else 0.0,
            requester=self._requester,
        )
        self._publish(command)
        return command

    def clear(self, subject: str) -> InjectionCommand:
        """Ask the adapter for ``subject`` to stop injecting (FR-30).

        Clearing is an injection of NONE rather than a withdrawal of the
        retained message. The difference matters: a withdrawal leaves a
        restarted adapter with nothing to read, while a retained NONE says
        plainly that nothing is being injected and that somebody decided so.
        """
        return self.inject(subject, InjectedFault.NONE)

    def _publish(self, command: InjectionCommand) -> None:
        topic = self._blackboard.publish(
            topics.INJECT, command, subject=command.subject
        )
        LOGGER.info(
            "injected %s on %s (magnitude %.4g) via %s",
            command.kind.value,
            command.subject,
            command.magnitude,
            topic,
        )
