"""The assistance executor: the one process that holds the providers (section 5.7.6).

Every tool call crosses the blackboard, including the ones the reasoning layer
is entitled to make on its own. The executor listens on
``space/assist/proposed`` and ``space/assist/confirmed``, runs each invocation
through the :class:`~src.common.tools.ToolRegistry` -- argument check, the
confirmation gate, expiry, failure containment -- and publishes every outcome
to ``space/assist/result``, refusals included (FR-73, FR-74, FR-75).

Whether an invocation was confirmed is decided by the topic it arrived on and
nothing else. A field in the message would be something a publisher could set
for itself; a topic is something the console chooses to publish on only after
a person has said yes.

A confirmed booking runs at most once. The confirmation topic is not retained,
but a console that republishes twice -- a double click, a reconnect -- would
otherwise book twice, and that is precisely the kind of thing a ``commit``
tool exists to guard against.
"""

from __future__ import annotations

import logging
from collections import deque

from src.common import topics
from src.common.clock import Clock
from src.common.mqtt_client import Blackboard
from src.common.tools import ToolInvocation, ToolRegistry, ToolResult, ToolStatus

LOGGER = logging.getLogger(__name__)

#: Confirmed invocation ids remembered to refuse a repeat. Bounded; an id
#: older than this many confirmations is long past its expiry anyway.
_REMEMBERED_CONFIRMATIONS = 256


class AssistanceExecutor:
    """Runs tool invocations published on the blackboard."""

    def __init__(self, clock: Clock, blackboard: Blackboard, registry: ToolRegistry) -> None:
        self._clock = clock
        self._blackboard = blackboard
        self._registry = registry
        self._confirmed: deque[str] = deque(maxlen=_REMEMBERED_CONFIRMATIONS)

    def subscribe(self) -> None:
        self._blackboard.subscribe(
            topics.ASSIST_PROPOSED, ToolInvocation, self._on_proposed
        )
        self._blackboard.subscribe(
            topics.ASSIST_CONFIRMED, ToolInvocation, self._on_confirmed
        )

    def publish_catalogue(self) -> None:
        """Retain the declared surface, so what may be asked for is readable (FR-60)."""
        self._blackboard.publish(topics.ASSIST_CATALOGUE, self._registry.catalogue())

    def _on_proposed(self, _topic: str, invocation: ToolInvocation) -> None:
        self._run(invocation, confirmed=False)

    def _on_confirmed(self, _topic: str, invocation: ToolInvocation) -> None:
        if invocation.invocation_id in self._confirmed:
            LOGGER.warning(
                "ignoring a repeated confirmation of %s", invocation.invocation_id
            )
            return
        self._confirmed.append(invocation.invocation_id)
        self._run(invocation, confirmed=True)

    def _run(self, invocation: ToolInvocation, *, confirmed: bool) -> ToolResult:
        result = self._registry.invoke(invocation, confirmed=confirmed)
        self._blackboard.publish(topics.ASSIST_RESULT, result)
        level = logging.INFO if result.status is ToolStatus.OK else logging.WARNING
        LOGGER.log(
            level,
            "%s %s -> %s%s: %s",
            invocation.invocation_id,
            invocation.tool,
            result.status.value,
            " (simulated)" if result.simulated else "",
            result.message,
        )
        return result
