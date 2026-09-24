"""Runs tool invocations, and is the only thing that may (FR-71 to FR-75).

The reasoning layer proposes; this executes. Between them is a topic, and the
separation is the point: ``src/reasoning/`` cannot import a provider, so a
change of calendar is a change of wiring rather than a change to the model's
surroundings, and FR-74's confirmation gate cannot be skipped by calling past
it.

**Confirmation is carried by the topic, never by a field.** An invocation on
``space/assist/proposed`` is a request; the same invocation republished on
``space/assist/confirmed`` is one an occupant has agreed to. A flag inside the
message would be something the publisher could set for itself, and the one
gate that protects an occupant from a model would be advisory.

**Every outcome is published, refusals included** (FR-75). A tool that was
refused, expired, failed, or was never bound produces a result on
``space/assist/result`` saying so. Silence would leave whoever asked waiting,
and would leave the audit trail with a hole exactly where a decision was made.

**The catalogue is retained** (FR-70), so an examiner can read what the
reasoning layer is permitted to ask for without running it.
"""

from __future__ import annotations

import logging

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.mqtt_client import Blackboard
from src.common.tools import ToolInvocation, ToolRegistry, ToolResult

LOGGER = logging.getLogger(__name__)


class AssistanceExecutor:
    """One executor, holding the registry and the providers bound into it."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        registry: ToolRegistry,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._registry = registry
        self._results = 0

    @property
    def results_published(self) -> int:
        """How many outcomes have been reported. For tests and the console."""
        return self._results

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        """Listen on both invocation topics.

        Two subscriptions rather than one with a flag, because which topic a
        message arrived on *is* the authorisation. Collapsing them into one
        handler that read a field would be the same mistake in a different
        place.
        """
        self._blackboard.subscribe(
            topics.ASSIST_PROPOSED, ToolInvocation, self._on_proposed
        )
        self._blackboard.subscribe(
            topics.ASSIST_CONFIRMED, ToolInvocation, self._on_confirmed
        )

    def publish_catalogue(self) -> None:
        """Announce the declared surface (FR-70).

        Retained, so a late subscriber -- or an examiner with
        ``mosquitto_sub`` -- can read what the reasoning layer is allowed to
        ask for without waiting for it to ask for something.
        """
        catalogue = self._registry.catalogue()
        self._blackboard.publish(topics.ASSIST_CATALOGUE, catalogue)
        LOGGER.info(
            "declared %d tool(s): %s",
            len(catalogue.tools),
            ", ".join(spec.name for spec in catalogue.tools),
        )

    # --- invocations --------------------------------------------------

    def _on_proposed(self, _topic: str, invocation: ToolInvocation) -> None:
        """An invocation nobody has agreed to yet."""
        self._execute(invocation, confirmed=False)

    def _on_confirmed(self, _topic: str, invocation: ToolInvocation) -> None:
        """An invocation an occupant has agreed to (FR-74).

        The agreement is the topic. Whoever obtained it republished the
        invocation here, and that act is what this reads -- not a claim inside
        the message.
        """
        LOGGER.info("confirmed invocation %s", invocation.invocation_id)
        self._execute(invocation, confirmed=True)

    def _execute(self, invocation: ToolInvocation, *, confirmed: bool) -> ToolResult:
        """Run it and publish whatever happened.

        The registry never raises for a refusal, so there is no failure path
        here that skips publication. That is deliberate: an exception escaping
        into this loop would reach nobody, and FR-75 asks for every outcome to
        reach the blackboard.
        """
        result = self._registry.invoke(invocation, confirmed=confirmed)
        self._blackboard.publish(topics.ASSIST_RESULT, result)
        self._results += 1
        LOGGER.info(
            "%s -> %s (%s)%s",
            invocation.tool,
            result.status.value,
            result.provider or "no provider",
            " [simulated]" if result.simulated else "",
        )
        return result
