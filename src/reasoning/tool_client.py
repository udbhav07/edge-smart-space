"""How the reasoning layer asks for an assistance tool (section 5.7.6, Flow).

It publishes a ``ToolInvocation`` to ``space/assist/proposed`` and waits, up
to ``assistance.result_timeout_s``, for the correlated ``ToolResult`` on
``space/assist/result``. That is the whole of it. There is no import of a
provider anywhere in ``src/reasoning/`` -- a calendar is reached only through
the executor, so FR-73's argument check and FR-74's confirmation gate cannot
be skipped by calling past them (FR-71).

A result that does not arrive in time is ``None``, not an exception: an
executor that has stopped answering must not hold a reply to the occupant
open, and the invocation is still on the blackboard, unhonoured rather than
lost (section 7.1).
"""

from __future__ import annotations

import itertools
import logging
import threading
from collections import OrderedDict
from collections.abc import Mapping

from pydantic import ValidationError

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.mqtt_client import Blackboard
from src.common.tools import (
    ArgumentValue,
    ToolInvocation,
    ToolRequester,
    ToolResult,
)

LOGGER = logging.getLogger(__name__)

#: Most invocations awaiting a result at once. One utterance makes a handful;
#: this bounds the table if results stop arriving altogether.
MAX_PENDING = 16


class _Pending:
    def __init__(self) -> None:
        self.arrived = threading.Event()
        self.result: ToolResult | None = None


class BlackboardToolClient:
    """Invokes tools by publishing, and collects their results by listening."""

    def __init__(self, config: Config, clock: Clock, blackboard: Blackboard) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, _Pending] = OrderedDict()
        self._sequence = itertools.count()

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        with self._lock:
            pending = self._pending.get(result.invocation_id)
        if pending is None:
            return  # someone else's invocation, or one we stopped waiting for
        pending.result = result
        pending.arrived.set()

    def call(
        self,
        tool: str,
        arguments: Mapping[str, ArgumentValue],
        rationale: str,
    ) -> ToolResult | None:
        """Invoke one tool and wait for what came of it.

        :returns: the result, or None when none arrived in time.
        :raises ValueError: if the arguments cannot be carried at all -- a
            nested value, a name that is not a tool name. That is the model's
            mistake, reported to the model; nothing is published.
        """
        now = self._clock.now()
        try:
            invocation = ToolInvocation(
                ts=now,
                invocation_id=f"inv_{int(now)}_{next(self._sequence)}",
                tool=tool,
                arguments=dict(arguments),
                requester=ToolRequester.PERSONAL_CONTEXT,
                rationale=rationale,
                expires_ts=now + self._config.assistance.confirmation_window_s,
            )
        except ValidationError as exc:
            raise ValueError(f"cannot invoke {tool!r} with those arguments: {exc}") from exc

        pending = _Pending()
        with self._lock:
            # Registered before publishing: on the in-process bus the result
            # arrives during the publish call itself.
            self._pending[invocation.invocation_id] = pending
            while len(self._pending) > MAX_PENDING:
                self._pending.popitem(last=False)
        try:
            self._blackboard.publish(topics.ASSIST_PROPOSED, invocation)
            self._clock.wait_for(
                pending.arrived, self._config.assistance.result_timeout_s
            )
        finally:
            with self._lock:
                self._pending.pop(invocation.invocation_id, None)

        if pending.result is None:
            LOGGER.warning(
                "no result for %s (%s) within %.1f s",
                invocation.invocation_id,
                tool,
                self._config.assistance.result_timeout_s,
            )
        return pending.result
