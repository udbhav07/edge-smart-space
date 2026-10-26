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
import queue
import threading
from collections import deque
from collections.abc import Callable

from src.common import topics
from src.common.clock import Clock
from src.common.mqtt_client import Blackboard
from src.common.tools import ToolInvocation, ToolRegistry, ToolResult, ToolStatus

LOGGER = logging.getLogger(__name__)

#: Confirmed invocation ids remembered to refuse a repeat. Bounded; an id
#: older than this many confirmations is long past its expiry anyway.
_REMEMBERED_CONFIRMATIONS = 256

#: Invocations waiting for the worker. Bounded: a flood is refused, not queued forever.
_WORK_QUEUE = 32

#: Runs a job somewhere: on the calling thread, or on a worker.
Dispatch = Callable[[Callable[[], None]], None]


class Worker:
    """Runs invocations off the network thread, one at a time, in order.

    A device's provider publishes a command and waits for the device's answer
    (FR-87). That answer arrives through the same network loop that delivered
    the invocation, so waiting for it there would wait forever: the executor
    hands each invocation to this worker instead.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=_WORK_QUEUE)
        self._thread = threading.Thread(target=self._serve, name="assistance-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def __call__(self, job: Callable[[], None]) -> None:
        """:raises queue.Full: if the worker is this far behind."""
        self._jobs.put_nowait(job)

    def _serve(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                job()
            except Exception:
                # The registry contains provider failures; anything reaching
                # here is a bug in the executor, and must not stop the worker.
                LOGGER.exception("an invocation crashed the worker's job")


class AssistanceExecutor:
    """Runs tool invocations published on the blackboard."""

    def __init__(
        self, clock: Clock, blackboard: Blackboard, registry: ToolRegistry, dispatch: Dispatch | None = None
    ) -> None:
        """
        :param dispatch: where invocations run. The service passes a
            :class:`Worker`; left out, they run on the calling thread, which
            is what an in-process blackboard needs.
        """
        self._clock = clock
        self._blackboard = blackboard
        self._registry = registry
        self._dispatch = dispatch or (lambda job: job())
        self._confirmed: deque[str] = deque(maxlen=_REMEMBERED_CONFIRMATIONS)

    def subscribe(self) -> None:
        self._blackboard.subscribe(
            topics.ASSIST_PROPOSED, ToolInvocation, self._on_proposed
        )
        self._blackboard.subscribe(
            topics.ASSIST_CONFIRMED, ToolInvocation, self._on_confirmed
        )

    @property
    def registry(self) -> ToolRegistry:
        """The tools this executor runs, for what declares tools at run time (FR-87)."""
        return self._registry

    def publish_catalogue(self) -> None:
        """Retain the declared surface, so what may be asked for is readable (FR-60)."""
        self._blackboard.publish(topics.ASSIST_CATALOGUE, self._registry.catalogue())

    def _on_proposed(self, _topic: str, invocation: ToolInvocation) -> None:
        self._submit(invocation, confirmed=False)

    def _on_confirmed(self, _topic: str, invocation: ToolInvocation) -> None:
        if invocation.invocation_id in self._confirmed:
            LOGGER.warning(
                "ignoring a repeated confirmation of %s", invocation.invocation_id
            )
            return
        self._confirmed.append(invocation.invocation_id)
        self._submit(invocation, confirmed=True)

    def _submit(self, invocation: ToolInvocation, *, confirmed: bool) -> None:
        try:
            self._dispatch(lambda: self._run(invocation, confirmed=confirmed))
        except queue.Full:
            LOGGER.warning("refusing %s: the executor is %d invocations behind", invocation.invocation_id, _WORK_QUEUE)
            self._blackboard.publish(
                topics.ASSIST_RESULT,
                ToolResult(
                    ts=self._clock.now(),
                    invocation_id=invocation.invocation_id,
                    tool=invocation.tool,
                    status=ToolStatus.UNAVAILABLE,
                    message="Too much is being asked at once; please try again in a moment.",
                ),
            )

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
