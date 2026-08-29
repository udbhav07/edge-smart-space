"""An in-process blackboard, for experiments that need no broker.

Publishing is delivered straight back to the same client's dispatch, so a
simulator and a subscriber wired through it talk over exactly the topics and
payloads they would use against mosquitto, with the network removed.

That matters for an experiment: E1 has to exercise the real path -- the
schemas, the topic names, the estimator's sample-interval checks -- and not
a shortcut that calls the estimator directly. A result obtained by bypassing
the plumbing tells you nothing about the system that will actually run.

Only ``eval`` and ``tools`` may use this. It is a test double, not a
transport: nothing in ``src`` or ``sim`` should ever import it.
"""

from __future__ import annotations

from src.common.mqtt_client import Blackboard


class LoopbackTransport:
    """Delivers everything published straight back to the subscribers.

    Ordering is synchronous and depth-first: a publish completes only once
    every handler it triggered has run. Real MQTT is neither, so a scenario
    that depends on this ordering is a scenario that will behave differently
    against a broker.
    """

    def __init__(self) -> None:
        self._blackboards: list[Blackboard] = []
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.delivered = 0

    def attach(self, blackboard: Blackboard) -> None:
        """Register a client to receive what anyone publishes."""
        self._blackboards.append(blackboard)

    def connect(self, host: str, port: int, keepalive: int) -> None: ...

    def publish(self, topic: str, payload: bytes, qos: int, retain: bool) -> None:
        self.published.append((topic, payload, qos, retain))
        for blackboard in self._blackboards:
            blackboard.dispatch(topic, payload)
            self.delivered += 1

    def subscribe(self, topic: str, qos: int) -> None: ...

    def loop_start(self) -> None: ...

    def loop_stop(self) -> None: ...

    def disconnect(self) -> None: ...

    def count_on(self, topic: str) -> int:
        return sum(1 for name, _, _, _ in self.published if name == topic)
