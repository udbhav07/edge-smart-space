"""The blackboard: typed publish and subscribe over MQTT.

Every component is an MQTT client and none holds a reference to another
(DESIGN.md section 4.4). That buys three things this project needs: any
component can be killed and restarted independently (FR-11, FR-47), the
whole system state is inspectable with ``mosquitto_sub`` and no debugger
(FR-60), and simulated and real components are interchangeable at the topic
level (FR-15, NFR-09).

Two rules are enforced here rather than left to callers:

* **Delivery is the topic's business.** QoS and retention come from the
  :class:`~src.common.topics.TopicSpec`, never from the call site, so a
  retained state topic cannot accidentally be published transient.
* **A malformed payload never reaches Layer 2.** Decoding happens at this
  boundary and a payload that fails its schema is logged and dropped. Fail
  fast at the edge; degrade gracefully inside.

The transport is injectable, so every component above it is testable with no
broker running.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Protocol, TypeVar

from pydantic import ValidationError

from src.common.config import MqttConfig
from src.common.schemas import BlackboardMessage
from src.common.topics import TopicSpec

LOGGER = logging.getLogger(__name__)

MessageT = TypeVar("MessageT", bound=BlackboardMessage)

#: Payloads are UTF-8 JSON so the tree stays readable with mosquitto_sub.
PAYLOAD_ENCODING = "utf-8"


class Transport(Protocol):
    """The slice of an MQTT client this module uses.

    Narrow on purpose: a fake transport is a handful of lines, so no test
    above this module needs a broker.
    """

    def connect(self, host: str, port: int, keepalive: int) -> None: ...

    def publish(self, topic: str, payload: bytes, qos: int, retain: bool) -> None: ...

    def subscribe(self, topic: str, qos: int) -> None: ...

    def loop_start(self) -> None: ...

    def loop_stop(self) -> None: ...

    def disconnect(self) -> None: ...


class Blackboard:
    """Typed access to the shared state store.

    Handlers are registered per topic pattern and receive already-validated
    messages. A handler is never called with a payload that failed its
    schema, and a handler raising does not take the client down: one
    component's bug must not silence the bus for everyone else.
    """

    def __init__(self, config: MqttConfig, transport: Transport) -> None:
        self._config = config
        self._transport = transport
        self._handlers: dict[str, tuple[type[BlackboardMessage], Callable]] = {}
        self._subscriptions: dict[str, int] = {}
        self._connected = False

    def publish(
        self, spec: TopicSpec, message: BlackboardMessage, **parameters: str
    ) -> str:
        """Publish a message with the delivery its topic specifies.

        :returns: the concrete topic published to, for logging and tests.
        :raises TopicParameterError: if the parameters do not fit the topic.
        """
        topic = spec.format(**parameters)
        payload = message.model_dump_json(by_alias=True).encode(PAYLOAD_ENCODING)
        self._transport.publish(topic, payload, spec.qos.value, spec.retain)
        return topic

    def subscribe(
        self,
        spec: TopicSpec,
        schema: type[MessageT],
        handler: Callable[[str, MessageT], None],
    ) -> str:
        """Register a handler for every message matching a topic pattern.

        Registering is not the same as subscribing. MQTT will not accept a
        SUBSCRIBE before the connection is up, and it drops every
        subscription when the connection goes down, so what is recorded here
        is the *intent*. :meth:`on_connected` turns it into an actual
        subscription, every time the link comes up.

        :returns: the wildcard subscription string used.
        """
        pattern = spec.wildcard()
        self._handlers[pattern] = (schema, handler)
        self._subscriptions[pattern] = spec.qos.value
        if self._connected:
            self._transport.subscribe(pattern, spec.qos.value)
        return pattern

    def on_connected(self) -> None:
        """Issue every registered subscription. Called on each connect.

        A reconnect starts a fresh session, so subscriptions have to be sent
        again. Without this a component keeps running, keeps looking healthy,
        and silently stops receiving anything after the first blip.
        """
        self._connected = True
        for pattern, qos in self._subscriptions.items():
            self._transport.subscribe(pattern, qos)
        if self._subscriptions:
            LOGGER.info("subscribed to %d topic(s)", len(self._subscriptions))

    def on_disconnected(self) -> None:
        """Note that the link is down; subscriptions will need re-issuing."""
        self._connected = False

    @property
    def subscriptions(self) -> tuple[str, ...]:
        """Patterns this client wants, whether or not the link is up."""
        return tuple(self._subscriptions)

    def start(self) -> None:
        """Connect and begin serving callbacks."""
        self._transport.connect(
            self._config.host, self._config.port, int(self._config.keepalive_s)
        )
        self._transport.loop_start()

    def stop(self) -> None:
        """Stop serving and disconnect."""
        self._connected = False
        self._transport.loop_stop()
        self._transport.disconnect()

    def dispatch(self, topic: str, payload: bytes) -> None:
        """Route one received message. Called by the transport's callback.

        Decodes against the subscribing handler's schema and drops anything
        that fails, so no component above this boundary has to defend itself
        against a malformed publisher.
        """
        registration = self._match(topic)
        if registration is None:
            LOGGER.debug("no handler for %s", topic)
            return

        schema, handler = registration
        try:
            message = schema.model_validate_json(payload.decode(PAYLOAD_ENCODING))
        except (ValidationError, UnicodeDecodeError) as exc:
            LOGGER.warning("dropping malformed payload on %s: %s", topic, exc)
            return

        try:
            handler(topic, message)
        except Exception:
            # A handler's bug must not silence the bus for every other
            # component sharing this client.
            LOGGER.exception("handler for %s raised", topic)

    def _match(self, topic: str) -> tuple[type[BlackboardMessage], Callable] | None:
        """Find the handler whose pattern matches, honouring MQTT wildcards."""
        for pattern, registration in self._handlers.items():
            if topic_matches(pattern, topic):
                return registration
        return None


def topic_matches(pattern: str, topic: str) -> bool:
    """MQTT topic matching for ``+`` and a trailing ``#``."""
    if pattern == topic:
        return True

    pattern_levels = pattern.split("/")
    topic_levels = topic.split("/")

    for index, expected in enumerate(pattern_levels):
        if expected == "#":
            return index <= len(topic_levels)
        if index >= len(topic_levels):
            return False
        if expected != "+" and expected != topic_levels[index]:
            return False

    return len(pattern_levels) == len(topic_levels)


def build_transport(config: MqttConfig, client_id: str, blackboard_ref: list):
    """Create a paho client wired to a blackboard's dispatch.

    paho is imported here rather than at module scope so the routing and
    decoding logic above stays importable and testable without it.

    ``blackboard_ref`` is a single-element list holding the Blackboard, which
    breaks the construction cycle: the transport needs a dispatch target and
    the blackboard needs a transport.
    """
    import paho.mqtt.client as mqtt

    # MQTT client ids must be unique on a broker. Two clients sharing one id
    # do not coexist: the broker disconnects the first when the second
    # arrives, which produces an endless connect/disconnect loop rather than
    # an error anyone can read. The suffix makes a stale process harmless.
    unique_id = f"{client_id}-{os.getpid()}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=unique_id)
    client.reconnect_delay_set(
        min_delay=int(config.reconnect_min_s), max_delay=int(config.reconnect_max_s)
    )

    def _on_message(_client, _userdata, message) -> None:
        blackboard_ref[0].dispatch(message.topic, message.payload)

    def _on_connect(_client, _userdata, _flags, reason_code, _properties=None) -> None:
        LOGGER.info("connected to the broker: %s", reason_code)
        # Subscriptions do not survive a reconnect, and one issued before the
        # link came up was never accepted at all. Re-issue them here.
        blackboard_ref[0].on_connected()

    def _on_disconnect(
        _client, _userdata, _flags=None, reason_code=None, _properties=None
    ) -> None:
        # paho reconnects on its own with the backoff set above; the
        # regulatory loop holds its last validated setpoint meanwhile.
        LOGGER.warning("disconnected from the broker: %s", reason_code)
        blackboard_ref[0].on_disconnected()

    client.on_message = _on_message
    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect
    return client
