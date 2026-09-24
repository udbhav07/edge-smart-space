"""Layer 1 on real hardware: devices in, blackboard out.

This is what runs instead of the simulator once there are sensors on a wall
(section 9.1). It publishes the same topics with the same schemas at the same
cadence, so nothing above Layer 1 can tell which is running -- which is the
property the whole architecture was arranged to get, and the reason the
transition is a configuration change rather than a rewrite.

**Device details stop here.** The topic an ESPHome node publishes on, and
whether its payload is a bare number or an object, are the only hardware facts
in the system, and they live in configuration rather than in code because they
are decided when a node is flashed rather than when this is written (R-03).

**A stale value is silence, not a reading.** A node that stops publishing must
look absent to everything above, because absence is what D1 detects. Repeating
the last value would turn a dropped node into a stuck one and send D1 looking
for a fault D2 would then find in the wrong place.

**Nothing here is confirmed by hardware yet.** The shape is written and tested
against a fake device publisher; what bring-up has to establish is in
``docs/DESIGN.md`` section 9.2 and in the deploy notes. Saying that plainly is
the point of R-03: code that looks finished and silently matches nothing is
worse than code that is obviously unfinished.
"""

from __future__ import annotations

import json
import logging

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.injection import (
    NO_FAULT,
    FaultInjection,
    InjectedFault,
)
from src.common.mqtt_client import PAYLOAD_ENCODING, Blackboard, Transport
from src.common.schemas import (
    AckStatus,
    ActuatorState,
    Command,
    CommandKind,
    InjectionCommand,
)
from src.io.sensor_adapters.base import SensorAdapter
from src.io.sensor_adapters.esphome import EsphomeSource

LOGGER = logging.getLogger(__name__)

#: Delivery for a device topic. Not taken from a TopicSpec, because a device
#: topic is not one of ours: how an ESPHome node wants to be addressed is a
#: device fact, and Layer 1 is the only layer permitted to hold one. At least
#: once and not retained is what a command wants -- a retained command would be
#: redelivered to the unit on every reconnect.
_DEVICE_QOS = 1
_DEVICE_RETAIN = False

#: Key looked for when a device publishes an object rather than a number.
#: ESPHome publishes a bare value by default; a node configured otherwise is
#: the case this covers, and which one a deployment uses is a bring-up fact.
_VALUE_KEY = "value"

#: What a binary device publishes. ESPHome uses these words for a binary
#: sensor's state rather than a number, so an occupancy node is read through
#: them and a temperature node never reaches this.
_TRUE_WORDS = frozenset({"ON", "TRUE", "1"})
_FALSE_WORDS = frozenset({"OFF", "FALSE", "0"})


class DevicePayloadError(ValueError):
    """A device published something that is not a reading."""


def parse_device_payload(payload: bytes) -> float:
    """Turn what a node published into a number.

    Three shapes are accepted because all three are configurations ESPHome
    actually produces: a bare number, an object carrying one, and the words a
    binary sensor uses. Which one a deployment emits is settled at bring-up,
    and accepting all three means settling it does not require a code change.

    :raises DevicePayloadError: if it is none of them. Raised rather than
        guessed: a device whose payload nobody understands is a device that
        must be noticed, and inventing a zero would publish a reading the
        instrument never took.
    """
    try:
        text = payload.decode(PAYLOAD_ENCODING).strip()
    except UnicodeDecodeError as exc:
        raise DevicePayloadError(f"undecodable device payload: {exc}") from exc

    if not text:
        raise DevicePayloadError("empty device payload")

    upper = text.upper()
    if upper in _TRUE_WORDS:
        return 1.0
    if upper in _FALSE_WORDS:
        return 0.0

    try:
        return float(text)
    except ValueError:
        pass

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DevicePayloadError(f"{text!r} is not a reading") from exc

    if isinstance(parsed, (int, float)):
        return float(parsed)
    if isinstance(parsed, dict) and _VALUE_KEY in parsed:
        try:
            return float(parsed[_VALUE_KEY])
        except (TypeError, ValueError) as exc:
            raise DevicePayloadError(
                f"{parsed[_VALUE_KEY]!r} under {_VALUE_KEY!r} is not a number"
            ) from exc
    raise DevicePayloadError(f"{text!r} carries no reading")


class HardwareLayer:
    """Every adapter and the one actuator, bound to their devices."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        adapters: dict[str, SensorAdapter],
        sources: dict[str, EsphomeSource],
        device_topics: dict[str, str],
        device_bus: Transport,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._adapters = adapters
        self._sources = sources
        self._device_topics = device_topics
        self._device_bus = device_bus
        self._last_kind = CommandKind.OFF
        self._last_command_ts: float | None = None
        self._injections: dict[str, FaultInjection] = {}

    @property
    def bound_sensors(self) -> frozenset[str]:
        """Sensors that have a device behind them."""
        return frozenset(self._adapters)

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        """Listen to the devices, to commands, and to injections."""
        for sensor_id, topic in self._device_topics.items():
            self._blackboard.subscribe_raw(
                topic, self._device_handler(sensor_id)
            )
        self._blackboard.subscribe(
            topics.ACTUATOR_COMMAND, Command, self._on_command
        )
        self._blackboard.subscribe(
            topics.INJECT, InjectionCommand, self._on_injection
        )

    def _device_handler(self, sensor_id: str):
        """A handler bound to one sensor.

        Raw rather than typed: a device publishes its own payload, not one of
        our schemas, and that payload is exactly the hardware detail Layer 1
        exists to absorb.
        """

        def handle(topic: str, payload: bytes) -> None:
            try:
                value = parse_device_payload(payload)
            except DevicePayloadError as exc:
                LOGGER.warning("%s on %s: %s", sensor_id, topic, exc)
                return
            self._sources[sensor_id].accept(value)

        return handle

    def _on_command(self, _topic: str, command: Command) -> None:
        """Send a command to the air conditioner and say what is known.

        Nothing is known, on an IR path. The command is forwarded and the
        acknowledgement is UNKNOWN, which is the correct answer rather than a
        missing one (R-02): D5 decides whether it worked by watching the room.
        """
        if command.actuator_id != topics.AIR_CONDITIONER_ID:
            return
        self._last_kind = command.kind
        self._last_command_ts = self._clock.now()
        self._device_bus.publish(
            self._config.io.actuator.command_topic,
            command.kind.value.encode(PAYLOAD_ENCODING),
            _DEVICE_QOS,
            _DEVICE_RETAIN,
        )
        LOGGER.debug("sent %s to the air conditioner", command.kind.value)

    def _on_injection(self, _topic: str, command: InjectionCommand) -> None:
        """Obey an injection on real hardware, exactly as the simulator does.

        FR-31 asks for the same mechanism in both places, and section 5.9.2
        injects at the adapter -- so a fault injected here reaches the
        detectors as an absent, frozen or implausible reading, the same as one
        the instrument suffered.
        """
        adapter = self._adapters.get(command.subject)
        if adapter is None:
            LOGGER.warning("no adapter for %s; injection ignored", command.subject)
            return
        if command.kind is InjectedFault.NONE:
            adapter.clear()
            return
        try:
            adapter.inject(
                FaultInjection(kind=command.kind, magnitude=command.magnitude)
            )
        except ValueError as exc:
            LOGGER.warning("injection on %s refused: %s", command.subject, exc)

    # --- the tick -----------------------------------------------------

    def poll(self) -> int:
        """Publish one round of readings and the actuator's state.

        :returns: how many sensors produced a reading. A count below the
            number of bound sensors is not an error here -- it is what a
            dropped node looks like, and D1 is the component whose job it is
            to have an opinion about that.
        """
        produced = sum(
            1 for adapter in self._adapters.values() if adapter.poll() is not None
        )
        self._publish_actuator_state()
        return produced

    def _publish_actuator_state(self) -> None:
        """Say what was last asked of the unit, and that nobody knows more."""
        acknowledges = self._config.io.actuator.acknowledges
        self._blackboard.publish(
            topics.ACTUATOR_STATE,
            ActuatorState(
                ts=self._clock.now(),
                actuator_id=topics.AIR_CONDITIONER_ID,
                simulated=False,
                kind=self._last_kind,
                setpoint_c=None,
                ack=(
                    AckStatus.ACKNOWLEDGED if acknowledges else AckStatus.UNKNOWN
                ),
                last_command_ts=self._last_command_ts,
            ),
            actuator_id=topics.AIR_CONDITIONER_ID,
        )


def build_layer(
    config: Config,
    clock: Clock,
    blackboard: Blackboard,
    device_bus: Transport,
) -> HardwareLayer:
    """Assemble an adapter per configured sensor that has a device behind it.

    A sensor with no device binding is left out rather than given a source
    that never speaks. The difference matters: an adapter with no device
    publishes silence, which D1 reads as a dropped sensor and reports as a
    fault that is really a configuration gap.
    """
    adapters: dict[str, SensorAdapter] = {}
    sources: dict[str, EsphomeSource] = {}
    device_topics: dict[str, str] = {}

    for sensor in config.sensors.adapters:
        topic = config.io.topic_for(sensor.sensor_id)
        if topic is None:
            LOGGER.warning(
                "%s has no device binding; it will not be published",
                sensor.sensor_id,
            )
            continue
        source = EsphomeSource(clock=clock, stale_after_s=config.io.stale_after_s)
        sources[sensor.sensor_id] = source
        device_topics[sensor.sensor_id] = topic
        adapters[sensor.sensor_id] = SensorAdapter(
            config=sensor, source=source, clock=clock, blackboard=blackboard
        )

    return HardwareLayer(
        config=config,
        clock=clock,
        blackboard=blackboard,
        adapters=adapters,
        sources=sources,
        device_topics=device_topics,
        device_bus=device_bus,
    )
