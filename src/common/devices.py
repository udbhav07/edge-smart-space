"""The contract every device service speaks on the blackboard (FR-87, FR-88).

A device -- a fan, a music player, a lamp behind an MCP server, the air
conditioner -- is a service of its own. It says what it is and what it can do
in a retained :class:`DeviceManifest`, says whether it is reachable in a
retained :class:`DeviceAvailability` (its last will says ``offline`` if it
dies), takes a :class:`~src.common.tools.ToolInvocation` on its ``command``
topic, answers with a :class:`~src.common.tools.ToolResult` on ``result``, and
keeps its current values in a retained :class:`DeviceState`.

Nothing here knows how a device is driven. A Python service, an ESPHome node
publishing the same JSON, or an MCP server behind the bridge are the same
thing to the rest of the system, which is what makes adding one a matter of
publishing a manifest rather than changing the core.

Why a capability carries a :class:`DeviceClass` as well as a
:class:`~src.common.tools.ToolEffect`: the effect says whose it is to undo
(the confirmation gate, FR-54), the class says what it can do to the room.
The air conditioner's capabilities are ``PLANT`` and never become tools: the
thermal loop is reached only through a preference, a goal and the safety
validator (FR-45), however a device describes itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType

from pydantic import Field, field_serializer, field_validator, model_validator

from src.common import topics
from src.common.clock import Clock
from src.common.schemas import ActuatorState, TimestampedMessage
from src.common.tools import ArgumentValue, ParameterType, ToolEffect, ToolParameter, ToolSpec

#: A device id becomes part of a tool name (``fan_set_speed``) and a topic
#: level, so it is one lower-case identifier, like a tool name.
DEVICE_ID_PATTERN = r"^[a-z][a-z0-9_]*$"

#: The joiner between a device id and a capability in the tool a model sees.
TOOL_JOINER = "_"


class DeviceClass(str, Enum):
    """What a capability can do to the room. Required, with no default.

    ``PLANT``
        Changes the thermal plant the regulatory loop controls. Never offered
        as a tool: reached only as a preference the validator gates (FR-45).
    ``COMFORT``
        Physical, outside the thermal model: a fan, a light. Run when asked,
        within the declared parameter bounds and the device's rate limit.
    ``MEDIA``
        Sound or display. Nothing physical beyond itself; parameters checked.
    """

    PLANT = "plant"
    COMFORT = "comfort"
    MEDIA = "media"


class DeviceKind(str, Enum):
    """What sort of thing a device is, so a console knows how to show it."""

    AC = "ac"
    FAN = "fan"
    LIGHT = "light"
    MEDIA = "media"
    OTHER = "other"


class DeviceCapability(ToolSpec):
    """One thing a device can do: a tool declaration, plus what it does to the room."""

    device_class: DeviceClass
    min_interval_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Shortest time between two runs of this capability, in s; "
        "a fan switched ten times a second is a fault, not a wish",
    )


class DeviceManifest(TimestampedMessage):
    """What a device is and what it can do. Retained on ``space/device/{id}/manifest``."""

    device_id: str = Field(pattern=DEVICE_ID_PATTERN)
    name: str = Field(min_length=1, description="What the occupant calls it: 'Ceiling fan'")
    kind: DeviceKind
    room: str = Field(default="", description="Where it is, if anywhere in particular")
    aliases: tuple[str, ...] = Field(
        default=(), description="Other words for it: 'the fan', 'ceiling fan'"
    )
    simulated: bool = Field(
        description="Whether it acts on anything real; no default (FR-15)"
    )
    provider: str = Field(
        default="native", description="'native', or 'mcp:<server>' behind the bridge"
    )
    capabilities: tuple[DeviceCapability, ...] = ()

    @model_validator(mode="after")
    def _capability_names_are_unique(self) -> DeviceManifest:
        names = [capability.name for capability in self.capabilities]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.device_id!r} declares a capability twice: {names}")
        return self

    def capability(self, name: str) -> DeviceCapability | None:
        """The declaration of one capability, or None if it has none of that name."""
        return next((capability for capability in self.capabilities if capability.name == name), None)

    def tool_name(self, capability: str) -> str:
        """The name a model sees for one of this device's capabilities."""
        return f"{self.device_id}{TOOL_JOINER}{capability}"


class DeviceAvailability(TimestampedMessage):
    """Whether a device's service is reachable. Retained; its last will says no."""

    device_id: str = Field(pattern=DEVICE_ID_PATTERN)
    online: bool


class DeviceState(TimestampedMessage):
    """A device's current values. Retained on ``space/device/{id}/state``.

    Flat values only, like a tool result's detail: a nested structure would
    not render on the board, and nothing a device reports needs one.
    """

    device_id: str = Field(pattern=DEVICE_ID_PATTERN)
    values: Mapping[str, ArgumentValue] = Field(default_factory=dict)
    simulated: bool

    @field_validator("values", mode="after")
    @classmethod
    def _freeze_values(cls, value: Mapping[str, ArgumentValue]) -> Mapping[str, ArgumentValue]:
        return MappingProxyType(dict(value))

    @field_serializer("values")
    def _serialise_values(self, value: Mapping[str, ArgumentValue]) -> dict[str, ArgumentValue]:
        return dict(value)


def last_will(device_id: str, clock: Clock) -> tuple[str, bytes, bool]:
    """What the broker publishes for a device whose service dies: offline, retained.

    For :func:`~src.common.mqtt_client.build_transport`'s ``will``. Its
    timestamp is the moment the service started, which is the last moment
    it is known to have been alive.
    """
    message = DeviceAvailability(ts=clock.now(), device_id=device_id, online=False)
    topic = topics.DEVICE_AVAILABILITY.format(device_id=device_id)
    return topic, message.model_dump_json().encode("utf-8"), topics.DEVICE_AVAILABILITY.retain


# --- The air conditioner, on the same contract (FR-87, FR-88, FR-45) -----------


def air_conditioner_manifest(now: float, simulated: bool) -> DeviceManifest:
    """The AC as a device: visible like any other, and every capability ``PLANT``.

    None of these becomes a tool. They are here so the console and the
    assistant can see the unit and what it is, and so a request naming it is
    recognised as being about the room's temperature -- which reaches the unit
    only as a preference the validator gates (FR-45).
    """

    def plant(name: str, purpose: str, *parameters: ToolParameter) -> DeviceCapability:
        return DeviceCapability(name=name, purpose=purpose, effect=ToolEffect.WRITE, device_class=DeviceClass.PLANT, parameters=parameters)

    return DeviceManifest(
        ts=now,
        device_id=topics.AIR_CONDITIONER_ID,
        name="Air conditioner",
        kind=DeviceKind.AC,
        aliases=("the ac", "air conditioner", "aircon", "a/c", "cooling"),
        simulated=simulated,
        capabilities=(
            plant(
                "set_temperature",
                "Set the room's temperature.",
                ToolParameter(name="target_c", type=ParameterType.NUMBER, description="Target in degrees Celsius"),
            ),
            plant("turn_off", "Turn the air conditioner off."),
        ),
    )


def air_conditioner_state(state: ActuatorState) -> DeviceState:
    """The unit's actuator state, as device state."""
    return DeviceState(
        ts=state.ts,
        device_id=state.actuator_id,
        values={"command": state.kind.value, "ack": state.ack.value, "last_command_ts": state.last_command_ts},
        simulated=state.simulated,
    )


class PlantAnnouncer:
    """Keeps the AC on the device contract for whichever process owns it.

    The simulator or the hardware bridge owns the unit; each calls
    :meth:`mirror` with every actuator state it publishes. The manifest and
    ``online`` go out whenever the link has (re)connected, since the broker
    publishes the last will -- offline -- whenever it drops.
    """

    def __init__(self, blackboard, clock: Clock, simulated: bool) -> None:
        self._blackboard = blackboard
        self._clock = clock
        self._simulated = simulated
        self._announced = False

    def mirror(self, state: ActuatorState) -> None:
        if not self._blackboard.connected:
            self._announced = False
            return
        if not self._announced:
            now = self._clock.now()
            self._blackboard.publish(topics.DEVICE_MANIFEST, air_conditioner_manifest(now, self._simulated), device_id=state.actuator_id)
            self._blackboard.publish(
                topics.DEVICE_AVAILABILITY, DeviceAvailability(ts=now, device_id=state.actuator_id, online=True), device_id=state.actuator_id
            )
            self._announced = True
        self._blackboard.publish(topics.DEVICE_STATE, air_conditioner_state(state), device_id=state.actuator_id)
