"""The part every device service shares (FR-87, FR-88).

A device is a manifest, a handler per capability, and a function reporting
its current values. :class:`DeviceService` does the rest: it announces the
device, answers its commands, refuses what the manifest does not allow, and
keeps its state on the board. Writing a new device is writing those three
things; see ``src/devices/fan`` for the smallest complete one.

The checks here repeat the ones the assistance service's registry makes
before a command is ever published. That is deliberate. The command topic is
open to anything on the broker, and a device must not run a fan at speed 9
because a publisher skipped the registry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from src.common import topics
from src.common.clock import Clock
from src.common.devices import DeviceAvailability, DeviceClass, DeviceManifest, DeviceState, last_will
from src.common.mqtt_client import Blackboard
from src.common.tools import ArgumentValue, ProviderOutcome, ToolArgumentError, ToolInvocation, ToolResult, ToolStatus

LOGGER = logging.getLogger(__name__)

#: Carries out one capability, given its checked arguments.
Handler = Callable[[Mapping[str, ArgumentValue]], ProviderOutcome]

#: Reports what the device is doing now.
StateReader = Callable[[], Mapping[str, ArgumentValue]]

#: Said when a handler fails for a reason it did not put into words.
_INTERNAL_FAILURE = "Something went wrong with the device; nothing was changed."


__all__ = ["DeviceService", "Handler", "StateReader", "last_will"]


class DeviceService:
    """Announces one device and answers its commands on the blackboard."""

    def __init__(
        self,
        manifest: DeviceManifest,
        handlers: Mapping[str, Handler],
        state: StateReader,
        clock: Clock,
        blackboard: Blackboard,
    ) -> None:
        """
        :raises ValueError: if a capability has no handler, or a handler no
            capability. Either is a device that would answer "unknown" to
            something it claims to do, found at startup rather than by an
            occupant.
        """
        declared = {capability.name for capability in manifest.capabilities if capability.device_class is not DeviceClass.PLANT}
        missing, extra = declared - set(handlers), set(handlers) - declared
        if missing or extra:
            raise ValueError(
                f"{manifest.device_id!r}: capabilities without a handler {sorted(missing)}, "
                f"handlers without a capability {sorted(extra)}"
            )
        self._manifest = manifest
        self._handlers = dict(handlers)
        self._state = state
        self._clock = clock
        self._blackboard = blackboard
        self._last_run: dict[str, float] = {}

    @property
    def device_id(self) -> str:
        return self._manifest.device_id

    def start(self) -> None:
        """Listen for commands, then say what the device is, that it is here, and how it is."""
        self._blackboard.subscribe(topics.DEVICE_COMMAND, ToolInvocation, self._on_command)
        self._blackboard.publish(topics.DEVICE_MANIFEST, self._manifest, device_id=self.device_id)
        self._set_online(True)
        self.publish_state()

    def stop(self) -> None:
        """Say the device is going away, rather than leave it to the last will."""
        self._set_online(False)

    def publish_state(self) -> None:
        """Put the device's current values on the board: after a command, or when they change by themselves."""
        values = dict(self._state())
        self._blackboard.publish(
            topics.DEVICE_STATE,
            DeviceState(ts=self._clock.now(), device_id=self.device_id, values=values, simulated=self._manifest.simulated),
            device_id=self.device_id,
        )

    def _set_online(self, online: bool) -> None:
        self._blackboard.publish(
            topics.DEVICE_AVAILABILITY,
            DeviceAvailability(ts=self._clock.now(), device_id=self.device_id, online=online),
            device_id=self.device_id,
        )

    def _on_command(self, topic: str, invocation: ToolInvocation) -> None:
        # One wildcard subscription serves every device on the broker; this one answers its own.
        if topic != topics.DEVICE_COMMAND.format(device_id=self.device_id):
            return
        result = self.handle(invocation)
        self._blackboard.publish(topics.DEVICE_RESULT, result, device_id=self.device_id)
        if result.status is ToolStatus.OK:
            self.publish_state()

    def handle(self, invocation: ToolInvocation) -> ToolResult:
        """Check one command against the manifest and run it. Never raises."""
        capability = self._manifest.capability(invocation.tool)
        if capability is None:
            return self._refuse(invocation, ToolStatus.UNKNOWN_TOOL, f"The {self._manifest.name.lower()} can't {invocation.tool.replace('_', ' ')}.")
        if capability.device_class is DeviceClass.PLANT:
            # The thermal loop is reached through a preference and the
            # validator, never a device command (FR-45, FR-88).
            return self._refuse(invocation, ToolStatus.UNAVAILABLE, f"The {self._manifest.name.lower()} is controlled through the room's temperature setting.")
        now = self._clock.now()
        if now > invocation.expires_ts:
            return self._refuse(invocation, ToolStatus.EXPIRED, "That request came too late to act on.")
        try:
            arguments = capability.validate_arguments(invocation.arguments)
        except ToolArgumentError as exc:
            return self._refuse(invocation, ToolStatus.BAD_ARGUMENTS, str(exc))
        last = self._last_run.get(capability.name)
        if last is not None and now - last < capability.min_interval_s:
            wait = capability.min_interval_s - (now - last)
            return self._refuse(invocation, ToolStatus.UNAVAILABLE, f"That was asked again too soon; try in {wait:.0f} s.")
        try:
            outcome = self._handlers[capability.name](arguments)
        except Exception as exc:
            # As the registry does: a broken device must not take its service
            # down, and a ValueError is a sentence written for the occupant.
            LOGGER.warning("%s failed on %s: %s", self.device_id, capability.name, exc)
            message = str(exc) if isinstance(exc, ValueError) else _INTERNAL_FAILURE
            return self._refuse(invocation, ToolStatus.FAILED, message)
        self._last_run[capability.name] = now
        return ToolResult(
            ts=self._clock.now(),
            invocation_id=invocation.invocation_id,
            tool=invocation.tool,
            status=ToolStatus.OK,
            message=outcome.message,
            detail=outcome.detail,
            provider=self.device_id,
            simulated=self._manifest.simulated,
        )

    def _refuse(self, invocation: ToolInvocation, status: ToolStatus, message: str) -> ToolResult:
        LOGGER.info("%s refused %s (%s): %s", self.device_id, invocation.tool, status.value, message)
        return ToolResult(ts=self._clock.now(), invocation_id=invocation.invocation_id, tool=invocation.tool, status=status, message=message)
