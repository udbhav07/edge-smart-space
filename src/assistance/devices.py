"""The devices on the board, as tools the executor can run (FR-87, FR-88).

:class:`DeviceDirectory` watches every device's retained manifest and
availability. While a device is online, each of its capabilities that is not
``PLANT`` is declared in the tool registry as ``<device>_<capability>``, with
a :class:`DeviceProvider` that carries the call to the device over the board
and waits for its answer. When the device goes offline, or announces a
different manifest, its tools are withdrawn and the catalogue republished --
so what the assistant may ask for is always what is actually reachable.

``PLANT`` capabilities never become tools: the thermal loop is reached only
through a preference and the validator (FR-45), whatever a device declares.
"""

from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Callable, Mapping

from src.common import topics
from src.common.clock import Clock
from src.common.devices import DeviceAvailability, DeviceClass, DeviceManifest
from src.common.mqtt_client import Blackboard
from src.common.tools import (
    ArgumentValue,
    ProviderOutcome,
    ToolInvocation,
    ToolRegistry,
    ToolRequester,
    ToolResult,
    ToolSpec,
    ToolStatus,
)

LOGGER = logging.getLogger(__name__)


class _Waiting:
    """One command on its way to a device, and the answer when it comes."""

    def __init__(self) -> None:
        self.answered = threading.Event()
        self.result: ToolResult | None = None


class DeviceProvider:
    """Performs a device's capabilities by asking the device, over the board."""

    def __init__(self, manifest: DeviceManifest, directory: DeviceDirectory) -> None:
        self._manifest = manifest
        self._directory = directory
        # The tool a model sees, back to the capability the device knows.
        self._capabilities = {manifest.tool_name(capability.name): capability.name for capability in manifest.capabilities}

    @property
    def name(self) -> str:
        return self._manifest.device_id

    @property
    def simulated(self) -> bool:
        return self._manifest.simulated

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """
        :raises ValueError: the device refused, failed, or did not answer --
            in a sentence the registry passes on to the occupant.
        """
        result = self._directory.ask(self._manifest, self._capabilities[tool], arguments)
        if result is None:
            raise ValueError(f"The {self._manifest.name.lower()} didn't answer; it may be switched off or disconnected.")
        if result.status is not ToolStatus.OK:
            raise ValueError(result.message or f"The {self._manifest.name.lower()} couldn't do that.")
        return ProviderOutcome(message=result.message or "Done.", detail=result.detail)


class DeviceDirectory:
    """Keeps the registry's device tools matching the devices that are online."""

    def __init__(
        self,
        clock: Clock,
        blackboard: Blackboard,
        registry: ToolRegistry,
        timeout_s: float,
        on_change: Callable[[], None] = lambda: None,
    ) -> None:
        self._clock = clock
        self._blackboard = blackboard
        self._registry = registry
        self._timeout_s = timeout_s
        self._on_change = on_change
        self._manifests: dict[str, DeviceManifest] = {}
        self._online: dict[str, bool] = {}
        self._declared: dict[str, tuple[str, ...]] = {}
        self._waiting: dict[str, _Waiting] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.DEVICE_MANIFEST, DeviceManifest, self._on_manifest)
        self._blackboard.subscribe(topics.DEVICE_AVAILABILITY, DeviceAvailability, self._on_availability)
        self._blackboard.subscribe(topics.DEVICE_RESULT, ToolResult, self._on_result)

    @property
    def online(self) -> tuple[str, ...]:
        """The devices whose tools are declared now."""
        with self._lock:
            return tuple(device_id for device_id, names in self._declared.items() if names)

    def ask(self, manifest: DeviceManifest, capability: str, arguments: Mapping[str, ArgumentValue]) -> ToolResult | None:
        """Send one command and wait for its answer. None if none came in time."""
        invocation_id = f"dev_{manifest.device_id}_{int(self._clock.now())}_{next(self._ids)}"
        waiting = _Waiting()
        with self._lock:
            self._waiting[invocation_id] = waiting
        try:
            self._blackboard.publish(
                topics.DEVICE_COMMAND,
                ToolInvocation(
                    ts=self._clock.now(),
                    invocation_id=invocation_id,
                    tool=capability,
                    arguments=dict(arguments),
                    requester=ToolRequester.ASSISTANCE,
                    expires_ts=self._clock.now() + self._timeout_s,
                ),
                device_id=manifest.device_id,
            )
            waiting.answered.wait(self._timeout_s)
            return waiting.result
        finally:
            with self._lock:
                self._waiting.pop(invocation_id, None)

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        with self._lock:
            waiting = self._waiting.get(result.invocation_id)
        if waiting is not None:
            waiting.result = result
            waiting.answered.set()

    def _on_manifest(self, _topic: str, manifest: DeviceManifest) -> None:
        with self._lock:
            self._manifests[manifest.device_id] = manifest
            # A manifest is published by a device announcing itself, so it is online
            # unless its availability, retained, says otherwise when it arrives.
            self._online.setdefault(manifest.device_id, True)
        self._refresh(manifest.device_id)

    def _on_availability(self, _topic: str, availability: DeviceAvailability) -> None:
        with self._lock:
            self._online[availability.device_id] = availability.online
        self._refresh(availability.device_id)

    def _refresh(self, device_id: str) -> None:
        with self._lock:
            manifest = self._manifests.get(device_id)
            online = self._online.get(device_id, False)
            before = self._declared.get(device_id, ())
        for name in before:
            self._registry.withdraw(name)
        declared: list[str] = []
        if manifest is not None and online:
            provider = DeviceProvider(manifest, self)
            for capability in manifest.capabilities:
                if capability.device_class is DeviceClass.PLANT:
                    continue
                name = manifest.tool_name(capability.name)
                spec = ToolSpec(
                    name=name,
                    purpose=f"{manifest.name}: {capability.purpose}",
                    effect=capability.effect,
                    parameters=capability.parameters,
                    device=device_id,
                )
                try:
                    self._registry.declare(spec)
                except ValueError as exc:
                    LOGGER.warning("%s: %s is not offered: %s", device_id, name, exc)
                    continue
                self._registry.bind(name, provider)
                declared.append(name)
        with self._lock:
            self._declared[device_id] = tuple(declared)
        if tuple(declared) != before:
            LOGGER.info("%s: %s", device_id, f"offering {', '.join(declared)}" if declared else "no tools offered")
        if declared or before:
            # Republished even when the names are the same: a re-announced
            # manifest may have changed what the tools take.
            self._on_change()
