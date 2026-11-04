"""Which devices are in the room, for the model to be told about (FR-89).

A 7B model's context cannot grow with every device plugged in. So the first
call -- the extraction -- is shown one short line per device and asked only
*which* one a request is about; the assistant's call is then shown that
device's tools and no others. :class:`KnownDevices` keeps the lines, from the
retained manifests and availability on the board, and recognises a device
named in the words, so a request about the fan reaches the fan even when the
model filed it elsewhere.

Plant devices -- the air conditioner -- are left out of both. A request
about the room's temperature takes the goal path to the validator (FR-45).
"""

from __future__ import annotations

import re
import threading

from src.common import topics
from src.common.devices import DeviceAvailability, DeviceClass, DeviceManifest
from src.common.mqtt_client import Blackboard


class KnownDevices:
    """The operable devices that are online now, by id, name and alias."""

    def __init__(self) -> None:
        self._manifests: dict[str, DeviceManifest] = {}
        self._online: dict[str, bool] = {}
        self._lock = threading.Lock()

    def subscribe(self, blackboard: Blackboard) -> None:
        blackboard.subscribe(topics.DEVICE_MANIFEST, DeviceManifest, self._on_manifest)
        blackboard.subscribe(topics.DEVICE_AVAILABILITY, DeviceAvailability, self._on_availability)

    def _on_manifest(self, _topic: str, manifest: DeviceManifest) -> None:
        with self._lock:
            self._manifests[manifest.device_id] = manifest
            self._online.setdefault(manifest.device_id, True)

    def _on_availability(self, _topic: str, availability: DeviceAvailability) -> None:
        with self._lock:
            self._online[availability.device_id] = availability.online

    def operable(self) -> list[DeviceManifest]:
        """Online devices with something a request can operate, in a stable order."""
        with self._lock:
            return [
                manifest
                for device_id, manifest in sorted(self._manifests.items())
                if self._online.get(device_id, False)
                and any(capability.device_class is not DeviceClass.PLANT for capability in manifest.capabilities)
            ]

    def name_of(self, device_id: str) -> str:
        with self._lock:
            manifest = self._manifests.get(device_id)
        return manifest.name if manifest else device_id

    def lines(self) -> list[str]:
        """One line per operable device, as the extraction is shown them."""
        lines = []
        for manifest in self.operable():
            also = f" (also: {', '.join(manifest.aliases)})" if manifest.aliases else ""
            lines.append(f"- {manifest.device_id}: {manifest.name}{also}")
        return lines

    def named_in(self, text: str) -> str | None:
        """The operable device the words name, by the longest name or alias said.

        Whole words only, so "fan" is not found in "fantastic".
        """
        said = text.lower()
        best: tuple[int, str] | None = None
        for manifest in self.operable():
            for name in (manifest.device_id.replace("_", " "), manifest.name, *manifest.aliases):
                name = name.lower().strip()
                if name and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", said):
                    if best is None or len(name) > best[0]:
                        best = (len(name), manifest.device_id)
        return best[1] if best else None
