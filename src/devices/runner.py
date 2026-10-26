"""Running device services as a process: ``python -m src.devices.<name>``.

Each device's ``__main__`` hands :func:`run` a function that builds its
:class:`DeviceParts` from the configuration -- one device, or several, as the
MCP bridge has one per server. Everything a service process needs besides --
the broker connection, the last will, re-announcing after a reconnect,
saying goodbye on the way out -- is here, once.

Each device gets a broker connection of its own, because a connection
carries one last will: two devices sharing one would both be reported alive
by it, or both dead.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.devices import DeviceManifest
from src.common.mqtt_client import Blackboard, build_transport
from src.devices.service import DeviceService, Handler, StateReader, last_will

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

#: How often the process looks at its devices and their connections, in s.
TICK_S = 1.0


@dataclass
class DeviceParts:
    """What a device is made of; the runner supplies the rest."""

    manifest: DeviceManifest
    handlers: Mapping[str, Handler]
    state: StateReader
    #: Called every tick, for a device whose state changes on its own (a track ending).
    tick: Callable[[DeviceService], None] | None = None
    #: Called once on the way out, to release what the device holds.
    close: Callable[[], None] = field(default=lambda: None)


@dataclass
class _Running:
    parts: DeviceParts
    blackboard: Blackboard
    service: DeviceService
    announced: bool = False


def run(
    argv: list[str] | None,
    description: str,
    client_id: str,
    build: Callable[[Config, Clock], DeviceParts | Sequence[DeviceParts]],
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    clock = RealClock()
    built = build(config, clock)
    devices = [_connect(config, clock, client_id, parts) for parts in ([built] if isinstance(built, DeviceParts) else built)]
    if not devices:
        LOGGER.info("no devices configured; nothing to run")
        return 0
    try:
        while True:
            for device in devices:
                _tick(device)
            clock.sleep(TICK_S)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        for device in devices:
            if device.announced:
                device.service.stop()
            device.parts.close()
            device.blackboard.stop()
    return 0


def _connect(config: Config, clock: Clock, client_id: str, parts: DeviceParts) -> _Running:
    device_id = parts.manifest.device_id
    holder: list = []
    transport = build_transport(config.mqtt, f"{client_id}-{device_id}", holder, will=last_will(device_id, clock))
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    service = DeviceService(parts.manifest, parts.handlers, parts.state, clock, blackboard)
    blackboard.start()
    LOGGER.info("%s (%s) is starting", parts.manifest.name, device_id)
    return _Running(parts, blackboard, service)


def _tick(device: _Running) -> None:
    # The broker publishes the last will if the link drops, so the device is
    # announced again every time the link comes back.
    if device.blackboard.connected and not device.announced:
        device.service.start()
        device.announced = True
        LOGGER.info("%s is on the board", device.service.device_id)
    elif not device.blackboard.connected:
        device.announced = False
    if device.announced and device.parts.tick is not None:
        device.parts.tick(device.service)
