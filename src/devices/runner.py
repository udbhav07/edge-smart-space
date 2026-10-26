"""Running a device service as a process: ``python -m src.devices.<name>``.

Each device's ``__main__`` hands :func:`run` a function that builds its
:class:`DeviceParts` from the configuration; everything a service process
needs besides -- the broker connection, the last will, re-announcing after a
reconnect, saying goodbye on the way out -- is here, once.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.devices import DeviceManifest
from src.common.mqtt_client import Blackboard, build_transport
from src.devices.service import DeviceService, Handler, StateReader, last_will

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

#: How often the process looks at its device and its connection, in s.
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


def run(
    argv: list[str] | None,
    description: str,
    client_id: str,
    build: Callable[[Config, Clock], DeviceParts],
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
    parts = build(config, clock)
    holder: list = []
    transport = build_transport(config.mqtt, client_id, holder, will=last_will(parts.manifest.device_id, clock))
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    service = DeviceService(parts.manifest, parts.handlers, parts.state, clock, blackboard)
    blackboard.start()
    LOGGER.info("%s (%s) is starting", parts.manifest.name, parts.manifest.device_id)
    announced = False
    try:
        while True:
            # The broker publishes the last will if the link drops, so the
            # device is announced again every time the link comes back.
            if blackboard.connected and not announced:
                service.start()
                announced = True
                LOGGER.info("%s is on the board", parts.manifest.device_id)
            elif not blackboard.connected:
                announced = False
            if announced and parts.tick is not None:
                parts.tick(service)
            clock.sleep(TICK_S)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        if announced:
            service.stop()
        parts.close()
        blackboard.stop()
    return 0
