"""Run the real room's Layer 1 as a service.

``python -m src.io``

The hardware counterpart of ``python -m sim.run_sim``: ESPHome nodes in,
the air conditioner out. ``start.py`` runs one or the other according to
``devices.source``, and nothing above Layer 1 can tell which is running.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common.clock import RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.io.devices import DeviceBridge

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "device-bridge"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge the real devices.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    bridge = DeviceBridge(config, clock, blackboard)
    bridge.subscribe()
    blackboard.start()
    LOGGER.info("bridging %s", ", ".join(bridge.sensor_ids))
    try:
        while True:
            bridge.tick()
            clock.sleep(config.loop.sensor_period_s)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
