"""Run Layer 1 against real hardware.

``python -m src.io``

Bridges ESPHome nodes onto the blackboard and forwards actuator commands to
the air conditioner. It is the process that replaces ``sim.run_sim`` once
there are sensors on a wall, and it publishes the same topics with the same
schemas at the same cadence, so nothing above Layer 1 changes and the whole
test suite applies unchanged (section 9.1).

``start.py`` chooses between this and the simulator from ``io.source`` in
configuration, so the phase transition is an edit rather than a deployment.

**What bring-up still has to establish** is every device fact this reads from
configuration: which topic each node publishes on, what shape its payload
takes, and whether the air conditioner path acknowledges anything at all. They
are configuration rather than code precisely so that settling them is a config
edit, and they are listed in the deployment notes rather than assumed here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, Layer1Source, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.io.service import HardwareLayer, build_layer

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "hardware-layer"


def run(
    layer: HardwareLayer,
    clock: Clock,
    period_s: float,
    polls: int | None = None,
) -> None:
    """Poll every sensor on the clock until interrupted.

    On the clock rather than on device messages: a node that has stopped
    publishing produces no callback, and that is exactly when the rest of the
    system most needs to hear silence reported rather than nothing at all.
    """
    completed = 0
    while polls is None or completed < polls:
        layer.poll()
        clock.sleep(period_s)
        completed += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Layer 1 against ESPHome hardware."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--polls",
        type=int,
        default=None,
        help="Stop after N polling cycles; default is forever.",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    if config.io.source is not Layer1Source.ESPHOME:
        # Refused rather than run anyway. Two Layer 1 implementations
        # publishing the same topics at once would look to everything above
        # like one sensor contradicting itself, which is a fault nobody
        # injected and a very confusing hour.
        LOGGER.error(
            "io.source is %r; this process is the hardware bridge. Set it to "
            "%r, or run the simulator instead.",
            config.io.source.value,
            Layer1Source.ESPHOME.value,
        )
        return 2

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)

    layer = build_layer(config, clock, blackboard, device_bus=transport)
    layer.subscribe()
    blackboard.start()
    LOGGER.info(
        "bridging %d device(s) onto %s",
        len(layer.bound_sensors),
        topics.SENSOR_STATE.wildcard(),
    )
    try:
        run(layer, clock, config.loop.sensor_period_s, arguments.polls)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
