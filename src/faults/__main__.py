"""Run the detector bank as a service.

``python -m src.faults``

Subscribes to every sensor topic, runs D1 to D3 at regulatory cadence, and
publishes faults and sensor health (FR-20 to FR-22, FR-61).

Its own process, for the reason every component has one: restart independence
is a requirement rather than a convenience (section 9.3). Killing this process
stops detection and leaves the regulatory loop running on whatever it last had
-- which is worse than detecting, and is exactly why it is not permitted to
take anything else down with it.

Unlike the estimator, this service is not purely callback-driven. Two of its
three detectors answer questions about *absence*: a dropout is the fact that no
message arrived, and an unfilled variance window is the fact that not enough
did. Neither can be noticed by a message-arrival callback, so the loop ticks on
the clock (FR-20).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.faults.service import DetectorBankService, build_service

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "detector-bank"


def run(
    service: DetectorBankService,
    clock: Clock,
    period_s: float,
    ticks: int | None = None,
) -> None:
    """Tick the bank until interrupted, or for a fixed number of ticks.

    ``ticks`` exists so a test drives real behaviour rather than a double of
    it, and so an accelerated experiment run can bound itself.
    """
    completed = 0
    while ticks is None or completed < ticks:
        service.tick()
        clock.sleep(period_s)
        completed += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fault detector bank.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Stop after N evaluation ticks; default is forever.",
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

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)

    service = build_service(config, clock, blackboard)
    service.subscribe()
    blackboard.start()
    LOGGER.info(
        "watching %d sensor(s); faults appear under %s",
        len(service.watched_subjects),
        topics.FAULT.wildcard(),
    )
    try:
        run(service, clock, config.loop.regulatory_period_s, arguments.ticks)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
