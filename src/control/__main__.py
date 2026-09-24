"""Run the regulatory loop as a service.

``python -m src.control``

Tracks the validated setpoint with a deadband and a dwell timer, gates every
command through the safety validator, and publishes both the command and the
verdict that produced it (FR-10 to FR-14).

This is the one process in the system that must keep its cadence. It is also
the one that must survive everything else being dead: no goal, no estimate, no
mode, no reasoning layer. Killing any of those degrades a feature; killing this
one stops the room being controlled at all, which is why it holds the last
validated setpoint rather than asking anyone for it (FR-11, FR-47).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.control.goal_manager import GoalManager
from src.control.service import ControlService, build_service
from src.control.tariff import TariffPublisher, TariffSchedule

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "regulatory-controller"


def run(
    service: ControlService,
    clock: Clock,
    period_s: float,
    ticks: int | None = None,
    goals: GoalManager | None = None,
    tariff: TariffPublisher | None = None,
) -> None:
    """Tick the loop until interrupted, or for a fixed number of cycles.

    ``ticks`` exists so a test drives the real loop rather than a double of it,
    and so an accelerated experiment run can bound itself.

    The goal manager is expired on the same tick. A source going silent is
    precisely the case where no message arrives, so noticing it has to happen
    on the clock rather than on a callback -- otherwise a supervisor that
    crashed mid-proposal keeps steering the room.

    The tariff is published from the same tick for the same reason: a band
    changing is an event nothing else will announce (FR-16).
    """
    completed = 0
    while ticks is None or completed < ticks:
        if tariff is not None:
            tariff.tick()
        if goals is not None:
            goals.expire()
        service.tick()
        clock.sleep(period_s)
        completed += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the regulatory loop.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Stop after N control cycles; default is forever.",
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
    # Arbitration runs in this process and publishes proposals the service
    # then gates, so a spoken request crosses the same gate a supervisor's
    # goal does (FR-45).
    goals = GoalManager(config, clock, blackboard)
    goals.subscribe()
    tariff = TariffPublisher(TariffSchedule(config.tariff), clock, blackboard)
    blackboard.start()
    LOGGER.info(
        "controlling to %.1f C every %.0f s; commands appear on %s",
        service.setpoint_c,
        config.loop.regulatory_period_s,
        topics.ACTUATOR_COMMAND.wildcard(),
    )
    try:
        run(
            service,
            clock,
            config.loop.regulatory_period_s,
            arguments.ticks,
            goals=goals,
            tariff=tariff,
        )
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
