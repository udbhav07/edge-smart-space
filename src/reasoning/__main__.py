"""Run the reasoning layer as a service.

``python -m src.reasoning``

The Environmental Supervisor on its cadence and on events, Personal Context on
every utterance, Fault Diagnosis on every confirmed fault -- all against the
one local inference server (section 5.7.5), all recorded on
``space/audit/reasoning``.

The server not answering is not a reason to exit. Each call records
UNAVAILABLE and the process keeps listening, so the reasoning layer comes back
by itself when the server does, and the regulatory loop never noticed it was
gone (FR-47).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.reasoning.service import ReasoningService, build_service

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "reasoning"

#: How often the loop looks for work. Short against the supervisor's cadence
#: and long against a callback: an utterance waits at most this long.
LOOP_PERIOD_S = 0.5


def run(
    service: ReasoningService,
    clock: Clock,
    period_s: float = LOOP_PERIOD_S,
    ticks: int | None = None,
) -> int:
    """Tick until interrupted, or for a fixed number of iterations."""
    completed = 0
    while ticks is None or completed < ticks:
        service.tick()
        clock.sleep(period_s)
        completed += 1
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reasoning layer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--ticks", type=int, default=None, help="Stop after N loops; default forever."
    )
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

    service = build_service(config, clock, blackboard)
    service.subscribe()
    blackboard.start()
    LOGGER.info(
        "reasoning against %s (%s); every call is recorded on %s",
        config.reasoning.base_url,
        config.reasoning.model,
        topics.AUDIT_REASONING.pattern,
    )
    try:
        run(service, clock, ticks=arguments.ticks)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
