"""Run the thermal estimator as a service.

``python -m src.estimation``

Subscribes to the sensor and actuator topics, identifies the four RC
coefficients online from operating data, and publishes the one-step
prediction, the residual and the coefficients (FR-04, FR-05).

Runs as its own process for the same reason every other component does:
restart independence is a requirement, not a convenience (section 9.3).
Killing this process stops adaptation and leaves the regulatory loop holding
its last validated setpoint, which is the behaviour FR-11 asks for.

Nothing here decides anything. The estimator reports what it has identified
and how well supported it is; what to do about a fault or a mode is the
detector bank's and the mode manager's business.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.estimation.persistence import CoefficientStore
from src.estimation.rls import ThermalEstimator
from src.estimation.service import ThermalEstimatorService

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "thermal-estimator"

#: The service is entirely callback-driven, so the main thread only has to
#: stay alive and remain interruptible.
_IDLE_INTERVAL_S = 1.0


def build_service(
    config: Config, clock: Clock, blackboard: Blackboard
) -> ThermalEstimatorService:
    """Assemble the estimator and its store from configuration."""
    return ThermalEstimatorService(
        config=config,
        clock=clock,
        blackboard=blackboard,
        estimator=ThermalEstimator(config.estimator, clock),
        store=CoefficientStore(config.persistence, clock),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the thermal estimator.")
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

    service = build_service(config, clock, blackboard)
    if service.restore():
        LOGGER.info("resumed from the persisted estimate")
    else:
        LOGGER.info("starting from the configured prior")

    service.subscribe()
    blackboard.start()
    LOGGER.info(
        "estimating; watch with: mosquitto_sub -t '%s' -v",
        topics.ESTIMATE_COEFFICIENTS.pattern,
    )
    try:
        while True:
            clock.sleep(_IDLE_INTERVAL_S)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
