"""Break the system on purpose, from a terminal (FR-31).

    python -m tools.inject temp_01 stuck 27.0    # freeze the indoor sensor
    python -m tools.inject temp_01 dropout       # make it go quiet
    python -m tools.inject temp_01 range 999.0   # report something impossible
    python -m tools.inject temp_01 drift 0.01    # 0.01 degrees per second
    python -m tools.inject ac dead               # the unit stops cooling
    python -m tools.inject temp_01 clear         # stop injecting
    python -m tools.inject --list                # what can be injected

Every fault class D1 to D3 detects is triggerable from here without editing
code or restarting anything, which is what FR-31 asks for and what an examiner
asking an unscripted question needs.

It publishes one retained message and exits. It does not wait for a fault to
be raised, because it has no way to know whether one should be: whether the
detector notices is the question being asked, and a tool that answered it
would be marking its own homework. Watch the answer with

    python -m tools.blackboard_view

or ``mosquitto_sub -t 'space/fault/#' -v``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import RealClock
from src.common.config import ConfigError, load_config
from src.common.injection import FAULTS_REQUIRING_MAGNITUDE, InjectedFault
from src.common.mqtt_client import Blackboard, build_transport
from src.faults.injector import FaultInjector

LOGGER = logging.getLogger("inject")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "fault-injector"

#: Short names, because an examiner types these at a prompt under pressure.
#: ``clear`` is spelled out rather than ``none``: what the operator is doing is
#: ending a fault, not injecting an absence.
FAULT_NAMES: dict[str, InjectedFault] = {
    "stuck": InjectedFault.STUCK_AT,
    "dropout": InjectedFault.DROPOUT,
    "range": InjectedFault.OUT_OF_RANGE,
    "drift": InjectedFault.DRIFT,
    "dead": InjectedFault.STUCK_OFF,
    "clear": InjectedFault.NONE,
}

#: What each fault's magnitude means, for the help text and the failure
#: message. A number with no unit is how the wrong one gets typed.
MAGNITUDE_UNITS: dict[InjectedFault, str] = {
    InjectedFault.STUCK_AT: "the value to freeze at, in the sensor's unit",
    InjectedFault.OUT_OF_RANGE: "the impossible value to report",
    InjectedFault.DRIFT: "drift rate, in units per second",
}

#: Subjects that are not sensors. The air conditioner is injectable too, so
#: FR-24 can be triggered the same way FR-20 to FR-23 are.
_ACTUATOR_SUBJECTS = (topics.AIR_CONDITIONER_ID,)

#: MQTT delivers from the network thread, so the publish has to reach the
#: broker before the process exits. One second is far longer than a local
#: publish needs and short enough not to be noticed at a demonstration.
_SETTLE_S = 1.0


def _describe_faults() -> str:
    lines = ["Injectable faults:"]
    for name, kind in FAULT_NAMES.items():
        unit = MAGNITUDE_UNITS.get(kind, "takes no magnitude")
        lines.append(f"  {name:<8} {kind.value:<13} {unit}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.inject",
        description="Inject a sensor fault into the running system (FR-31).",
        epilog=_describe_faults(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "subject", nargs="?", help="Sensor id, for example temp_01"
    )
    parser.add_argument(
        "fault",
        nargs="?",
        choices=sorted(FAULT_NAMES),
        help="What to inject",
    )
    parser.add_argument(
        "magnitude",
        nargs="?",
        type=float,
        default=None,
        help="How hard, where the fault takes one",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--requester",
        default="operator",
        help="Recorded on the command for the audit trail",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the injectable faults and the configured sensors, then exit",
    )
    return parser


def _list_targets(config) -> str:
    sensors = "\n".join(
        f"  {sensor.sensor_id:<12} {sensor.unit.value:<5} {sensor.description}"
        for sensor in config.sensors.adapters
    )
    actuators = "\n".join(
        f"  {name:<12} {'':<5} Air conditioner (takes 'dead')"
        for name in _ACTUATOR_SUBJECTS
    )
    return (
        f"{_describe_faults()}\n\nConfigured sensors:\n{sensors}"
        f"\n\nActuators:\n{actuators}"
    )


def _validate(arguments) -> str:
    """Check the request before opening a connection.

    :returns: an error message, or an empty string when the request is usable.
    """
    if not arguments.subject or not arguments.fault:
        return "a subject and a fault are required; try --list"

    kind = FAULT_NAMES[arguments.fault]
    if kind in FAULTS_REQUIRING_MAGNITUDE and arguments.magnitude is None:
        return (
            f"{arguments.fault} needs a magnitude: "
            f"{MAGNITUDE_UNITS[kind]}"
        )
    if kind not in FAULTS_REQUIRING_MAGNITUDE and arguments.magnitude is not None:
        return (
            f"{arguments.fault} takes no magnitude; {arguments.magnitude!r} "
            f"would be ignored, which looks exactly like a fault that did not "
            f"take effect"
        )
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    if arguments.list:
        # What the operator asked to see, not a diagnostic about the run.
        print(_list_targets(config))
        return 0

    problem = _validate(arguments)
    if problem:
        LOGGER.error("%s", problem)
        return 2

    known = {sensor.sensor_id for sensor in config.sensors.adapters}
    known.update(_ACTUATOR_SUBJECTS)
    if arguments.subject not in known:
        # A warning rather than a refusal: the configured list is what this
        # machine expects, and a subject published by something else is a
        # legitimate thing to aim at.
        LOGGER.warning(
            "%s is not in this configuration's sensors: %s",
            arguments.subject,
            ", ".join(sorted(known)),
        )

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    blackboard.start()

    injector = FaultInjector(blackboard, clock, requester=arguments.requester)
    try:
        command = injector.inject(
            arguments.subject,
            FAULT_NAMES[arguments.fault],
            arguments.magnitude,
        )
    except ValueError as exc:
        LOGGER.error("%s", exc)
        blackboard.stop()
        return 2

    clock.sleep(_SETTLE_S)
    blackboard.stop()
    print(
        f"asked {command.subject} for {command.kind.value}; "
        f"watch space/fault/# for what the detectors make of it"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
