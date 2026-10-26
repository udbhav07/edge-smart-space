"""Clear a held system, from a terminal (section 5.6).

    python -m tools.reset --reason "replaced the indoor sensor"
    python -m tools.reset --requester udbhav --reason "door was open"

SAFE_HOLD is the one state the system will not leave on its own, and
DEGRADED_ACTUATOR is the other: on an open-loop IR path there is no
acknowledgement to restore and the mode blocks the cooling that would prove the
air conditioner works, so no observation can ever clear it. A person has to say
they have dealt with it.

**This does not override anything.** The reset retires the active faults so the
detectors start gathering evidence again from scratch. If whatever was wrong is
still wrong, it is detected again within its own window -- a dropout in 15 s, a
stuck sensor in five minutes. The reset re-tests; it cannot conceal.

The reason is recorded on the message and ends up in the audit trail, because a
hold cleared with no record of who cleared it or why is an audit trail with a
hole exactly where the interesting thing happened (FR-46).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.schemas import ModeReset, ModeState
from tools.console import quiet_connection_logs

#: How long to wait for the mode the reset produced.
_OUTCOME_WAIT_S = 5.0
_POLL_S = 0.1

LOGGER = logging.getLogger("reset")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "mode-reset"

#: MQTT delivers from the network thread, so the publish has to reach the
#: broker before the process exits.
_SETTLE_S = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.reset",
        description="Clear SAFE_HOLD or DEGRADED_ACTUATOR by hand (section 5.6).",
        epilog=(
            "This re-tests rather than overrides: the detectors start again "
            "from scratch, so a fault that is still present comes straight back."
        ),
    )
    parser.add_argument(
        "--requester",
        default="operator",
        help="Who is clearing the hold; recorded in the audit trail",
    )
    parser.add_argument(
        "--reason",
        default="",
        help="What was done about it, in your own words",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    quiet_connection_logs()

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    if not arguments.requester:
        LOGGER.error("a reset must say who asked for it")
        return 2

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    blackboard.start()

    reset = ModeReset(
        ts=clock.now(),
        requester=arguments.requester,
        reason=arguments.reason,
    )
    modes: list[ModeState] = []
    blackboard.subscribe(topics.SYSTEM_MODE, ModeState, lambda _topic, state: modes.append(state))
    blackboard.publish(topics.SYSTEM_RESET, reset)
    clock.sleep(_SETTLE_S)
    deadline = clock.monotonic() + _OUTCOME_WAIT_S
    while clock.monotonic() < deadline and not any(state.ts >= reset.ts for state in modes):
        clock.sleep(_POLL_S)
    blackboard.stop()

    print(describe_outcome(reset, modes))
    return 0


def describe_outcome(reset: ModeReset, modes: list[ModeState]) -> str:
    """What the reset did, and what it did not promise (U-18).

    A reset re-tests rather than overrides: every fault is retired and a
    fault still present comes back within its detector's window, which for
    the air conditioner is about ten minutes. Showing NORMAL with no word
    of that made a unit still broken look fixed.

    :param modes: every mode heard, oldest first; the retained one arrives
        first, so a reset that changed nothing still has a mode to report.
    """
    if not modes:
        return "reset sent, but no mode was heard: is the detectors service running?"
    state = modes[-1]
    changed = state.ts >= reset.ts
    line = f"mode {'is now' if changed else 'stays'} {state.mode.value}: {state.reason}"
    if state.active_fault_ids:
        line += f"\nstill active: {', '.join(state.active_fault_ids)}"
    return (
        f"{line}\nfaults are re-tested from scratch: one still present comes back "
        f"within its detector's window (about ten minutes for the air conditioner)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
