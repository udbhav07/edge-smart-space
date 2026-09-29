"""Replay a recorded run onto the blackboard (FR-62).

    python -m sim.replay runs/demo.jsonl
    python -m sim.replay runs/demo.jsonl --speed 10
    python -m sim.replay runs/demo.jsonl --only space/sensor/ space/actuator/

Publishes each recorded message on its original topic, in order, at its
recorded offset divided by ``--speed``. Retain and QoS come from the topic
table (section 6.1) rather than from the recording, so a replay reproduces
the blackboard as it was specified, not as one broker happened to deliver it.

Two uses. Offline analysis: point the estimator or the detector bank at a
replayed recording and it sees the run again, which is how an experiment is
repeated without the room. And the viva fallback: if a live demonstration
fails, the recorded one plays on the same topics, to the same dashboard.

Replaying onto a broker a live system is using would feed it the past; run it
against a separate broker, or with ``--only`` restricted to what is wanted.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import PAYLOAD_ENCODING, build_transport, topic_matches

LOGGER = logging.getLogger("sim.replay")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "replay"
_DEFAULT_QOS = 1


@dataclass(frozen=True)
class Recorded:
    t: float
    topic: str
    payload: str


def load(path: Path) -> list[Recorded]:
    """Read a recording, skipping lines that are not one (and saying so)."""
    recorded = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            recorded.append(Recorded(float(entry["t"]), str(entry["topic"]), str(entry["payload"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("line %d is not a recorded message: %s", number, exc)
    return recorded


def delivery(topic: str) -> tuple[int, bool]:
    """QoS and retain for a topic, from the section 6.1 table."""
    for spec in vars(topics).values():
        if isinstance(spec, topics.TopicSpec) and topic_matches(spec.wildcard(), topic):
            return spec.qos.value, spec.retain
    return _DEFAULT_QOS, False


def replay(
    recorded: Iterable[Recorded],
    publish,
    clock: Clock,
    speed: float = 1.0,
    only: tuple[str, ...] = (),
) -> int:
    """Publish a recording in order, keeping its timing scaled by ``speed``.

    :param publish: ``(topic, payload_bytes, qos, retain)``, as a transport.
    :returns: how many messages were published.
    """
    if speed <= 0.0:
        raise ValueError(f"speed must be positive, got {speed!r}")
    started = clock.monotonic()
    count = 0
    for entry in recorded:
        if only and not any(entry.topic.startswith(prefix) for prefix in only):
            continue
        due = started + entry.t / speed
        wait = due - clock.monotonic()
        if wait > 0.0:
            clock.sleep(wait)
        qos, retain = delivery(entry.topic)
        publish(entry.topic, entry.payload.encode(PAYLOAD_ENCODING), qos, retain)
        count += 1
    return count


class _Silent:
    """A dispatch target for a client that only publishes."""

    def dispatch(self, topic: str, payload: bytes) -> None: ...
    def on_connected(self) -> None: ...
    def on_disconnected(self) -> None: ...


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim.replay", description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--speed", type=float, default=1.0, help="10 plays ten times faster")
    parser.add_argument("--only", nargs="*", default=(), help="Replay only topics with these prefixes")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2
    recorded = load(arguments.path)
    clock = RealClock()
    transport = build_transport(config.mqtt, CLIENT_ID, [_Silent()])
    transport.connect(config.mqtt.host, config.mqtt.port, int(config.mqtt.keepalive_s))
    transport.loop_start()
    try:
        count = replay(recorded, transport.publish, clock, arguments.speed, tuple(arguments.only))
        LOGGER.info("replayed %d of %d messages", count, len(recorded))
    except KeyboardInterrupt:
        LOGGER.info("stopped")
    finally:
        transport.loop_stop()
        transport.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
