"""Put a recorded run back on the blackboard (FR-62).

    python -m sim.replay runs/demo.jsonl
    python -m sim.replay runs/demo.jsonl --speed 10
    python -m sim.replay runs/demo.jsonl --speed 0     # as fast as it will go

Stands where the simulator stands: it *is* Layer 1 for the duration of a
replay, publishing the same bytes on the same topics in the same order. Every
component above it behaves as it did during the recording, because nothing
above Layer 1 can tell the difference -- which is the same property that makes
the simulator worth having.

Two jobs, and they are different from the demonstration script's. A recorded
run can be re-examined after the fact with a tool that did not exist when it
was recorded: attach the console to a replay and watch the fault again. And it
is the honest fallback for a demonstration, because it is a recording of the
system working rather than a second implementation of it.

**Delivery comes from the topic, not from the recording.** A line holds a
concrete topic and the encoded payload; the quality of service and retention
are looked up from the declared contract. A recording cannot observe whether
a message was retained, and guessing would replay a state topic as transient.

**Timing is reconstructed from the recorded timestamps**, so a replay unfolds
at the pace the run did. ``--speed`` scales that, because nobody wants to
watch half an hour of room again to reach the interesting part.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import PAYLOAD_ENCODING, Blackboard, build_transport

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "replayer"

#: Keys written by tools/record.py. Named here rather than imported: sim may
#: import src.common and nothing else, and a recording is a file format rather
#: than a piece of either module.
TS_KEY = "ts"
TOPIC_KEY = "topic"
PAYLOAD_KEY = "payload"

#: A replay with no delay at all. Useful for feeding an analysis quickly, and
#: what the tests use so they do not sleep.
UNTIMED = 0.0


class RecordingError(ValueError):
    """A recording cannot be read as one."""


@dataclass(frozen=True)
class RecordedMessage:
    """One line of a recording."""

    ts: float
    topic: str
    payload: bytes


def parse_line(line: str, number: int) -> RecordedMessage:
    """Read one recorded line.

    :raises RecordingError: if the line is not a recorded message. A recording
        is an experimental artefact, so a malformed one is worth a clear
        complaint rather than a silent skip: an experiment replayed from a
        half-written file would produce a result nobody could account for.
    """
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RecordingError(f"line {number} is not JSON: {exc}") from exc

    missing = {TS_KEY, TOPIC_KEY, PAYLOAD_KEY} - set(parsed)
    if missing:
        raise RecordingError(
            f"line {number} is missing {sorted(missing)}"
        )
    return RecordedMessage(
        ts=float(parsed[TS_KEY]),
        topic=str(parsed[TOPIC_KEY]),
        payload=str(parsed[PAYLOAD_KEY]).encode(PAYLOAD_ENCODING),
    )


def read(path: Path) -> Iterator[RecordedMessage]:
    """Stream a recording from disk.

    A generator rather than a list: a half-hour run is hundreds of thousands
    of messages, and there is no reason for all of them to be resident at once
    (NFR-05).

    A trailing incomplete line is tolerated, because that is what a recording
    truncated by a hard kill looks like and it is still a usable run.
    """
    with path.open("r", encoding=PAYLOAD_ENCODING) as stream:
        for number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield parse_line(stripped, number)
            except RecordingError as exc:
                LOGGER.warning("%s; stopping here", exc)
                return


class Replayer:
    """Publishes a recording back onto the blackboard."""

    def __init__(
        self,
        blackboard: Blackboard,
        clock: Clock,
        speed: float = 1.0,
    ) -> None:
        if speed < 0.0:
            raise ValueError(f"speed cannot be negative, got {speed!r}")
        self._blackboard = blackboard
        self._clock = clock
        self._speed = speed
        self._published = 0
        self._skipped = 0

    @property
    def published(self) -> int:
        return self._published

    @property
    def skipped(self) -> int:
        """Messages on topics this build does not declare."""
        return self._skipped

    def play(self, messages: Iterator[RecordedMessage]) -> None:
        """Republish every message, pacing to the recorded timestamps."""
        previous_ts: float | None = None
        for message in messages:
            self._wait(previous_ts, message.ts)
            previous_ts = message.ts
            self._publish(message)

    def _wait(self, previous_ts: float | None, current_ts: float) -> None:
        """Sleep the recorded gap, scaled.

        A non-monotonic recording is replayed without waiting rather than
        refused: timestamps come from whatever clock the recorder had, and a
        step backwards is a fact about that clock, not a reason to discard the
        rest of a run.
        """
        if previous_ts is None or self._speed == UNTIMED:
            return
        gap_s = current_ts - previous_ts
        if gap_s <= 0.0:
            return
        self._clock.sleep(gap_s / self._speed)

    def _publish(self, message: RecordedMessage) -> None:
        spec = topics.spec_for(message.topic)
        if spec is None:
            LOGGER.warning(
                "%s is not a topic this build declares; skipped", message.topic
            )
            self._skipped += 1
            return
        self._blackboard.publish_recorded(message.topic, message.payload, spec)
        self._published += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sim.replay",
        description="Replay a recorded run onto the blackboard (FR-62).",
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback rate; 1 is real time, 0 publishes with no delay.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if not arguments.recording.is_file():
        LOGGER.error("no recording at %s", arguments.recording)
        return 2

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
    blackboard.start()

    replayer = Replayer(blackboard, clock, speed=arguments.speed)
    LOGGER.info("replaying %s at %.3gx", arguments.recording, arguments.speed)
    try:
        replayer.play(read(arguments.recording))
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()

    LOGGER.info(
        "published %d message(s), skipped %d",
        replayer.published,
        replayer.skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
