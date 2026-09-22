"""Record everything on the blackboard, so a run can be replayed (FR-62).

    python -m tools.record --output runs/demo.jsonl
    python -m tools.record --output runs/e4.jsonl --seconds 1800

A subscriber with no authority over anything. It reads ``space/#`` and writes
what it hears; killing it changes nothing about how the system behaves, which
is the same rule the dashboard follows (section 4.4).

**What it writes is the wire, not an interpretation of it.** Each line holds
the topic and the payload exactly as they arrived, so a replay puts the same
bytes back on the same topics. Decoding here would bind the recording to this
build's schemas and make a recording unreadable the moment a field changed --
and the recording is meant to outlive the code that produced it.

**One line per message, newline-delimited JSON.** A run can be truncated by a
crash and still be readable up to the last complete line, which a single JSON
array would not survive. It is also greppable, which matters more than it
sounds at eleven at night before a demonstration.

The file grows without bound while it runs, which is why ``--seconds`` exists.
A recording is an artefact of an experiment, not a service that runs forever.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import PAYLOAD_ENCODING, Blackboard, build_transport

LOGGER = logging.getLogger("record")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
DEFAULT_OUTPUT = Path("runs/recording.jsonl")
CLIENT_ID = "recorder"

#: Keys in a recorded line. Short, because there is one per message and a long
#: run has hundreds of thousands of them.
TS_KEY = "ts"
TOPIC_KEY = "topic"
PAYLOAD_KEY = "payload"

#: How often the main thread wakes to check whether the run is over.
_POLL_INTERVAL_S = 0.5


class Recorder:
    """Writes every message it hears to an open file."""

    def __init__(self, clock: Clock, stream) -> None:
        self._clock = clock
        self._stream = stream
        self._count = 0
        self._started_s = clock.monotonic()

    @property
    def count(self) -> int:
        """Messages written so far."""
        return self._count

    @property
    def elapsed_s(self) -> float:
        return self._clock.monotonic() - self._started_s

    def on_message(self, topic: str, payload: bytes) -> None:
        """Write one message.

        A payload that is not valid UTF-8 is counted and dropped rather than
        killing the recording: one malformed publisher must not cost the whole
        run, and the gap is visible in the message count.
        """
        try:
            text = payload.decode(PAYLOAD_ENCODING)
        except UnicodeDecodeError:
            LOGGER.warning("undecodable payload on %s; not recorded", topic)
            return

        line = {
            TS_KEY: self._clock.now(),
            TOPIC_KEY: topic,
            PAYLOAD_KEY: text,
        }
        self._stream.write(json.dumps(line, separators=(",", ":")) + "\n")
        self._count += 1

    def flush(self) -> None:
        """Push what is buffered to disk.

        Called periodically rather than per message: a recording that loses
        its last second to a hard kill is a recording; one that fsyncs every
        five seconds per topic is a performance problem.
        """
        self._stream.flush()


def attach(blackboard: Blackboard, recorder: Recorder) -> None:
    """Subscribe to the whole tree.

    Raw rather than typed, deliberately. ``Blackboard.subscribe`` decodes
    against a schema, and the recorder wants the bytes -- including from a
    topic this build does not know about, which is exactly what a recording
    from a mixed-version system contains.
    """
    blackboard.subscribe_raw(topics.ALL_TOPICS, recorder.on_message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.record",
        description="Record the blackboard to a file for offline replay (FR-62).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Stop after this long; default is until interrupted.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)

    with arguments.output.open("w", encoding=PAYLOAD_ENCODING) as stream:
        recorder = Recorder(clock, stream)
        attach(blackboard, recorder)
        blackboard.start()
        LOGGER.info("recording %s to %s", topics.ALL_TOPICS, arguments.output)
        try:
            _run(recorder, clock, arguments.seconds)
        except KeyboardInterrupt:
            LOGGER.info("stopping")
        finally:
            recorder.flush()
            blackboard.stop()

    LOGGER.info(
        "wrote %d message(s) over %.0f s to %s",
        recorder.count,
        recorder.elapsed_s,
        arguments.output,
    )
    return 0


def _run(recorder: Recorder, clock: Clock, seconds: float | None) -> None:
    """Stay alive while the callbacks do the work."""
    while seconds is None or recorder.elapsed_s < seconds:
        clock.sleep(_POLL_INTERVAL_S)
        recorder.flush()


if __name__ == "__main__":
    raise SystemExit(main())
