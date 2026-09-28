"""Record everything on the blackboard, so a run can be replayed (FR-62).

    python -m tools.record runs/demo.jsonl
    python -m tools.record runs/demo.jsonl --seconds 600

One JSON object per line: seconds since recording began, topic, and the
payload exactly as it crossed the broker. Nothing is decoded or filtered --
a recording that only kept what someone thought mattered could not replay
what nobody thought to keep. ``python -m sim.replay`` publishes it back.

Like the dashboard, the recorder has no authority: it only listens, and
killing it costs nothing but the recording (section 3 of the coding rules).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path
from typing import TextIO

from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import PAYLOAD_ENCODING, build_transport
from src.common.topics import TOPIC_ROOT

LOGGER = logging.getLogger("tools.record")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "recorder"
_EVERYTHING_QOS = 1


class Recorder:
    """Writes every message it hears, in order, with its time offset."""

    def __init__(self, clock: Clock, sink: TextIO) -> None:
        self._clock = clock
        self._sink = sink
        self._started = clock.monotonic()
        self._lock = threading.Lock()
        self._transport = None
        self.count = 0

    def attach(self, transport) -> None:
        self._transport = transport

    # The transport calls these, exactly as it calls a Blackboard.
    def on_connected(self) -> None:
        if self._transport is not None:
            self._transport.subscribe(f"{TOPIC_ROOT}/#", _EVERYTHING_QOS)

    def on_disconnected(self) -> None:
        LOGGER.warning("recorder lost the broker; the gap will show in the recording")

    def dispatch(self, topic: str, payload: bytes) -> None:
        line = json.dumps(
            {
                "t": round(self._clock.monotonic() - self._started, 3),
                "topic": topic,
                "payload": payload.decode(PAYLOAD_ENCODING, errors="replace"),
            }
        )
        with self._lock:
            self._sink.write(line + "\n")
            self.count += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.record", description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="Where to write the recording (JSON lines)")
    parser.add_argument("--seconds", type=float, default=None, help="Stop after this long")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    clock = RealClock()
    arguments.path.parent.mkdir(parents=True, exist_ok=True)
    with arguments.path.open("w", encoding="utf-8", buffering=1) as sink:
        recorder = Recorder(clock, sink)
        transport = build_transport(config.mqtt, CLIENT_ID, [recorder])
        recorder.attach(transport)
        transport.connect(config.mqtt.host, config.mqtt.port, int(config.mqtt.keepalive_s))
        transport.loop_start()
        LOGGER.info("recording %s/# to %s", TOPIC_ROOT, arguments.path)
        try:
            started = clock.monotonic()
            while arguments.seconds is None or clock.monotonic() - started < arguments.seconds:
                clock.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            transport.loop_stop()
            transport.disconnect()
        LOGGER.info("recorded %d messages", recorder.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
