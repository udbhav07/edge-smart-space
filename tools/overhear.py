"""Say something in the room without addressing the assistant, by typing.

    python -m tools.overhear "I've got the dentist on Thursday at four" --as ravi
    python -m tools.overhear "we're flying to Goa next Friday"
    python -m tools.overhear --no-wait "I really hate it when it's below 23" --as ravi

A stand-in for the microphone when the room listens continuously (FR-76),
as ``tools.say`` is for an addressed request. The speech pipeline turns
speech without the wake word into an :class:`AmbientUtterance` on
``space/context/ambient``; this publishes the same message, so the reasoning
layer cannot tell a typed line from an overheard one. It then waits for the
reasoning layer's record of the call and prints what came of it: a plan
pencilled into the calendar as tentative (FR-77), a fact offered to keep
(FR-78), or nothing.
"""

from __future__ import annotations

import argparse
import logging
import queue
import re
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.schemas import (
    PROFILE_ID_PATTERN,
    AmbientUtterance,
    ReasoningCaller,
    ReasoningRecord,
)
from src.common.tools import ToolInvocation, ToolRequester, ToolResult
from tools.console import safe_console

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "overhear-stand-in"

#: How long to wait for the reasoning layer to read the line, in seconds. A
#: cold model takes most of a minute to answer its first call (U-09).
RECORD_TIMEOUT_S = 90.0

#: After the record, how long to wait for the executor's results.
RESULT_TIMEOUT_S = 5.0


class Overhearing:
    """Publishes one overheard line and gathers what came of it."""

    def __init__(self, clock: Clock, blackboard: Blackboard, speaker: str = "") -> None:
        self._clock = clock
        self._blackboard = blackboard
        self._speaker = speaker
        self._line = ""
        self._records: queue.Queue[ReasoningRecord] = queue.Queue()
        self._proposed: dict[str, ToolInvocation] = {}
        #: Every result, by invocation: one can arrive before its proposal
        #: has been seen here, so they are matched once the record is in.
        self._results: dict[str, ToolResult] = {}
        self._arrived: queue.Queue[str] = queue.Queue()

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.AUDIT_REASONING, ReasoningRecord, self._on_record)
        self._blackboard.subscribe(topics.ASSIST_PROPOSED, ToolInvocation, self._on_proposed)
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)

    def _on_record(self, _topic: str, record: ReasoningRecord) -> None:
        if record.caller is ReasoningCaller.AMBIENT and self._line and self._line in record.inputs:
            self._records.put(record)

    def _on_proposed(self, _topic: str, invocation: ToolInvocation) -> None:
        if invocation.requester is ToolRequester.AMBIENT and invocation.rationale == self._line:
            self._proposed[invocation.invocation_id] = invocation

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        self._results[result.invocation_id] = result
        self._arrived.put(result.invocation_id)

    def say(self, line: str) -> AmbientUtterance:
        """Publish the line as speech overheard from the speaker, or a guest."""
        self._line = " ".join(line.split())
        heard = AmbientUtterance(ts=self._clock.now(), transcript=self._line, profile_id=self._speaker)
        self._blackboard.publish(topics.CONTEXT_AMBIENT, heard)
        return heard

    def outcome(self, record_timeout_s: float = RECORD_TIMEOUT_S, result_timeout_s: float = RESULT_TIMEOUT_S) -> list[str]:
        """What came of it, as lines to print."""
        try:
            record = self._records.get(timeout=record_timeout_s)
        except queue.Empty:
            return ["Nothing read it: is the reasoning service running?"]
        lines = [f"Read in {record.latency_s:.1f} s: {record.verdict}."]
        waiting = set(self._proposed)
        while waiting - set(self._results):
            try:
                self._arrived.get(timeout=result_timeout_s)
            except queue.Empty:
                lines.append("No answer from the tool executor: is assistance running?")
                break
        lines += [
            f"  {self._results[invocation_id].tool}: {self._results[invocation_id].message}"
            for invocation_id in self._proposed
            if invocation_id in self._results
        ]
        return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.overhear",
        description="Say something in the room without addressing the assistant.",
    )
    parser.add_argument("line", nargs="+", help="What was said")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--as", dest="speaker", default="", help="As if this enrolled voice said it")
    parser.add_argument("--no-wait", action="store_true", help="Publish and exit")
    arguments = parser.parse_args(argv)
    safe_console()
    if arguments.speaker and not re.match(PROFILE_ID_PATTERN, arguments.speaker):
        print(f"error: {arguments.speaker!r} is not a profile id (lower case, like ravi)")
        return 2
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    overhearing = Overhearing(RealClock(), blackboard, arguments.speaker)
    overhearing.subscribe()
    blackboard.start()
    try:
        heard = overhearing.say(" ".join(arguments.line))
        who = arguments.speaker or "a guest"
        print(f"Overheard from {who}: {heard.transcript}")
        if not arguments.no_wait:
            for line in overhearing.outcome():
                print(line)
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
