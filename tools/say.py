"""Say something to the room from a terminal, and hear what it says back.

    python -m tools.say "it is too warm in here"
    python -m tools.say "put the design review in on Thursday at three"
    python -m tools.say "what have I got on Thursday?"
    python -m tools.say "book me a flight to Delhi on Friday"
    python -m tools.say --wait 60 "..."          # a slow model on a laptop CPU

Publishes the words as an ``Utterance`` on ``space/context/utterance`` --
exactly what the speech pipeline publishes after transcription -- and prints
what came back: the reply on ``space/context/preference`` and every tool
result on ``space/assist/result``. Nothing here reaches a provider or the
plant; it is a microphone made of a keyboard, and the reasoning process
cannot tell the difference, which is the point (FR-53).

A flight comes back as a question. Answer it with ``python -m tools.confirm``.
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.schemas import PreferenceHint, Utterance, UtteranceSource
from src.common.tools import ToolResult

LOGGER = logging.getLogger("say")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "say"

#: Default wait for an answer. A 7B model on the Orin answers in a few
#: seconds; on a laptop CPU it can take tens.
DEFAULT_WAIT_S = 45.0


class Conversation:
    """One utterance out, and whatever comes back for it."""

    def __init__(self, clock: Clock, blackboard: Blackboard) -> None:
        self._clock = clock
        self._blackboard = blackboard
        self._asked_ts: float | None = None
        self.reply: PreferenceHint | None = None
        self.results: list[ToolResult] = []
        self.answered = threading.Event()

    def subscribe(self) -> None:
        self._blackboard.subscribe(
            topics.CONTEXT_PREFERENCE, PreferenceHint, self._on_reply
        )
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)

    def say(self, words: str) -> Utterance:
        utterance = Utterance(
            ts=self._clock.now(), text=words, source=UtteranceSource.OPERATOR
        )
        self._asked_ts = utterance.ts
        self._blackboard.publish(topics.CONTEXT_UTTERANCE, utterance)
        return utterance

    def _on_reply(self, _topic: str, hint: PreferenceHint) -> None:
        if self._asked_ts is None or hint.ts < self._asked_ts:
            return
        self.reply = hint
        self.answered.set()

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        if self._asked_ts is not None and result.ts >= self._asked_ts:
            self.results.append(result)

    def transcript(self) -> str:
        """What happened, for a person reading a terminal."""
        lines = []
        for result in self.results:
            mark = " [MOCK]" if result.simulated else ""
            lines.append(f"  {result.tool}: {result.status.value}{mark} -- {result.message}")
        if self.reply is None:
            lines.append(
                "no answer. Is the reasoning process running, and the inference "
                "server? (python start.py --check)"
            )
        else:
            lines.append(f"room: {self.reply.spoken_reply or '(nothing to say)'}")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Say something to the room.")
    parser.add_argument("words", nargs="+", help="What to say")
    parser.add_argument("--wait", type=float, default=DEFAULT_WAIT_S, help="Seconds to wait")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
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
    conversation = Conversation(clock, blackboard)
    conversation.subscribe()
    blackboard.start()
    try:
        # Subscriptions are issued on connect; give the link a moment first
        # so the reply is not published before anyone is listening for it.
        clock.sleep(0.5)
        conversation.say(" ".join(arguments.words))
        clock.wait_for(conversation.answered, arguments.wait)
        # Tool results can trail the reply by a moment on a real broker.
        clock.sleep(0.2)
    finally:
        blackboard.stop()
    print(conversation.transcript())
    return 0 if conversation.reply is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
