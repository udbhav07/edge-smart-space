"""Talk to the system by typing, as if it had heard you.

    python -m tools.say "make it 22 degrees"
    python -m tools.say "put a meeting with Ravi on Thursday at 3" --yes
    python -m tools.say                      # interactive: type, read, repeat

A stand-in for a voice, not a change to one. The speech pipeline is
microphone -> wake word -> Whisper -> Personal Context -> a hint on
``space/context/preference``. This skips the first three and does the rest
exactly as ``src/speech/__main__.py`` does: the same :class:`PersonalContext`,
built from the same configuration, publishing the same hint to the same
topic. Everything downstream -- the goal path, the assistant, the executor --
cannot tell a typed sentence from a spoken one, which is what makes this a
fair test of them and leaves the voice pipeline untouched.

It then waits for what the system says back on ``space/context/reply`` and
prints it. When a booking needs confirming it plays the console's part
(FR-54, section 5.7.6): on a yes it republishes the invocation verbatim to
``space/assist/confirmed``. Nothing else can confirm a booking.
"""

from __future__ import annotations

import argparse
import logging
import queue
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.schemas import AssistantReply, PreferenceHint, ValidationVerdict
from src.common.tools import ToolInvocation, ToolResult
from src.reasoning.single_shot import PersonalContext, ReasoningUnavailableError

LOGGER = logging.getLogger("tools.say")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "voice-stand-in"

#: How long to wait for the system to answer, in seconds. Generous: a service
#: request is two completions on a 7B model plus a tool round trip.
REPLY_TIMEOUT_S = 90.0

_YES = frozenset({"y", "yes", "yeah", "yep", "sure", "go ahead", "confirm", "ok", "okay"})


class Conversation:
    """One person talking to the system through the blackboard."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        personal_context: PersonalContext,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._personal_context = personal_context
        self._replies: queue.Queue[AssistantReply] = queue.Queue()
        self._results: queue.Queue[ToolResult] = queue.Queue()
        self._invocations: dict[str, ToolInvocation] = {}

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.CONTEXT_REPLY, AssistantReply, self._on_reply)
        self._blackboard.subscribe(topics.ASSIST_PROPOSED, ToolInvocation, self._on_invocation)
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)
        self._blackboard.subscribe(topics.AUDIT_VALIDATION, ValidationVerdict, self._ignore)

    def _on_reply(self, _topic: str, reply: AssistantReply) -> None:
        self._replies.put(reply)

    def _on_invocation(self, _topic: str, invocation: ToolInvocation) -> None:
        self._invocations[invocation.invocation_id] = invocation

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        self._results.put(result)

    def _ignore(self, _topic: str, _message: object) -> None:
        return None

    # --- speaking -----------------------------------------------------

    def hear(self, utterance: str) -> PreferenceHint | None:
        """What the speech pipeline does after Whisper: extract, then publish."""
        hint = self._personal_context.extract(utterance)
        if hint is not None:
            self._blackboard.publish(topics.CONTEXT_PREFERENCE, hint)
        return hint

    def await_reply(self, transcript: str, timeout_s: float = REPLY_TIMEOUT_S) -> AssistantReply | None:
        deadline = self._clock.monotonic() + timeout_s
        while self._clock.monotonic() < deadline:
            try:
                reply = self._replies.get(timeout=0.2)
            except queue.Empty:
                continue
            if reply.transcript.strip() == transcript.strip():
                return reply
        return None

    def confirm(self, invocation_id: str, timeout_s: float = 30.0) -> ToolResult | None:
        """The console's job: republish the invocation verbatim (FR-74)."""
        invocation = self._invocations.get(invocation_id)
        if invocation is None:
            return None
        self._blackboard.publish(topics.ASSIST_CONFIRMED, invocation)
        deadline = self._clock.monotonic() + timeout_s
        while self._clock.monotonic() < deadline:
            try:
                result = self._results.get(timeout=0.2)
            except queue.Empty:
                continue
            if result.invocation_id == invocation_id and result.status.value != "CONFIRMATION_REQUIRED":
                return result
        return None


def converse(conversation: Conversation, utterance: str, answer_yes: bool | None, ask) -> None:
    """Say one thing, print what comes back, and handle a confirmation."""
    try:
        hint = conversation.hear(utterance)
    except ReasoningUnavailableError as exc:
        print(f"  (the model server is unreachable: {exc})")
        return
    if hint is None:
        print("  (nothing actionable was heard; the speech pipeline would publish nothing)")
        return
    print(f"  heard as: {hint.intent.value}"
          + (f", {hint.target_c:g} C" if hint.target_c is not None else "")
          + (f", {hint.comfort.value}" if hint.comfort.value != "unchanged" else ""))
    reply = conversation.await_reply(hint.transcript)
    if reply is None:
        print("  (no reply -- is the reasoning service running?)")
        return
    print(f"  system: {reply.reply}")
    if not reply.awaiting_confirmation:
        return
    agreed = answer_yes if answer_yes is not None else ask("  confirm? ").strip().lower() in _YES
    if not agreed:
        print("  (not confirmed; nothing was booked)")
        return
    result = conversation.confirm(reply.awaiting_confirmation)
    if result is None:
        print("  (the confirmation got no answer)")
        return
    print(f"  system: {result.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.say",
        description="Talk to the system by typing, as if it had heard you.",
    )
    parser.add_argument("utterance", nargs="*", help="What to say; omit to converse")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    confirmation = parser.add_mutually_exclusive_group()
    confirmation.add_argument("--yes", action="store_true", help="Confirm any booking")
    confirmation.add_argument("--no", action="store_true", help="Decline any booking")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    conversation = Conversation(config, clock, blackboard, PersonalContext(config.reasoning, clock))
    conversation.subscribe()
    blackboard.start()
    answer_yes = True if arguments.yes else False if arguments.no else None
    try:
        if arguments.utterance:
            utterance = " ".join(arguments.utterance)
            print(f"you: {utterance}")
            converse(conversation, utterance, answer_yes, input)
        else:
            while True:
                utterance = input("you: ").strip()
                if utterance.lower() in ("", "quit", "exit", "bye"):
                    break
                converse(conversation, utterance, answer_yes, input)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
