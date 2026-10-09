"""Talk to the system by typing, as if it had heard you.

    python -m tools.say "make it 22 degrees"
    python -m tools.say "put a meeting with Ravi on Thursday at 3" --yes
    python -m tools.say                      # interactive: type, read, repeat
    python -m tools.say --as ravi "remember that I am vegetarian"

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
import re
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.conversation import Conversations
from src.common.mqtt_client import Blackboard, build_transport
from src.common.schemas import (
    PROFILE_ID_PATTERN,
    AssistantReply,
    FaultDiagnosis,
    PreferenceHint,
    ValidationVerdict,
)
from src.common.tools import ToolInvocation, ToolResult
from src.reasoning.single_shot import PersonalContext, ReasoningUnavailableError
from tools.console import safe_console

LOGGER = logging.getLogger("tools.say")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "voice-stand-in"

#: How long to wait for the system to answer, in seconds. Generous: a service
#: request is two completions on a 7B model plus a tool round trip.
REPLY_TIMEOUT_S = 90.0

_YES = frozenset({"y", "yes", "yeah", "yep", "sure", "go ahead", "confirm", "ok", "okay"})
_NO = frozenset({"n", "no", "nope", "don't", "dont", "cancel", "stop", "no thanks"})

#: How long to wait before saying the system is still thinking. The first
#: answer after startup took 85 s while the model loaded, in silence (U-09).
_STILL_THINKING_S = 10.0


class Conversation:
    """One person talking to the system through the blackboard."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        personal_context: PersonalContext,
        speaker: str = "",
    ) -> None:
        """
        :param speaker: the profile to speak as, standing in for a recognised
            voice (FR-52); empty speaks as a guest.
        """
        self._config = config
        self._speaker = speaker
        self._clock = clock
        self._blackboard = blackboard
        self._personal_context = personal_context
        self._replies: queue.Queue[AssistantReply] = queue.Queue()
        self._notices: queue.Queue[FaultDiagnosis] = queue.Queue()
        self._conversations = Conversations()
        self._results: queue.Queue[ToolResult] = queue.Queue()
        self._invocations: dict[str, ToolInvocation] = {}

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.CONTEXT_REPLY, AssistantReply, self._on_reply)
        self._blackboard.subscribe(topics.DIAGNOSIS, FaultDiagnosis, self._on_notice)
        self._conversations.subscribe(self._blackboard)
        self._blackboard.subscribe(topics.ASSIST_PROPOSED, ToolInvocation, self._on_invocation)
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)
        self._blackboard.subscribe(topics.AUDIT_VALIDATION, ValidationVerdict, self._ignore)

    def _on_reply(self, _topic: str, reply: AssistantReply) -> None:
        self._replies.put(reply)

    def _on_notice(self, _topic: str, diagnosis: FaultDiagnosis) -> None:
        """A fault, explained: what the occupant is told when it happens (FR-28)."""
        self._notices.put(diagnosis)

    def notices(self) -> list[str]:
        """Fault explanations heard since last asked, oldest first."""
        heard = []
        while True:
            try:
                heard.append(self._notices.get_nowait().user_message)
            except queue.Empty:
                return heard

    def _on_invocation(self, _topic: str, invocation: ToolInvocation) -> None:
        self._invocations[invocation.invocation_id] = invocation

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        self._results.put(result)

    def _ignore(self, _topic: str, _message: object) -> None:
        return None

    # --- speaking -----------------------------------------------------

    def hear(self, utterance: str) -> PreferenceHint | None:
        """What the speech pipeline does after Whisper: extract, then publish."""
        hint = self._personal_context.extract(utterance, self._conversations.recent(self._speaker))
        if hint is not None:
            if self._speaker:
                hint = hint.model_copy(update={"profile_id": self._speaker})
            self._blackboard.publish(topics.CONTEXT_PREFERENCE, hint)
        return hint

    def await_reply(
        self, transcript: str, timeout_s: float = REPLY_TIMEOUT_S, waiting=None
    ) -> AssistantReply | None:
        """The reply to this utterance. ``waiting`` is called once if it is slow."""
        started = self._clock.monotonic()
        deadline = started + timeout_s
        told = False
        while self._clock.monotonic() < deadline:
            if waiting is not None and not told and self._clock.monotonic() - started >= _STILL_THINKING_S:
                waiting()
                told = True
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
    reply = conversation.await_reply(
        hint.transcript,
        waiting=lambda: print(
            "  (still thinking -- the first answer after startup can take a minute "
            "while the model loads)"
        ),
    )
    for notice in conversation.notices():
        print(f"  notice: {notice}")
    if reply is None:
        print("  (no reply -- is the reasoning service running?)")
        return
    print(f"  system: {reply.reply}")
    if not reply.awaiting_confirmation:
        return
    if answer_yes is not None:
        agreed = answer_yes
    else:
        answer = ask("  confirm? ").strip().lower()
        agreed = answer in _YES
        if not agreed and answer not in _NO:
            print(f"  (I didn't understand {answer!r}, so I'm taking it as no)")
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
    parser.add_argument(
        "--as",
        dest="speaker",
        default="",
        help="Speak as this enrolled profile, as if the voice were recognised",
    )
    confirmation = parser.add_mutually_exclusive_group()
    confirmation.add_argument("--yes", action="store_true", help="Confirm any booking")
    confirmation.add_argument("--no", action="store_true", help="Decline any booking")
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

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    conversation = Conversation(
        config, clock, blackboard, PersonalContext(config.reasoning, clock), speaker=arguments.speaker
    )
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
