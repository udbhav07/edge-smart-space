"""Words typed in the console, understood exactly as speech is (FR-56, FR-57).

The console publishes what someone typed on ``space/context/utterance`` and
nothing else; it has no extraction of its own and could not command anything
if it wanted to. Here the same Personal Context call the speech pipeline uses
turns the words into a preference hint, which then takes the path every
spoken request takes: the assistant answers it, the validator gates any
setpoint in it, the executor gates any tool in it.

When nothing in the words can be acted on, the speech pipeline publishes
nothing -- there is no one to tell. A console has someone watching, so the
occupant is answered rather than left looking at silence.
"""

from __future__ import annotations

import logging
from collections import deque

from src.common import topics
from src.common.clock import Clock
from src.common.conversation import Conversations
from src.common.mqtt_client import Blackboard
from src.common.schemas import AssistantReply, ConsoleUtterance, Intent, PreferenceHint
from src.reasoning.single_shot import PersonalContext, ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Typed utterances waiting to be understood. Bounded: a console left
#: hammering the broker cannot grow this without limit.
_PENDING_KEPT = 32

NOTHING_TO_DO = (
    "I didn't find anything I can act on in that. You can ask about the room, "
    "change the temperature, or use your calendar."
)
NO_MODEL = (
    "I can't understand requests right now: the language model isn't answering. "
    "The room keeps running on its own."
)


class UtteranceInterpreter:
    """Turns console utterances into preference hints, one at a time."""

    def __init__(
        self,
        clock: Clock,
        blackboard: Blackboard,
        personal_context: PersonalContext,
        conversations: Conversations | None = None,
    ) -> None:
        self._clock = clock
        self._blackboard = blackboard
        self._personal_context = personal_context
        self._conversations = conversations or Conversations()
        self._pending: deque[ConsoleUtterance] = deque(maxlen=_PENDING_KEPT)

    def subscribe(self) -> None:
        self._conversations.subscribe(self._blackboard)
        self._blackboard.subscribe(topics.CONTEXT_UTTERANCE, ConsoleUtterance, self._on_utterance)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def _on_utterance(self, _topic: str, utterance: ConsoleUtterance) -> None:
        self._pending.append(utterance)

    def process_pending(self) -> list[PreferenceHint | AssistantReply]:
        """Understand every waiting utterance; returns what was published."""
        published: list[PreferenceHint | AssistantReply] = []
        while self._pending:
            published.append(self._understand(self._pending.popleft()))
        return published

    def _understand(self, utterance: ConsoleUtterance) -> PreferenceHint | AssistantReply:
        recent = self._conversations.recent(utterance.profile_id)
        try:
            hint = self._personal_context.extract(utterance.text, recent)
        except ReasoningUnavailableError as exc:
            LOGGER.warning("console utterance not understood, model unavailable: %s", exc)
            return self._answer(utterance, NO_MODEL)
        if hint is None:
            return self._answer(utterance, NOTHING_TO_DO)
        hint = hint.model_copy(update={"transcript": utterance.text, "profile_id": utterance.profile_id})
        self._blackboard.publish(topics.CONTEXT_PREFERENCE, hint)
        LOGGER.info("console utterance understood as %s: %r", hint.intent.value, utterance.text)
        return hint

    def _answer(self, utterance: ConsoleUtterance, text: str) -> AssistantReply:
        reply = AssistantReply(
            ts=self._clock.now(),
            transcript=utterance.text,
            intent=Intent.NONE,
            reply=text,
            profile_id=utterance.profile_id,
        )
        self._blackboard.publish(topics.CONTEXT_REPLY, reply)
        return reply
