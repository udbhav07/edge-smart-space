"""The conversation so far, per speaker, as the blackboard carries it (FR-52).

Whatever classifies an utterance -- the speech pipeline, ``tools.say`` --
needs the last few exchanges with that speaker, so "move it to 4" is read
against the meeting just booked rather than as a temperature (U-01). The
memory keeper publishes each speaker's context retained on
``space/context/profile/{profile_id}``; this keeps the latest of each.
"""

from __future__ import annotations

from src.common import topics
from src.common.schemas import GUEST_PROFILE_ID, ContextTurn, ProfileContext

#: Speakers held. Bounded; the memory keeper bounds them too.
_SPEAKERS_KEPT = 32


class Conversations:
    """The latest short-term context of every speaker heard of."""

    def __init__(self) -> None:
        self._recent: dict[str, tuple[ContextTurn, ...]] = {}

    def subscribe(self, blackboard) -> None:
        blackboard.subscribe(topics.CONTEXT_PROFILE, ProfileContext, self._on_context)

    def _on_context(self, _topic: str, context: ProfileContext) -> None:
        if context.profile_id not in self._recent and len(self._recent) >= _SPEAKERS_KEPT:
            self._recent.pop(next(iter(self._recent)))
        self._recent[context.profile_id] = context.recent

    def recent(self, profile_id: str) -> tuple[ContextTurn, ...]:
        """The conversation with a speaker so far; a guest's when unrecognised."""
        return self._recent.get(profile_id or GUEST_PROFILE_ID, ())
