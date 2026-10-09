"""``remember`` and ``forget``: a person's long-term memory, as tools (FR-52, FR-70).

The reasoning layer keeps a fact the same way it puts an entry in the
calendar: by invoking a declared ``write`` tool through the executor, never
by holding the store. Whose memory it goes into is the ``profile_id`` the
requester filled in from the recognised speaker -- the model is never shown
that parameter -- so nothing a model writes can put a fact into somebody
else's file.

A guest has no long-term memory, and a request to remember something for one
is refused with a sentence that says so.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.assistance.memory import ProfileNotFoundError, ProfileStore
from src.common.schemas import GUEST_PROFILE_ID
from src.common.tools import FORGET, REMEMBER, ArgumentValue, ProviderOutcome

PROVIDER_NAME = "profile_memory"

#: Detail key naming whose memory changed, so whatever publishes profiles
#: can republish that one without being told by anything but the result.
PROFILE_DETAIL_KEY = "profile_id"


class ProfileMemory:
    """Long-term memory in the person's own Markdown file."""

    def __init__(self, store: ProfileStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def simulated(self) -> bool:
        """A real effect: the fact is in the file afterwards."""
        return False

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """Keep or remove a fact.

        :raises ProfileError: for a guest, an unknown person, or a full memory.
        :raises ValueError: for a tool this provider does not serve.
        """
        profile_id = str(arguments["profile_id"])
        if profile_id == GUEST_PROFILE_ID:
            raise ValueError(
                "I don't recognise your voice, so there is nowhere to keep that. "
                "Someone can enrol you first."
            )
        try:
            if tool == REMEMBER.name:
                return self._remember(profile_id, str(arguments["fact"]))
            if tool == FORGET.name:
                return self._forget(profile_id, str(arguments["about"]))
        except ProfileNotFoundError as exc:
            # Said to the occupant, so in their terms (U-19).
            raise ValueError(
                "I don't have a profile for you any more, so I can't keep or "
                "remove anything. Someone can enrol you again."
            ) from exc
        raise ValueError(f"{PROVIDER_NAME} does not serve {tool!r}")

    def _remember(self, profile_id: str, fact: str) -> ProviderOutcome:
        profile, added = self._store.remember(profile_id, fact)
        kept = profile.facts[-1] if added else fact
        message = (
            f"I'll remember that, {profile.display_name}."
            if added
            else f"I already knew that, {profile.display_name}."
        )
        return ProviderOutcome(
            message=message,
            detail={PROFILE_DETAIL_KEY: profile_id, "fact": kept, "facts_kept": len(profile.facts)},
        )

    def _forget(self, profile_id: str, about: str) -> ProviderOutcome:
        profile, removed = self._store.forget(profile_id, about)
        if not removed:
            message = f"I had nothing about {about!r} to forget."
        elif len(removed) == 1:
            message = f"Forgotten: {removed[0]}."
        else:
            message = f"Forgotten {len(removed)} things about {about!r}."
        return ProviderOutcome(
            message=message,
            detail={PROFILE_DETAIL_KEY: profile_id, "removed": len(removed)},
        )
