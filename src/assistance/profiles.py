"""Keeps each person's context on the blackboard (FR-52, FR-60).

The store is files; the assistant reads the blackboard. This is what joins
them, in the process that owns the files:

* **Every exchange becomes short-term context.** The assistant already
  publishes what it said back, with whose voice it answered, on
  ``space/context/reply``; that is recorded against the person, so no second
  path writes conversation and nothing in the reasoning layer touches a file.
* **Every person's context is published retained** on
  ``space/context/profile/{profile_id}``: on start, after a ``remember`` or
  ``forget`` the executor ran, after each recorded exchange, and every
  ``profiles.publish_interval_s`` -- so a profile edited or deleted by hand
  reaches the blackboard without a restart, and a deleted one is withdrawn.

A failure to record or publish is logged and costs that one update; it never
costs the executor its tools.
"""

from __future__ import annotations

import logging

from src.assistance.memory import ProfileError, ProfileStore
from src.assistance.providers.profile_memory import PROFILE_DETAIL_KEY
from src.common import topics
from src.common.clock import Clock
from src.common.config import ProfilesConfig
from src.common.mqtt_client import Blackboard
from src.common.schemas import GUEST_PROFILE_ID, AssistantReply
from src.common.tools import PROFILE_TOOLS, ToolResult, ToolStatus

LOGGER = logging.getLogger(__name__)


class ProfileKeeper:
    """Records conversation and publishes what is known about each person."""

    def __init__(
        self, config: ProfilesConfig, clock: Clock, blackboard: Blackboard, store: ProfileStore
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._store = store
        self._published: set[str] = set()
        self._last_sweep_s: float | None = None

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.CONTEXT_REPLY, AssistantReply, self._on_reply)
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)

    # --- inputs -------------------------------------------------------

    def _on_reply(self, _topic: str, reply: AssistantReply) -> None:
        """One exchange, kept as the speaker's short-term context."""
        profile_id = reply.profile_id or GUEST_PROFILE_ID
        try:
            self._store.record_turn(profile_id, reply.transcript, reply.reply)
        except (ProfileError, OSError) as exc:
            LOGGER.warning("could not keep the exchange for %s: %s", profile_id, exc)
            return
        self.publish(profile_id)

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        """A remember or forget the executor ran: republish that person."""
        if result.tool not in PROFILE_TOOLS or result.status is not ToolStatus.OK:
            return
        profile_id = result.detail.get(PROFILE_DETAIL_KEY)
        if isinstance(profile_id, str):
            self.publish(profile_id)

    # --- publishing ---------------------------------------------------

    def publish(self, profile_id: str) -> bool:
        """Publish one person's context retained. Returns whether it was."""
        try:
            context = self._store.context(profile_id)
        except (ProfileError, OSError) as exc:
            LOGGER.warning("cannot publish the context of %s: %s", profile_id, exc)
            return False
        self._blackboard.publish(topics.CONTEXT_PROFILE, context, profile_id=profile_id)
        self._published.add(profile_id)
        return True

    def publish_all(self) -> None:
        """Every profile and the guest; withdraw any deleted since last time."""
        current = {*self._store.profile_ids(), GUEST_PROFILE_ID}
        for profile_id in sorted(current):
            self.publish(profile_id)
        for gone in sorted(self._published - current):
            self._blackboard.clear_retained(topics.CONTEXT_PROFILE, profile_id=gone)
            self._published.discard(gone)
            LOGGER.info("withdrew the context of %s: the profile is gone", gone)

    def tick(self) -> None:
        """Republish everything on the configured interval (hand edits, expiry)."""
        now_s = self._clock.monotonic()
        if self._last_sweep_s is not None and now_s - self._last_sweep_s < self._config.publish_interval_s:
            return
        self._last_sweep_s = now_s
        self.publish_all()
