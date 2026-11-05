"""What the room's voice is doing, as the music needs to know it (FR-96).

Two things follow from it. A song asked for aloud waits for Jarvis to say he
is playing it, rather than starting under his reply; and while the room
listens or speaks, the music is turned down, so the microphone hears the
person and the person hears Jarvis.

All of it is read from the board -- the request, the reply and the speech
service's status -- never from the speech service itself.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from src.common.clock import Clock
from src.common.schemas import AssistantReply, PreferenceHint, SpeechActivity, SpeechStatus

#: While the room is doing any of these, the music is turned down.
_DUCKED = frozenset({SpeechActivity.LISTENING, SpeechActivity.TRANSCRIBING, SpeechActivity.UNDERSTANDING, SpeechActivity.SPEAKING})


class RoomVoice:
    """Whether a spoken request awaits its reply, and whether the room is talking."""

    def __init__(self, clock: Clock, request_window_s: float) -> None:
        """
        :param request_window_s: how recent a spoken request must be for a
            song to be taken as its answer, and wait for the reply.
        """
        self._clock = clock
        self._window_s = request_window_s
        self._lock = threading.Lock()
        self._asked_at: float | None = None
        self._replied = False
        self._reply_spoken = False
        self._spoke_since_reply = False
        self._activity = SpeechActivity.WAITING
        #: Told after every change, outside the lock: the music follows at once.
        self.on_change: Callable[[], None] | None = None

    # --- from the board ----------------------------------------------------

    def on_request(self, _topic: str, hint: PreferenceHint) -> None:
        if hint.typed or hint.remote:
            return  # its reply is on a screen, not said in the room
        with self._lock:
            self._asked_at, self._replied, self._spoke_since_reply = self._clock.now(), False, False

    def on_reply(self, _topic: str, reply: AssistantReply) -> None:
        with self._lock:
            if self._asked_at is None or self._replied:
                return
            self._replied, self._reply_spoken = True, reply.spoken and not reply.remote
        self._changed()

    def on_status(self, _topic: str, status: SpeechStatus) -> None:
        with self._lock:
            self._activity = status.activity
            if self._replied and status.activity is SpeechActivity.SPEAKING:
                self._spoke_since_reply = True
        self._changed()

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    # --- for the music -----------------------------------------------------

    def awaiting_reply(self) -> bool:
        """Whether a spoken request was just made and its reply is still to come."""
        with self._lock:
            return self._asked_at is not None and not self._replied and self._clock.now() - self._asked_at <= self._window_s

    def reply_said(self) -> bool:
        """Whether the reply has come and, if it was to be said, has been said."""
        with self._lock:
            if not self._replied:
                return False
            if not self._reply_spoken:
                return True
            return self._spoke_since_reply and self._activity is not SpeechActivity.SPEAKING

    @property
    def ducked(self) -> bool:
        """Whether the room is listening or speaking now."""
        with self._lock:
            return self._activity in _DUCKED
