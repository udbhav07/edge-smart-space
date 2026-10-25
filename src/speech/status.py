"""What the speech pipeline is doing, on the blackboard (FR-60).

Published on every change and restated every ``speech.status_interval_s``,
so whoever watches -- the console, an examiner -- can tell a pipeline waiting
quietly for the wake word from one that has died. Nothing depends on it:
the pipeline behaves the same whether anyone reads this or not.
"""

from __future__ import annotations

import logging

from src.common import topics
from src.common.clock import Clock
from src.common.config import SpeechConfig
from src.common.mqtt_client import Blackboard
from src.common.schemas import SpeechActivity, SpeechStatus

LOGGER = logging.getLogger(__name__)


class StatusPublisher:
    """Publishes the pipeline's activity, and restates it on a heartbeat."""

    def __init__(
        self, config: SpeechConfig, clock: Clock, blackboard: Blackboard, microphone: str = ""
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._microphone = microphone
        self._activity = SpeechActivity.WAITING
        self._heard = ""
        self._last_published_s: float | None = None

    @property
    def activity(self) -> SpeechActivity:
        return self._activity

    def changed(self, activity: SpeechActivity, heard: str = "") -> None:
        """The pipeline moved on; say so at once."""
        self._activity = activity
        self._heard = heard
        self._publish()

    def heartbeat(self) -> None:
        """Restate the current activity if the interval has passed."""
        now_s = self._clock.monotonic()
        if self._last_published_s is None or now_s - self._last_published_s >= self._config.status_interval_s:
            self._publish()

    def _publish(self) -> None:
        self._last_published_s = self._clock.monotonic()
        status = SpeechStatus(
            ts=self._clock.now(),
            activity=self._activity,
            heard=self._heard,
            microphone=self._microphone,
            wake_word=self._config.wake_word,
            heartbeat_s=self._config.status_interval_s,
            listen_mode=self._config.listen_mode,
        )
        try:
            self._blackboard.publish(topics.SPEECH_STATUS, status)
        except (OSError, ValueError) as exc:  # a courtesy; speech goes on without it
            LOGGER.warning("could not publish speech status: %s", exc)
