"""The speech pipeline's status on the blackboard (FR-60)."""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import SpeechActivity, SpeechStatus
from src.speech.status import StatusPublisher


class Transport:
    def __init__(self):
        self.published = []
    def connect(self, *a): ...
    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, retain))
    def subscribe(self, *a): ...
    def loop_start(self): ...
    def loop_stop(self): ...
    def disconnect(self): ...

    def statuses(self):
        return [SpeechStatus.model_validate_json(p) for t, p, _ in self.published if t == "space/speech/status"]


@pytest.fixture(name="wired")
def _wired():
    config = load_config(Path("config/default.yaml"))
    clock = SimClock()
    transport = Transport()
    status = StatusPublisher(config.speech, clock, Blackboard(config.mqtt, transport), "Microphone (C-Media)")
    return status, transport, clock, config


def test_a_change_is_published_at_once(wired):
    status, transport, _, _ = wired
    status.changed(SpeechActivity.UNDERSTANDING, "make it cooler")
    published = transport.statuses()[-1]
    assert published.activity is SpeechActivity.UNDERSTANDING and published.heard == "make it cooler"


def test_it_is_retained_so_a_late_console_sees_it(wired):
    status, transport, _, _ = wired
    status.changed(SpeechActivity.WAITING)
    assert transport.published[-1][2] is True


def test_the_heartbeat_waits_for_its_interval(wired):
    status, transport, clock, config = wired
    status.changed(SpeechActivity.WAITING)
    clock.advance(config.speech.status_interval_s / 2)
    status.heartbeat()
    assert len(transport.statuses()) == 1
    clock.advance(config.speech.status_interval_s)
    status.heartbeat()
    assert len(transport.statuses()) == 2


def test_it_names_the_microphone(wired):
    status, transport, _, _ = wired
    status.changed(SpeechActivity.WAITING)
    assert transport.statuses()[-1].microphone == "Microphone (C-Media)"
