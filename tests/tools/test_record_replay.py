"""A recorded run replays onto the same topics, in order and in time (FR-62)."""

import io
import json

import pytest

from sim.replay import Recorded, delivery, load, replay
from src.common.clock import SimClock
from tools.record import Recorder


def test_every_message_is_recorded_verbatim_with_its_offset():
    clock = SimClock()
    sink = io.StringIO()
    recorder = Recorder(clock, sink)
    clock.advance(2.5)
    recorder.dispatch("space/system/mode", b'{"mode": "NORMAL"}')
    entry = json.loads(sink.getvalue())
    assert entry == {"t": 2.5, "topic": "space/system/mode", "payload": '{"mode": "NORMAL"}'}


def test_the_recorder_listens_to_everything_under_the_root():
    class Transport:
        subscribed = []
        def subscribe(self, topic, qos):
            self.subscribed.append(topic)
    recorder = Recorder(SimClock(), io.StringIO())
    transport = Transport()
    recorder.attach(transport)
    recorder.on_connected()
    assert transport.subscribed == ["space/#"]


def test_a_recording_reads_back(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"t": 0.0, "topic": "a", "payload": "x"}\nnot json\n', encoding="utf-8")
    assert load(path) == [Recorded(0.0, "a", "x")]


def test_replay_keeps_order_and_timing():
    clock = SimClock()
    published = []
    recording = [Recorded(0.0, "space/system/mode", "1"), Recorded(10.0, "space/system/mode", "2")]
    replay(recording, lambda *args: published.append((clock.monotonic(), args[1])), clock)
    assert [payload for _, payload in published] == [b"1", b"2"]
    assert published[1][0] - published[0][0] == pytest.approx(10.0)


def test_replay_can_run_faster():
    clock = SimClock()
    started = clock.monotonic()
    replay([Recorded(100.0, "space/system/mode", "x")], lambda *args: None, clock, speed=10.0)
    assert clock.monotonic() - started == pytest.approx(10.0)


def test_retain_comes_from_the_topic_table():
    assert delivery("space/system/mode") == (1, True)
    assert delivery("space/sensor/temp_01/state") == (0, False)


def test_replay_can_be_restricted_to_some_topics():
    published = []
    recording = [Recorded(0.0, "space/sensor/temp_01/state", "a"), Recorded(0.0, "space/goal/proposed", "b")]
    count = replay(recording, lambda topic, *rest: published.append(topic), SimClock(), only=("space/sensor/",))
    assert count == 1 and published == ["space/sensor/temp_01/state"]


def test_a_non_positive_speed_is_refused():
    with pytest.raises(ValueError):
        replay([], lambda *args: None, SimClock(), speed=0.0)
