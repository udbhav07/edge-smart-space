"""Recording and replay, including the round trip (FR-62).

The property that matters is not that each half works but that they compose:
what comes out of a replay has to be what went into the recording, on the same
topics, in the same order, with the delivery each topic declares. A recorder
that drops a message and a replayer that invents one both pass their own unit
tests.
"""

import json
from pathlib import Path

import pytest

from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    FaultClass,
    DetectorId,
    FaultEvent,
    Mode,
    ModeState,
    SensorReading,
    Unit,
)
from sim.replay import (
    UNTIMED,
    RecordingError,
    Replayer,
    parse_line,
    read,
)
from tools.record import Recorder, attach

SENSOR_ID = "temp_01"
TS = 1756032000.0


class FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.subscribed: list[tuple[str, int]] = []

    def connect(self, host, port, keepalive):
        pass

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


def _reading(value: float = 27.4) -> SensorReading:
    return SensorReading(
        ts=TS, sensor_id=SENSOR_ID, value=value, unit=Unit.CELSIUS
    )


def _mode() -> ModeState:
    return ModeState(
        ts=TS, mode=Mode.NORMAL, since_ts=TS, reason="no active faults"
    )


def _fault() -> FaultEvent:
    return FaultEvent(
        fault_id="f_temp01_stuck_1756032300",
        detector=DetectorId.D2_STUCK_AT,
        subject=SENSOR_ID,
        fault_class=FaultClass.SENSOR,
        confidence=0.94,
        detected_ts=TS,
        evidence={"variance": 0.0002},
        mode_impact=Mode.DEGRADED_SENSOR,
    )


class TestRecording:
    def _recorder(self, clock, tmp_path):
        stream = (tmp_path / "run.jsonl").open("w", encoding="utf-8")
        return Recorder(clock, stream), stream

    def test_a_message_is_written_as_one_line(self, clock, tmp_path):
        recorder, stream = self._recorder(clock, tmp_path)
        recorder.on_message("space/system/mode", _mode().model_dump_json().encode())
        recorder.flush()
        stream.close()
        assert len((tmp_path / "run.jsonl").read_text().splitlines()) == 1

    def test_the_line_holds_the_topic_and_the_payload_verbatim(
        self, clock, tmp_path
    ):
        """Decoding here would bind the recording to this build's schemas."""
        recorder, stream = self._recorder(clock, tmp_path)
        payload = _mode().model_dump_json().encode()
        recorder.on_message("space/system/mode", payload)
        recorder.flush()
        stream.close()
        line = json.loads((tmp_path / "run.jsonl").read_text().splitlines()[0])
        assert line["topic"] == "space/system/mode"
        assert line["payload"].encode() == payload

    def test_the_line_is_stamped_from_the_injected_clock(self, clock, tmp_path):
        recorder, stream = self._recorder(clock, tmp_path)
        clock.advance(120.0)
        recorder.on_message("space/system/mode", _mode().model_dump_json().encode())
        recorder.flush()
        stream.close()
        line = json.loads((tmp_path / "run.jsonl").read_text().splitlines()[0])
        assert line["ts"] == clock.now()

    def test_a_withdrawn_retained_message_is_recorded(self, clock, tmp_path):
        """An empty payload withdraws a fault, which is a real event in the
        run. A recording that dropped it would replay into a different state
        than the one recorded."""
        recorder, stream = self._recorder(clock, tmp_path)
        recorder.on_message("space/fault/f_x_1", b"")
        recorder.flush()
        stream.close()
        assert recorder.count == 1

    def test_an_undecodable_payload_does_not_end_the_run(self, clock, tmp_path):
        recorder, stream = self._recorder(clock, tmp_path)
        recorder.on_message("space/system/mode", b"\xff\xfe")
        recorder.on_message("space/system/mode", _mode().model_dump_json().encode())
        recorder.flush()
        stream.close()
        assert recorder.count == 1

    def test_it_counts_what_it_wrote(self, clock, tmp_path):
        recorder, stream = self._recorder(clock, tmp_path)
        for _ in range(5):
            recorder.on_message("space/system/mode", _mode().model_dump_json().encode())
        stream.close()
        assert recorder.count == 5


class TestRecorderWiring:
    def test_it_subscribes_to_the_whole_tree(self, config, clock, tmp_path):
        transport = FakeTransport()
        blackboard = Blackboard(config.mqtt, transport)
        stream = (tmp_path / "run.jsonl").open("w", encoding="utf-8")
        attach(blackboard, Recorder(clock, stream))
        blackboard.on_connected()
        stream.close()
        assert any(topic == topics.ALL_TOPICS for topic, _ in transport.subscribed)

    def test_it_records_what_the_bus_delivers(self, config, clock, tmp_path):
        transport = FakeTransport()
        blackboard = Blackboard(config.mqtt, transport)
        stream = (tmp_path / "run.jsonl").open("w", encoding="utf-8")
        recorder = Recorder(clock, stream)
        attach(blackboard, recorder)
        blackboard.on_connected()

        blackboard.dispatch(
            f"space/sensor/{SENSOR_ID}/state", _reading().model_dump_json().encode()
        )
        stream.close()
        assert recorder.count == 1

    def test_a_payload_failing_its_schema_is_still_recorded(
        self, config, clock, tmp_path
    ):
        """The typed path drops it; the recording must not, or a replay would
        not reproduce what the system actually received."""
        transport = FakeTransport()
        blackboard = Blackboard(config.mqtt, transport)
        stream = (tmp_path / "run.jsonl").open("w", encoding="utf-8")
        recorder = Recorder(clock, stream)
        attach(blackboard, recorder)
        blackboard.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        blackboard.on_connected()

        blackboard.dispatch(f"space/sensor/{SENSOR_ID}/state", b'{"ts": -1}')
        stream.close()
        assert recorder.count == 1

    def test_a_raising_recorder_does_not_silence_the_bus(self, config):
        transport = FakeTransport()
        blackboard = Blackboard(config.mqtt, transport)
        seen: list[str] = []
        blackboard.subscribe_raw(
            topics.ALL_TOPICS,
            lambda t, p: (_ for _ in ()).throw(RuntimeError("bug")),
        )
        blackboard.subscribe(
            topics.SENSOR_STATE, SensorReading, lambda t, m: seen.append(t)
        )
        blackboard.dispatch(
            f"space/sensor/{SENSOR_ID}/state", _reading().model_dump_json().encode()
        )
        assert seen


class TestReadingARecording:
    def test_a_well_formed_line_parses(self):
        line = json.dumps({"ts": TS, "topic": "space/system/mode", "payload": "{}"})
        assert parse_line(line, 1).topic == "space/system/mode"

    @pytest.mark.parametrize("key", ["ts", "topic", "payload"])
    def test_a_line_missing_a_field_is_refused(self, key):
        fields = {"ts": TS, "topic": "space/system/mode", "payload": "{}"}
        del fields[key]
        with pytest.raises(RecordingError):
            parse_line(json.dumps(fields), 1)

    def test_a_line_that_is_not_json_is_refused(self):
        with pytest.raises(RecordingError):
            parse_line("{not json", 1)

    def test_a_truncated_recording_is_read_up_to_the_last_good_line(
        self, tmp_path
    ):
        """What a run killed mid-write looks like, and still usable."""
        path = tmp_path / "run.jsonl"
        good = json.dumps({"ts": TS, "topic": "space/system/mode", "payload": "{}"})
        path.write_text(f"{good}\n{good}\n{{partial", encoding="utf-8")
        assert len(list(read(path))) == 2

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "run.jsonl"
        good = json.dumps({"ts": TS, "topic": "space/system/mode", "payload": "{}"})
        path.write_text(f"{good}\n\n{good}\n", encoding="utf-8")
        assert len(list(read(path))) == 2


class TestReplaying:
    def _replayer(self, config, clock, speed=UNTIMED):
        transport = FakeTransport()
        blackboard = Blackboard(config.mqtt, transport)
        return Replayer(blackboard, clock, speed=speed), transport

    def _recording(self, tmp_path, entries) -> Path:
        path = tmp_path / "run.jsonl"
        path.write_text(
            "\n".join(json.dumps(entry) for entry in entries) + "\n",
            encoding="utf-8",
        )
        return path

    def test_a_recorded_message_is_republished_on_its_topic(
        self, config, clock, tmp_path
    ):
        replayer, transport = self._replayer(config, clock)
        path = self._recording(
            tmp_path,
            [{"ts": TS, "topic": "space/system/mode", "payload": "{}"}],
        )
        replayer.play(read(path))
        assert transport.published[0][0] == "space/system/mode"

    def test_the_payload_is_republished_untouched(self, config, clock, tmp_path):
        """Re-encoding would make the replay reflect this build's
        serialisation rather than what crossed the wire."""
        replayer, transport = self._replayer(config, clock)
        payload = _fault().model_dump_json(by_alias=True)
        path = self._recording(
            tmp_path,
            [{"ts": TS, "topic": "space/fault/f_1", "payload": payload}],
        )
        replayer.play(read(path))
        assert transport.published[0][1].decode() == payload

    def test_delivery_comes_from_the_topic_not_the_recording(
        self, config, clock, tmp_path
    ):
        """A recording cannot observe whether a message was retained, so
        guessing would replay a state topic as transient."""
        replayer, transport = self._replayer(config, clock)
        path = self._recording(
            tmp_path,
            [{"ts": TS, "topic": "space/system/mode", "payload": "{}"}],
        )
        replayer.play(read(path))
        _, _, qos, retain = transport.published[0]
        assert (qos, retain) == (topics.SYSTEM_MODE.qos.value, True)

    def test_order_is_preserved(self, config, clock, tmp_path):
        replayer, transport = self._replayer(config, clock)
        path = self._recording(
            tmp_path,
            [
                {"ts": TS, "topic": "space/system/mode", "payload": "{}"},
                {"ts": TS + 1, "topic": "space/estimate/thermal", "payload": "{}"},
                {"ts": TS + 2, "topic": "space/goal/active", "payload": "{}"},
            ],
        )
        replayer.play(read(path))
        assert [entry[0] for entry in transport.published] == [
            "space/system/mode",
            "space/estimate/thermal",
            "space/goal/active",
        ]

    def test_an_undeclared_topic_is_skipped_rather_than_fatal(
        self, config, clock, tmp_path
    ):
        """A recording from a newer build is a thing to report, not a crash."""
        replayer, transport = self._replayer(config, clock)
        path = self._recording(
            tmp_path,
            [
                {"ts": TS, "topic": "space/from/the/future", "payload": "{}"},
                {"ts": TS, "topic": "space/system/mode", "payload": "{}"},
            ],
        )
        replayer.play(read(path))
        assert (replayer.published, replayer.skipped) == (1, 1)

    def test_timing_follows_the_recorded_gaps(self, config, clock, tmp_path):
        replayer, _ = self._replayer(config, clock, speed=1.0)
        path = self._recording(
            tmp_path,
            [
                {"ts": TS, "topic": "space/system/mode", "payload": "{}"},
                {"ts": TS + 5.0, "topic": "space/system/mode", "payload": "{}"},
                {"ts": TS + 15.0, "topic": "space/system/mode", "payload": "{}"},
            ],
        )
        started = clock.now()
        replayer.play(read(path))
        assert clock.now() - started == pytest.approx(15.0)

    def test_speed_scales_the_gaps(self, config, clock, tmp_path):
        replayer, _ = self._replayer(config, clock, speed=10.0)
        path = self._recording(
            tmp_path,
            [
                {"ts": TS, "topic": "space/system/mode", "payload": "{}"},
                {"ts": TS + 100.0, "topic": "space/system/mode", "payload": "{}"},
            ],
        )
        started = clock.now()
        replayer.play(read(path))
        assert clock.now() - started == pytest.approx(10.0)

    def test_untimed_replay_does_not_sleep(self, config, clock, tmp_path):
        replayer, _ = self._replayer(config, clock, speed=UNTIMED)
        path = self._recording(
            tmp_path,
            [
                {"ts": TS, "topic": "space/system/mode", "payload": "{}"},
                {"ts": TS + 3600.0, "topic": "space/system/mode", "payload": "{}"},
            ],
        )
        started = clock.now()
        replayer.play(read(path))
        assert clock.now() == started

    def test_a_backwards_timestamp_does_not_stall_the_replay(
        self, config, clock, tmp_path
    ):
        """Timestamps come from whatever clock the recorder had."""
        replayer, transport = self._replayer(config, clock, speed=1.0)
        path = self._recording(
            tmp_path,
            [
                {"ts": TS + 10.0, "topic": "space/system/mode", "payload": "{}"},
                {"ts": TS, "topic": "space/system/mode", "payload": "{}"},
            ],
        )
        replayer.play(read(path))
        assert len(transport.published) == 2

    def test_a_negative_speed_is_refused(self, config, clock):
        with pytest.raises(ValueError):
            self._replayer(config, clock, speed=-1.0)


class TestTheRoundTrip:
    """What comes out has to be what went in."""

    def test_recorded_messages_replay_byte_for_byte(
        self, config, clock, tmp_path
    ):
        sent = [
            (f"space/sensor/{SENSOR_ID}/state", _reading().model_dump_json()),
            ("space/system/mode", _mode().model_dump_json()),
            (
                f"space/fault/{_fault().fault_id}",
                _fault().model_dump_json(by_alias=True),
            ),
        ]

        path = tmp_path / "run.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            recorder = Recorder(clock, stream)
            for topic, payload in sent:
                recorder.on_message(topic, payload.encode())
                clock.advance(5.0)

        transport = FakeTransport()
        replayer = Replayer(
            Blackboard(config.mqtt, transport), SimClock(), speed=UNTIMED
        )
        replayer.play(read(path))

        replayed = [
            (topic, payload.decode())
            for topic, payload, _, _ in transport.published
        ]
        assert replayed == sent

    def test_a_replayed_run_still_validates_against_the_schemas(
        self, config, clock, tmp_path
    ):
        """The recording is bytes, but those bytes were real messages."""
        path = tmp_path / "run.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            recorder = Recorder(clock, stream)
            recorder.on_message(
                f"space/sensor/{SENSOR_ID}/state", _reading().model_dump_json().encode()
            )

        transport = FakeTransport()
        replayer = Replayer(
            Blackboard(config.mqtt, transport), SimClock(), speed=UNTIMED
        )
        replayer.play(read(path))
        restored = SensorReading.model_validate_json(transport.published[0][1])
        assert restored == _reading()

    def test_a_withdrawal_survives_the_round_trip(self, config, clock, tmp_path):
        """The empty payload that retires a fault has to come back out, or a
        replayed run would hold a fault the recorded one had cleared."""
        path = tmp_path / "run.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            Recorder(clock, stream).on_message("space/fault/f_1", b"")

        transport = FakeTransport()
        replayer = Replayer(
            Blackboard(config.mqtt, transport), SimClock(), speed=UNTIMED
        )
        replayer.play(read(path))
        assert transport.published[0][1] == b""
