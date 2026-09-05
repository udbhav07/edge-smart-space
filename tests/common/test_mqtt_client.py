"""Unit tests for the blackboard.

No broker is involved: the transport is injected. What is tested is the part
that matters to every other component -- that delivery comes from the topic,
that a malformed payload never reaches a handler, and that one handler's bug
does not silence the bus.
"""

from pathlib import Path

import pytest

from src.common import topics
from src.common.config import MqttConfig, load_config
from src.common.mqtt_client import Blackboard, topic_matches
from src.common.schemas import SensorReading, Unit
from src.common.topics import TopicParameterError

SENSOR_ID = "temp_01"
TS = 1756032000.123


@pytest.fixture(name="config")
def _config() -> MqttConfig:
    return load_config(Path("config/default.yaml")).mqtt


class FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.subscribed: list[tuple[str, int]] = []
        self.connected: tuple | None = None
        self.looping = False
        self.disconnected = False

    def connect(self, host, port, keepalive):
        self.connected = (host, port, keepalive)

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))

    def loop_start(self):
        self.looping = True

    def loop_stop(self):
        self.looping = False

    def disconnect(self):
        self.disconnected = True


def _reading() -> SensorReading:
    return SensorReading(
        ts=TS, sensor_id=SENSOR_ID, value=27.4, unit=Unit.CELSIUS
    )


def _blackboard(config) -> tuple[Blackboard, FakeTransport]:
    transport = FakeTransport()
    return Blackboard(config, transport), transport


class TestPublishing:
    def test_publishes_to_the_formatted_topic(self, config):
        board, transport = _blackboard(config)
        board.publish(topics.SENSOR_STATE, _reading(), sensor_id=SENSOR_ID)
        assert transport.published[0][0] == f"space/sensor/{SENSOR_ID}/state"

    def test_delivery_comes_from_the_topic_not_the_caller(self, config):
        """A retained state topic cannot be published transient by mistake."""
        board, transport = _blackboard(config)
        board.publish(topics.SYSTEM_MODE, _reading())
        _, _, qos, retain = transport.published[0]
        assert (qos, retain) == (topics.SYSTEM_MODE.qos.value, True)

    def test_a_transient_topic_is_not_retained(self, config):
        board, transport = _blackboard(config)
        board.publish(topics.SENSOR_STATE, _reading(), sensor_id=SENSOR_ID)
        assert transport.published[0][3] is False

    def test_the_payload_is_the_message_as_json(self, config):
        board, transport = _blackboard(config)
        board.publish(topics.SENSOR_STATE, _reading(), sensor_id=SENSOR_ID)
        restored = SensorReading.model_validate_json(transport.published[0][1])
        assert restored == _reading()

    def test_a_bad_topic_parameter_is_refused_before_publishing(self, config):
        board, transport = _blackboard(config)
        with pytest.raises(TopicParameterError):
            board.publish(topics.SENSOR_STATE, _reading(), sensor_id="a/b")
        assert transport.published == []

    def test_the_concrete_topic_is_returned(self, config):
        board, _ = _blackboard(config)
        assert board.publish(topics.SYSTEM_MODE, _reading()) == "space/system/mode"


class TestSubscribing:
    def test_subscribes_with_the_wildcard_form(self, config):
        board, transport = _blackboard(config)
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        board.on_connected()
        assert transport.subscribed[0][0] == "space/sensor/+/state"

    def test_subscribes_at_the_topic_quality_of_service(self, config):
        board, transport = _blackboard(config)
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        board.on_connected()
        assert transport.subscribed[0][1] == topics.SENSOR_STATE.qos.value


class TestSubscriptionLifecycle:
    """MQTT drops subscriptions on disconnect and refuses them before connect.

    These are the tests for the bug that made the estimator connect, sit
    there looking healthy, and never receive a single reading.
    """

    def test_registering_before_connect_does_not_reach_the_broker(self, config):
        """paho returns MQTT_ERR_NO_CONN and the subscription is lost."""
        board, transport = _blackboard(config)
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        assert transport.subscribed == []

    def test_connecting_issues_every_registered_subscription(self, config):
        board, transport = _blackboard(config)
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        board.on_connected()
        assert transport.subscribed == [
            ("space/sensor/+/state", topics.SENSOR_STATE.qos.value)
        ]

    def test_reconnecting_issues_them_again(self, config):
        """A fresh session starts with no subscriptions at all. Without this
        a component keeps running, looks healthy, and silently stops
        receiving anything after the first blip."""
        board, transport = _blackboard(config)
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        board.on_connected()
        board.on_disconnected()
        board.on_connected()
        assert len(transport.subscribed) == 2

    def test_subscribing_while_already_connected_takes_effect_at_once(self, config):
        board, transport = _blackboard(config)
        board.on_connected()
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        assert transport.subscribed

    def test_every_registered_pattern_is_reported(self, config):
        board, _ = _blackboard(config)
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        board.subscribe(topics.SYSTEM_MODE, SensorReading, lambda t, m: None)
        assert set(board.subscriptions) == {
            "space/sensor/+/state",
            "space/system/mode",
        }

    def test_the_same_topic_is_not_subscribed_twice(self, config):
        board, transport = _blackboard(config)
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        board.subscribe(topics.SENSOR_STATE, SensorReading, lambda t, m: None)
        board.on_connected()
        assert len(transport.subscribed) == 1


class TestDispatch:
    def _received(self, config):
        board, _ = _blackboard(config)
        seen: list[tuple[str, SensorReading]] = []
        board.subscribe(
            topics.SENSOR_STATE, SensorReading, lambda t, m: seen.append((t, m))
        )
        return board, seen

    def test_a_valid_payload_reaches_the_handler_decoded(self, config):
        board, seen = self._received(config)
        board.dispatch(
            f"space/sensor/{SENSOR_ID}/state", _reading().model_dump_json().encode()
        )
        assert seen[0][1] == _reading()

    def test_the_handler_is_told_which_topic_it_came_from(self, config):
        board, seen = self._received(config)
        topic = f"space/sensor/{SENSOR_ID}/state"
        board.dispatch(topic, _reading().model_dump_json().encode())
        assert seen[0][0] == topic

    def test_a_malformed_payload_never_reaches_the_handler(self, config):
        """Fail fast at the boundary: Layer 2 never sees a bad message."""
        board, seen = self._received(config)
        board.dispatch(f"space/sensor/{SENSOR_ID}/state", b"{not json")
        assert seen == []

    def test_a_payload_failing_its_schema_never_reaches_the_handler(self, config):
        board, seen = self._received(config)
        board.dispatch(f"space/sensor/{SENSOR_ID}/state", b'{"ts": -1}')
        assert seen == []

    def test_undecodable_bytes_are_dropped(self, config):
        board, seen = self._received(config)
        board.dispatch(f"space/sensor/{SENSOR_ID}/state", b"\xff\xfe")
        assert seen == []

    def test_a_message_on_an_unsubscribed_topic_is_ignored(self, config):
        board, seen = self._received(config)
        board.dispatch("space/goal/active", _reading().model_dump_json().encode())
        assert seen == []

    def test_a_raising_handler_does_not_take_the_client_down(self, config):
        """One component's bug must not silence the bus for everyone else."""
        board, _ = _blackboard(config)
        board.subscribe(
            topics.SENSOR_STATE,
            SensorReading,
            lambda t, m: (_ for _ in ()).throw(RuntimeError("bug")),
        )
        board.dispatch(
            f"space/sensor/{SENSOR_ID}/state", _reading().model_dump_json().encode()
        )


class TestLifecycle:
    def test_start_connects_with_the_configured_broker(self, config):
        board, transport = _blackboard(config)
        board.start()
        assert transport.connected == (config.host, config.port, int(config.keepalive_s))

    def test_start_begins_serving_callbacks(self, config):
        board, transport = _blackboard(config)
        board.start()
        assert transport.looping is True

    def test_stop_disconnects(self, config):
        board, transport = _blackboard(config)
        board.start()
        board.stop()
        assert transport.looping is False and transport.disconnected is True


class TestTopicMatching:
    @pytest.mark.parametrize(
        ("pattern", "topic"),
        [
            ("space/system/mode", "space/system/mode"),
            ("space/sensor/+/state", "space/sensor/temp_01/state"),
            ("space/fault/+", "space/fault/f_123"),
            ("space/#", "space/anything/at/all"),
            ("space/#", "space/one"),
        ],
    )
    def test_matching_topics(self, pattern, topic):
        assert topic_matches(pattern, topic) is True

    @pytest.mark.parametrize(
        ("pattern", "topic"),
        [
            ("space/sensor/+/state", "space/sensor/temp_01/health"),
            ("space/sensor/+/state", "space/sensor/state"),
            ("space/sensor/+/state", "space/sensor/a/b/state"),
            ("space/system/mode", "space/system/mode/extra"),
            ("space/fault/+", "space/fault"),
        ],
    )
    def test_non_matching_topics(self, pattern, topic):
        assert topic_matches(pattern, topic) is False
