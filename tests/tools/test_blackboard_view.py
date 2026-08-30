"""Unit tests for the blackboard view.

No broker: the view accumulates messages handed to it directly, which is
also how it works in production -- the MQTT client only calls ``accept``.

The behaviour worth guarding is the part that answers a real question during
debugging: which declared topics have gone silent, and which publishers have
drifted from the contract.
"""

from pathlib import Path

import pytest

from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.schemas import (
    AckStatus,
    ActuatorState,
    AdaptationState,
    Coefficients,
    CommandKind,
    SensorReading,
    ThermalEstimate,
    Unit,
)
from tools.blackboard_view import BlackboardView, describe, summarise

TEMP_TOPIC = "space/sensor/temp_01/state"


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="view")
def _view(clock) -> BlackboardView:
    return BlackboardView(clock=clock)


def _reading(clock, value: float = 27.4) -> bytes:
    return (
        SensorReading(ts=clock.now(), sensor_id="temp_01", value=value, unit=Unit.CELSIUS)
        .model_dump_json()
        .encode()
    )


def _coefficients(clock) -> bytes:
    return (
        Coefficients(
            ts=clock.now(),
            a1=0.996518,
            a2=0.003482,
            a3=-0.020933,
            a4=0.036342,
            trace_p=0.031,
            steady_state_residual=0.0,
            samples_since_reset=14203,
        )
        .model_dump_json()
        .encode()
    )


class TestAccumulating:
    def test_a_message_is_counted(self, view, clock):
        view.accept(TEMP_TOPIC, _reading(clock))
        assert view.message_count == 1

    def test_repeated_messages_accumulate_on_one_topic(self, view, clock):
        for _ in range(5):
            view.accept(TEMP_TOPIC, _reading(clock))
        assert view.activity[TEMP_TOPIC].count == 5

    def test_a_known_topic_is_decoded_with_its_schema(self, view, clock):
        entry = view.accept(TEMP_TOPIC, _reading(clock))
        assert isinstance(entry.latest, SensorReading)
        assert entry.latest.value == 27.4

    def test_the_latest_message_replaces_the_previous_one(self, view, clock):
        view.accept(TEMP_TOPIC, _reading(clock, 20.0))
        entry = view.accept(TEMP_TOPIC, _reading(clock, 30.0))
        assert entry.latest.value == 30.0

    def test_age_is_measured_from_the_injected_clock(self, view, clock):
        view.accept(TEMP_TOPIC, _reading(clock))
        clock.advance(12.0)
        assert view.activity[TEMP_TOPIC].age_s(clock.monotonic()) == pytest.approx(12.0)

    def test_an_unknown_topic_is_still_counted(self, view):
        view.accept("space/something/new", b'{"ts": 1.0}')
        assert view.message_count == 1

    def test_an_unknown_topic_keeps_its_raw_payload(self, view):
        entry = view.accept("space/something/new", b'{"ts": 1.0}')
        assert entry.raw == {"ts": 1.0}


class TestDecodeFailures:
    """A publisher drifting from the contract should be visible as a failure,
    not as plausible-looking nonsense."""

    def test_unparseable_bytes_are_recorded_as_a_failure(self, view):
        entry = view.accept(TEMP_TOPIC, b"{not json")
        assert entry.decode_failures == 1

    def test_a_payload_failing_its_schema_is_recorded_as_a_failure(self, view):
        entry = view.accept(TEMP_TOPIC, b'{"ts": -1, "sensor_id": "x"}')
        assert entry.decode_failures == 1

    def test_a_failed_decode_leaves_no_decoded_message(self, view):
        entry = view.accept(TEMP_TOPIC, b'{"ts": -1}')
        assert entry.latest is None

    def test_a_failed_decode_is_still_counted_as_traffic(self, view):
        view.accept(TEMP_TOPIC, b"{not json")
        assert view.message_count == 1

    def test_failures_appear_in_the_summary(self, view):
        view.accept(TEMP_TOPIC, b"{not json")
        assert "DECODE FAILURES" in summarise(view)


class TestSilence:
    """The line that answers 'is the estimator actually publishing?'."""

    def test_everything_is_silent_before_anything_arrives(self, view):
        assert topics.ESTIMATE_COEFFICIENTS.wildcard() in view.silent_patterns()

    def test_a_topic_stops_being_silent_once_it_speaks(self, view, clock):
        view.accept("space/estimate/coefficients", _coefficients(clock))
        assert topics.ESTIMATE_COEFFICIENTS.wildcard() not in view.silent_patterns()

    def test_a_wildcard_topic_counts_as_heard_from_any_instance(self, view, clock):
        view.accept(TEMP_TOPIC, _reading(clock))
        assert topics.SENSOR_STATE.wildcard() not in view.silent_patterns()

    def test_one_sensor_speaking_does_not_unsilence_the_estimator(self, view, clock):
        """The exact situation of the bug: sensors flowing, estimates not."""
        view.accept(TEMP_TOPIC, _reading(clock))
        silent = view.silent_patterns()
        assert topics.ESTIMATE_COEFFICIENTS.wildcard() in silent
        assert topics.ESTIMATE_THERMAL.wildcard() in silent

    def test_silence_is_reported_in_the_summary(self, view, clock):
        view.accept(TEMP_TOPIC, _reading(clock))
        assert "silent:" in summarise(view)


class TestSummary:
    def test_it_renders_with_nothing_received(self, view):
        assert "blackboard" in summarise(view)

    def test_a_sensor_reading_appears(self, view, clock):
        view.accept(TEMP_TOPIC, _reading(clock))
        assert "temp_01" in summarise(view)

    def test_coefficients_appear(self, view, clock):
        view.accept("space/estimate/coefficients", _coefficients(clock))
        assert "a1" in summarise(view) and "14203 samples" in summarise(view)

    def test_an_actuator_reports_that_it_is_simulated(self, view, clock):
        state = ActuatorState(
            ts=clock.now(),
            actuator_id="ac",
            simulated=True,
            kind=CommandKind.COOL,
            setpoint_c=24.0,
            ack=AckStatus.UNKNOWN,
        )
        view.accept("space/actuator/ac/state", state.model_dump_json().encode())
        assert "simulated" in summarise(view)

    def test_no_faults_is_stated_rather_than_left_blank(self, view):
        assert "FAULTS    none" in summarise(view)

    def test_the_message_count_is_shown(self, view, clock):
        for _ in range(3):
            view.accept(TEMP_TOPIC, _reading(clock))
        assert "3 msgs" in summarise(view)


class TestStreamLines:
    def test_a_sensor_reading_reads_naturally(self, view, clock):
        entry = view.accept(TEMP_TOPIC, _reading(clock))
        assert describe(TEMP_TOPIC, entry).startswith("[sensor")

    def test_coefficients_read_naturally(self, view, clock):
        entry = view.accept("space/estimate/coefficients", _coefficients(clock))
        assert "a1 0.996518" in describe("space/estimate/coefficients", entry)

    def test_a_thermal_estimate_shows_the_residual(self, view, clock):
        estimate = ThermalEstimate(
            ts=clock.now(),
            t_in=27.4,
            t_pred=27.31,
            residual=0.09,
            residual_sigma=0.12,
            model_confidence=0.87,
            adaptation=AdaptationState.ACTIVE,
        )
        entry = view.accept(
            "space/estimate/thermal", estimate.model_dump_json().encode()
        )
        assert "residual" in describe("space/estimate/thermal", entry)

    def test_an_undecodable_message_says_so(self, view):
        entry = view.accept(TEMP_TOPIC, b"{not json")
        assert "undecodable" in describe(TEMP_TOPIC, entry)


class TestNoAuthority:
    """A viewer with power over anything would be a different component."""

    def test_it_holds_no_publisher(self, view):
        assert not any(
            name in vars(view) for name in ("client", "transport", "blackboard")
        )

    def test_it_only_ever_reads(self):
        source = Path("tools/blackboard_view.py").read_text(encoding="utf-8")
        assert ".publish(" not in source
