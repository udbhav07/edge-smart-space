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
    DiagnosisConfidence,
    FaultDiagnosis,
    Goal,
    GoalSource,
    Hypothesis,
    Mode,
    Quality,
    SensorHealth,
    SensorReading,
    TariffBand,
    TrackingState,
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


class TestWhatTheOccupantWasNotTold:
    """U-04, U-14, U-21: the view shows what matters even when it is absent."""

    def test_an_expected_sensor_that_never_spoke_is_listed_as_silent(self, clock):
        view = BlackboardView(clock=clock, expected_sensors=("temp_01", "hum_01"))
        view.accept("space/sensor/hum_01/state", _reading(clock).replace(b"temp_01", b"hum_01"))
        assert any("temp_01" in line and "silent" in line for line in summarise(view).splitlines())

    def test_a_troubled_sensor_is_listed_first(self, clock):
        view = BlackboardView(clock=clock, expected_sensors=("hum_01", "temp_01"))
        view.accept("space/sensor/hum_01/state", _reading(clock).replace(b"temp_01", b"hum_01"))
        lines = summarise(view).splitlines()
        first = lines[lines.index("SENSORS") + 1]
        assert first.split()[0] == "temp_01"

    def test_a_faulted_sensor_names_its_fault(self, view, clock):
        view.accept(TEMP_TOPIC, _reading(clock))
        health = SensorHealth(ts=clock.now(), sensor_id="temp_01", quality=Quality.FAULTED, active_fault_id="f_d2")
        view.accept("space/sensor/temp_01/health", health.model_dump_json().encode())
        assert "fault f_d2" in summarise(view)

    def test_a_diagnosis_is_shown_to_a_late_viewer(self, view, clock):
        diagnosis = FaultDiagnosis(
            ts=clock.now(),
            fault_id="f_d2",
            primary_hypothesis=Hypothesis.SENSOR_STUCK,
            confidence=DiagnosisConfidence.HIGH,
            recommended_mode=Mode.DEGRADED_SENSOR,
            user_message="The thermometer is stuck.",
        )
        view.accept("space/diagnosis/f_d2", diagnosis.model_dump_json().encode())
        assert "f_d2: The thermometer is stuck." in summarise(view)

    def test_a_withdrawn_retained_message_is_forgotten_not_a_failure(self, view, clock):
        view.accept("space/diagnosis/f_d2", b"")
        assert "DECODE FAILURES" not in summarise(view) and "DIAGNOSIS" not in summarise(view)

    def test_the_held_setpoint_is_shown_beside_the_asked_one(self, view, clock):
        tracking = TrackingState(
            ts=clock.now(), setpoint_c=22.0, tracked_c=23.0, band=TariffBand.PEAK, reason="peak tariff"
        )
        view.accept("space/control/tracking", tracking.model_dump_json().encode())
        assert "HOLDING   23.0 C" in summarise(view) and "asked 22.0 C" in summarise(view)

    def test_cool_beside_a_standby_draw_is_pointed_out(self, clock):
        view = BlackboardView(clock=clock, power_sensor_id="pwr_01", running_power_w=200.0)
        state = ActuatorState(
            ts=clock.now(), actuator_id="ac", simulated=True, kind=CommandKind.COOL,
            setpoint_c=24.0, ack=AckStatus.UNKNOWN,
        )
        meter = SensorReading(ts=clock.now(), sensor_id="pwr_01", value=16.0, unit=Unit.WATT)
        view.accept("space/actuator/ac/state", state.model_dump_json().encode())
        view.accept("space/sensor/pwr_01/state", meter.model_dump_json().encode())
        assert "not running" in summarise(view)


class TestStreamLines:
    def test_a_goal_reads_naturally_not_as_a_repr(self, view, clock):
        """U-22: no <GoalSource.PREFERENCE: 'preference'> in the stream."""
        goal = Goal(ts=clock.now(), source=GoalSource.PREFERENCE, setpoint_c=23.0, mode=Mode.NORMAL, expires_ts=clock.now() + 60)
        entry = view.accept("space/goal/active", goal.model_dump_json().encode())
        line = describe("space/goal/active", entry)
        assert line.startswith("[goal") and "<" not in line

    def test_an_unknown_topic_is_shown_as_json(self, view):
        entry = view.accept("space/other/thing", b'{"a": 1}')
        assert describe("space/other/thing", entry).endswith('{"a": 1}')

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
