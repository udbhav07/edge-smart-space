"""Unit tests for D5, the actuator response detector (FR-24).

No acknowledgement appears anywhere in these tests, which is the point: R-02
says there is none to have. What is asserted is that the verdict comes from
the plant moving, and that MAINTAIN does not look like cooling having stopped.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import ActuatorDetectorConfig, load_config
from src.common.schemas import CommandKind, DetectorId, SensorReading, Unit
from src.faults.detectors.actuator import ActuatorResponseDetector
from src.faults.detectors.base import Judgment

ACTUATOR = "ac"
SENSOR = "temp_01"
WINDOW_S = 600.0
MIN_COOLING_C = 0.3
START_C = 29.0


def _config(**overrides) -> ActuatorDetectorConfig:
    return ActuatorDetectorConfig(
        **{
            "evaluation_window_s": WINDOW_S,
            "min_cooling_c": MIN_COOLING_C,
            **overrides,
        }
    )


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="detector")
def _detector(clock) -> ActuatorResponseDetector:
    return ActuatorResponseDetector(
        subject=ACTUATOR, config=_config(), clock=clock
    )


def _reading(clock, value: float) -> SensorReading:
    return SensorReading(
        ts=clock.now(), sensor_id=SENSOR, value=value, unit=Unit.CELSIUS
    )


def _cool_for(detector, clock, seconds: float, final_c: float, start_c=START_C):
    """Command cooling, hold it for a while, and end at a given temperature."""
    detector.observe_reading(_reading(clock, start_c))
    detector.observe_command(CommandKind.COOL)
    clock.advance(seconds)
    detector.observe_reading(_reading(clock, final_c))


class TestNothingToJudge:
    def test_an_idle_actuator_is_unknown(self, detector):
        """An actuator nobody has asked to do anything has not been shown to
        work; CLEAR would claim evidence never gathered."""
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_cooling_for_less_than_the_window_is_unknown(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S / 2, final_c=START_C)
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_an_actuator_turned_off_goes_back_to_unknown(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C - 1.0)
        detector.observe_command(CommandKind.OFF)
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_cooling_with_no_reading_yet_is_unknown(self, detector, clock):
        """A window with no temperatures in it would otherwise blame the air
        conditioner for a broken sensor."""
        detector.observe_command(CommandKind.COOL)
        clock.advance(WINDOW_S * 2)
        assert detector.evaluate().judgment is Judgment.UNKNOWN


class TestAWorkingActuator:
    def test_a_room_that_cools_is_clear(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C - 1.0)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_cooling_exactly_the_minimum_is_enough(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C - MIN_COOLING_C)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_a_pass_re_anchors_so_the_next_window_is_a_fresh_test(
        self, detector, clock
    ):
        """One success must not stand as a verdict forever."""
        _cool_for(detector, clock, WINDOW_S, final_c=START_C - 1.0)
        assert detector.evaluate().judgment is Judgment.CLEAR

        clock.advance(WINDOW_S)
        detector.observe_reading(_reading(clock, START_C - 1.0))
        assert detector.evaluate().judgment is Judgment.FAULTED


class TestABrokenActuator:
    def test_a_room_that_does_not_move_is_a_fault(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_a_room_that_gets_warmer_is_also_a_fault(self, detector, clock):
        """Section 5.5's |dT| test would miss this, and it is the more certain
        actuator fault of the two."""
        _cool_for(detector, clock, WINDOW_S, final_c=START_C + 1.0)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_cooling_just_short_of_the_minimum_is_a_fault(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C - MIN_COOLING_C + 0.01)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_the_fault_is_reported_with_full_confidence(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C)
        assert detector.evaluate().confidence == 1.0

    def test_no_acknowledgement_is_required_to_raise_it(self, detector, clock):
        """R-02: there is none to have. The plant not moving is the evidence."""
        _cool_for(detector, clock, WINDOW_S, final_c=START_C)
        assert detector.evaluate().judgment is Judgment.FAULTED


class TestMaintainDoesNotLookLikeStopping:
    def test_maintain_does_not_restart_the_window(self, detector, clock):
        """The controller emits MAINTAIN on every tick while the compressor
        stays on. Counting COOL messages would see one and conclude cooling
        had stopped."""
        detector.observe_reading(_reading(clock, START_C))
        detector.observe_command(CommandKind.COOL)
        for _ in range(10):
            clock.advance(WINDOW_S / 10)
            detector.observe_command(CommandKind.MAINTAIN)
            detector.observe_reading(_reading(clock, START_C))
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_maintain_leaves_an_idle_actuator_idle(self, detector, clock):
        detector.observe_command(CommandKind.MAINTAIN)
        clock.advance(WINDOW_S * 2)
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_hold_does_not_start_an_evaluation(self, detector, clock):
        detector.observe_command(CommandKind.HOLD)
        assert not detector.cooling

    def test_a_second_cool_does_not_restart_the_window(self, detector, clock):
        """Otherwise a controller repeating COOL would reset the clock every
        tick and the window would never elapse."""
        detector.observe_reading(_reading(clock, START_C))
        detector.observe_command(CommandKind.COOL)
        for _ in range(10):
            clock.advance(WINDOW_S / 10)
            detector.observe_command(CommandKind.COOL)
            detector.observe_reading(_reading(clock, START_C))
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_an_off_between_two_cools_does_restart_it(self, detector, clock):
        """A window straddling an OFF would compare two different regimes."""
        _cool_for(detector, clock, WINDOW_S / 2, final_c=START_C)
        detector.observe_command(CommandKind.OFF)
        detector.observe_command(CommandKind.COOL)
        clock.advance(WINDOW_S / 2)
        detector.observe_reading(_reading(clock, START_C))
        assert detector.evaluate().judgment is Judgment.UNKNOWN


class TestResetting:
    def test_resetting_abandons_the_current_window(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C)
        assert detector.evaluate().judgment is Judgment.FAULTED

        detector.reset()
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_after_a_reset_a_working_actuator_reads_clear(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C)
        detector.reset()

        detector.observe_reading(_reading(clock, START_C))
        detector.observe_command(CommandKind.COOL)
        clock.advance(WINDOW_S)
        detector.observe_reading(_reading(clock, START_C - 1.0))
        assert detector.evaluate().judgment is Judgment.CLEAR


class TestEvidence:
    def test_the_finding_names_the_detector(self, detector):
        assert detector.evaluate().detector is DetectorId.D5_ACTUATOR_NO_RESPONSE

    def test_the_finding_names_the_actuator_not_the_sensor(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C)
        assert detector.evaluate().subject == ACTUATOR

    def test_the_evidence_carries_the_cooling_achieved_and_required(
        self, detector, clock
    ):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C - 0.1)
        evidence = detector.evaluate().evidence
        assert evidence["cooled_c"] == pytest.approx(0.1)
        assert evidence["min_cooling_c"] == MIN_COOLING_C

    def test_the_evidence_carries_both_ends_of_the_window(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C - 0.1)
        evidence = detector.evaluate().evidence
        assert evidence["start_temperature_c"] == START_C
        assert evidence["latest_temperature_c"] == pytest.approx(START_C - 0.1)

    def test_the_evidence_reports_how_long_the_window_ran(self, detector, clock):
        _cool_for(detector, clock, WINDOW_S, final_c=START_C)
        assert detector.evaluate().evidence["elapsed_s"] == pytest.approx(WINDOW_S)


class TestConstruction:
    def test_an_unnamed_subject_is_refused(self, clock):
        with pytest.raises(ValueError):
            ActuatorResponseDetector(subject="", config=_config(), clock=clock)

    def test_the_documented_defaults_match_the_design(self):
        """Section 5.5: window 600 s, threshold 0.3 C."""
        config = load_config(Path("config/default.yaml")).detectors.actuator
        assert config.evaluation_window_s == 600.0
        assert config.min_cooling_c == 0.3
