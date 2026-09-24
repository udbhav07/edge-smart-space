"""Unit tests for D5, the actuator response detector (FR-24).

No acknowledgement appears anywhere in these tests, which is the point: R-02
says there is none to have. What is asserted is that the verdict comes from
the plant moving, and that MAINTAIN does not look like cooling having stopped.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import ActuatorDetectorConfig, load_config
from src.common.schemas import (
    AdaptationState,
    CommandKind,
    DetectorId,
    SensorReading,
    ThermalEstimate,
    Unit,
)
from src.faults.detectors.actuator import ActuatorResponseDetector
from src.faults.detectors.base import Judgment

ACTUATOR = "ac"
SENSOR = "temp_01"
WINDOW_S = 600.0
TS = 1756032000.0
RESPONSE_FRACTION = 0.35
MIN_EXPECTED_C = 0.2
START_C = 29.0

#: What the model expects over a whole window in these tests. Chosen so the
#: bar (35% of it) is a round 0.35 C, which keeps every boundary below legible.
EXPECTED_C = 1.0
REQUIRED_C = EXPECTED_C * RESPONSE_FRACTION


def _config(**overrides) -> ActuatorDetectorConfig:
    return ActuatorDetectorConfig(
        **{
            "evaluation_window_s": WINDOW_S,
            "warmup_samples": 0,
            "min_expected_cooling_c": MIN_EXPECTED_C,
            "response_fraction": RESPONSE_FRACTION,
            **overrides,
        }
    )


def _estimate(t_pred: float, t_in: float) -> ThermalEstimate:
    """An estimate carrying a given prediction, for the expectation."""
    return ThermalEstimate(
        ts=TS,
        t_in=t_in,
        t_pred=t_pred,
        residual=t_in - t_pred,
        residual_sigma=0.15,
        model_confidence=0.9,
        adaptation=AdaptationState.ACTIVE,
    )


def _expect_cooling(detector, total_c: float, steps: int = 10) -> None:
    """Feed estimates whose predicted change sums to the given cooling.

    The model's expectation over a step is its prediction of the next instant
    against the reading it was predicted from, so the pair has to be fed in
    that order for the detector to difference them correctly.
    """
    step_c = total_c / steps
    temperature = START_C
    # A model that predicts perfectly publishes t_pred equal to the reading it
    # predicted, so the change it expected over a step is the difference
    # between consecutive readings. Feeding a t_pred that is already a step
    # ahead of its own t_in would double-count every step.
    detector.observe_estimate(_estimate(t_pred=temperature, t_in=temperature))
    for _ in range(steps):
        temperature -= step_c
        detector.observe_estimate(
            _estimate(t_pred=temperature, t_in=temperature)
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


def _cool_for(
    detector,
    clock,
    seconds: float,
    final_c: float,
    start_c=START_C,
    expected_c: float = EXPECTED_C,
):
    """Command cooling, let the model expect some of it, and end somewhere.

    ``expected_c`` is what the model predicted over the window. The verdict is
    the achieved cooling against a fraction of that, never against a fixed
    number of degrees.
    """
    detector.observe_reading(_reading(clock, start_c))
    detector.observe_command(CommandKind.COOL)
    _expect_cooling(detector, expected_c)
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

    def test_cooling_exactly_the_bar_is_enough(self, detector, clock):
        _cool_for(
            detector, clock, WINDOW_S, final_c=START_C - REQUIRED_C - 1e-6
        )
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_a_room_at_equilibrium_is_not_blamed(self, detector, clock):
        """The model says nothing should happen, so nothing is claimed. With
        a fixed threshold this fired within an hour of every healthy run."""
        _cool_for(
            detector, clock, WINDOW_S, final_c=START_C, expected_c=0.05
        )
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_a_pass_re_anchors_so_the_next_window_is_a_fresh_test(
        self, detector, clock
    ):
        """One success must not stand as a verdict forever."""
        _cool_for(detector, clock, WINDOW_S, final_c=START_C - 1.0)
        assert detector.evaluate().judgment is Judgment.CLEAR

        # A fresh window, with the model still expecting cooling and the room
        # refusing to deliver any.
        _expect_cooling(detector, EXPECTED_C)
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

    def test_cooling_just_short_of_the_bar_is_a_fault(self, detector, clock):
        _cool_for(
            detector, clock, WINDOW_S, final_c=START_C - REQUIRED_C + 0.01
        )
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_the_bar_moves_with_what_the_model_expected(self, detector, clock):
        """Half a degree of cooling passes against a modest expectation and
        fails against a large one. That is the whole point of the change."""
        _cool_for(
            detector, clock, WINDOW_S, final_c=START_C - 0.5, expected_c=1.0
        )
        assert detector.evaluate().judgment is Judgment.CLEAR

        strict = ActuatorResponseDetector(
            subject=ACTUATOR, config=_config(), clock=clock
        )
        _cool_for(
            strict, clock, WINDOW_S, final_c=START_C - 0.5, expected_c=4.0
        )
        assert strict.evaluate().judgment is Judgment.FAULTED

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
        _expect_cooling(detector, EXPECTED_C)
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
        _expect_cooling(detector, EXPECTED_C)
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
        _expect_cooling(detector, EXPECTED_C)
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
        assert evidence["expected_cooling_c"] == pytest.approx(EXPECTED_C)
        assert evidence["required_cooling_c"] == pytest.approx(REQUIRED_C)

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

    def test_the_documented_window_matches_the_design(self):
        """Section 5.5: a 600 s evaluation window."""
        config = load_config(Path("config/default.yaml")).detectors.actuator
        assert config.evaluation_window_s == 600.0

    def test_the_bar_is_a_fraction_rather_than_a_temperature(self):
        """A fixed number of degrees blames the actuator for physics when the
        room is near its equilibrium."""
        config = load_config(Path("config/default.yaml")).detectors.actuator
        assert 0.0 < config.response_fraction < 1.0
