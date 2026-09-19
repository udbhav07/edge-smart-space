"""Unit tests for the fault aggregator.

Three behaviours carry the weight: a fault is raised once and not again, a
clear has to hold before the fault retires, and an UNKNOWN judgment never
retires anything.
"""

import pytest

from src.common.clock import SimClock
from src.common.schemas import DetectorId, Mode
from src.faults.aggregator import FaultAggregator
from src.faults.detectors.base import Finding, Judgment

SUBJECT = "temp_01"
OTHER_SUBJECT = "outdoor_01"
CONFIRM_S = 120.0


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="aggregator")
def _aggregator(clock) -> FaultAggregator:
    return FaultAggregator(clock=clock, clear_confirm_s=CONFIRM_S)


def _finding(
    judgment: Judgment = Judgment.FAULTED,
    detector: DetectorId = DetectorId.D2_STUCK_AT,
    subject: str = SUBJECT,
    confidence: float = 0.94,
) -> Finding:
    return Finding(
        detector=detector,
        subject=subject,
        judgment=judgment,
        confidence=confidence if judgment is Judgment.FAULTED else 0.0,
        evidence={"variance": 0.0002},
    )


class TestRaising:
    def test_a_faulted_finding_raises_a_fault(self, aggregator):
        outcome = aggregator.ingest([_finding()])
        assert len(outcome.raised) == 1
        assert outcome.raised[0].subject == SUBJECT

    def test_the_raised_fault_carries_the_findings_evidence(self, aggregator):
        outcome = aggregator.ingest([_finding()])
        assert outcome.raised[0].evidence["variance"] == 0.0002

    def test_the_same_fault_is_not_raised_twice(self, aggregator, clock):
        aggregator.ingest([_finding()])
        clock.advance(5.0)
        outcome = aggregator.ingest([_finding()])
        assert outcome.raised == ()
        assert len(outcome.active) == 1

    def test_a_clear_finding_raises_nothing(self, aggregator):
        outcome = aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        assert (outcome.raised, outcome.active) == ((), ())

    def test_an_unknown_finding_raises_nothing(self, aggregator):
        outcome = aggregator.ingest([_finding(judgment=Judgment.UNKNOWN)])
        assert (outcome.raised, outcome.active) == ((), ())

    def test_two_detectors_on_one_subject_are_two_faults(self, aggregator):
        outcome = aggregator.ingest(
            [
                _finding(detector=DetectorId.D1_DROPOUT),
                _finding(detector=DetectorId.D2_STUCK_AT),
            ]
        )
        assert len(outcome.raised) == 2

    def test_one_detector_on_two_subjects_is_two_faults(self, aggregator):
        outcome = aggregator.ingest(
            [_finding(subject=SUBJECT), _finding(subject=OTHER_SUBJECT)]
        )
        assert len(outcome.raised) == 2

    def test_nothing_reported_changes_nothing(self, aggregator):
        aggregator.ingest([_finding()])
        outcome = aggregator.ingest([])
        assert not outcome.changed
        assert len(outcome.active) == 1


class TestClearConfirmation:
    def test_a_clear_does_not_retire_the_fault_at_once(self, aggregator):
        aggregator.ingest([_finding()])
        outcome = aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        assert outcome.cleared == ()
        assert len(outcome.active) == 1

    def test_the_fault_retires_once_the_clear_has_held(self, aggregator, clock):
        aggregator.ingest([_finding()])
        aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        clock.advance(CONFIRM_S)
        outcome = aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        assert len(outcome.cleared) == 1
        assert outcome.active == ()

    def test_a_clear_just_short_of_the_period_does_not_retire_it(
        self, aggregator, clock
    ):
        aggregator.ingest([_finding()])
        aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        clock.advance(CONFIRM_S - 0.001)
        assert aggregator.ingest([_finding(judgment=Judgment.CLEAR)]).cleared == ()

    def test_faulting_again_restarts_the_countdown(self, aggregator, clock):
        """A detector that flaps produces one fault, not a storm of raise and
        clear pairs each minting a new id."""
        aggregator.ingest([_finding()])
        aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        clock.advance(CONFIRM_S - 1.0)
        aggregator.ingest([_finding()])

        aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        clock.advance(CONFIRM_S - 1.0)
        outcome = aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        assert outcome.cleared == ()
        assert len(outcome.active) == 1

    def test_the_retired_fault_keeps_the_id_it_was_raised_under(
        self, aggregator, clock
    ):
        raised = aggregator.ingest([_finding()]).raised[0]
        aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        clock.advance(CONFIRM_S)
        cleared = aggregator.ingest([_finding(judgment=Judgment.CLEAR)]).cleared[0]
        assert cleared.fault_id == raised.fault_id

    def test_a_zero_confirmation_period_retires_at_once(self, clock):
        """Configurable includes configurably immediate, which is what an
        accelerated experiment run wants."""
        aggregator = FaultAggregator(clock=clock, clear_confirm_s=0.0)
        aggregator.ingest([_finding()])
        outcome = aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        assert len(outcome.cleared) == 1

    def test_a_negative_confirmation_period_is_refused(self, clock):
        with pytest.raises(ValueError):
            FaultAggregator(clock=clock, clear_confirm_s=-1.0)


class TestUnknownNeverClears:
    def test_an_unknown_judgment_does_not_start_the_countdown(
        self, aggregator, clock
    ):
        """A stuck sensor that then goes silent is more broken, not less: D2's
        window empties and it stops being able to say anything."""
        aggregator.ingest([_finding()])
        aggregator.ingest([_finding(judgment=Judgment.UNKNOWN)])
        clock.advance(CONFIRM_S * 10)
        outcome = aggregator.ingest([_finding(judgment=Judgment.UNKNOWN)])
        assert outcome.cleared == ()
        assert len(outcome.active) == 1

    def test_a_clear_after_unknown_still_has_to_hold(self, aggregator, clock):
        aggregator.ingest([_finding()])
        clock.advance(CONFIRM_S * 2)
        aggregator.ingest([_finding(judgment=Judgment.UNKNOWN)])
        outcome = aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        assert outcome.cleared == ()

        clock.advance(CONFIRM_S)
        assert len(aggregator.ingest([_finding(judgment=Judgment.CLEAR)]).cleared) == 1


class TestPriority:
    def test_nothing_is_dominant_when_nothing_is_wrong(self, aggregator):
        assert aggregator.dominant is None

    def test_an_actuator_fault_outranks_a_sensor_fault(self, aggregator):
        """An actuator fault stops actuation; a sensor fault degrades onto
        prediction."""
        aggregator.ingest(
            [
                _finding(detector=DetectorId.D1_DROPOUT),
                _finding(
                    detector=DetectorId.D5_ACTUATOR_NO_RESPONSE, subject="ac"
                ),
            ]
        )
        assert aggregator.dominant.mode_impact is Mode.DEGRADED_ACTUATOR

    def test_model_divergence_outranks_an_actuator_fault(self, aggregator):
        """There is no prediction left to degrade onto."""
        aggregator.ingest(
            [
                _finding(
                    detector=DetectorId.D5_ACTUATOR_NO_RESPONSE, subject="ac"
                ),
                _finding(detector=DetectorId.MODEL_DIVERGENCE, subject="thermal"),
            ]
        )
        assert aggregator.dominant.mode_impact is Mode.SAFE_HOLD

    def test_the_earlier_fault_wins_a_tie(self, aggregator, clock):
        """The earliest fault is the likelier root cause of the later ones."""
        first = aggregator.ingest(
            [_finding(detector=DetectorId.D2_STUCK_AT)]
        ).raised[0]
        clock.advance(60.0)
        aggregator.ingest([_finding(detector=DetectorId.D1_DROPOUT)])
        assert aggregator.dominant.fault_id == first.fault_id

    def test_active_faults_are_reported_most_severe_first(self, aggregator):
        aggregator.ingest(
            [
                _finding(detector=DetectorId.D1_DROPOUT),
                _finding(
                    detector=DetectorId.D5_ACTUATOR_NO_RESPONSE, subject="ac"
                ),
            ]
        )
        impacts = [event.mode_impact for event in aggregator.active]
        assert impacts == [Mode.DEGRADED_ACTUATOR, Mode.DEGRADED_SENSOR]

    def test_retiring_the_dominant_fault_promotes_the_next_one(
        self, aggregator, clock
    ):
        aggregator.ingest(
            [
                _finding(detector=DetectorId.D1_DROPOUT),
                _finding(
                    detector=DetectorId.D5_ACTUATOR_NO_RESPONSE, subject="ac"
                ),
            ]
        )
        aggregator.ingest(
            [
                _finding(
                    judgment=Judgment.CLEAR,
                    detector=DetectorId.D5_ACTUATOR_NO_RESPONSE,
                    subject="ac",
                )
            ]
        )
        clock.advance(CONFIRM_S)
        aggregator.ingest(
            [
                _finding(
                    judgment=Judgment.CLEAR,
                    detector=DetectorId.D5_ACTUATOR_NO_RESPONSE,
                    subject="ac",
                )
            ]
        )
        assert aggregator.dominant.detector is DetectorId.D1_DROPOUT


class TestOutcome:
    def test_a_raise_counts_as_a_change_worth_publishing(self, aggregator):
        assert aggregator.ingest([_finding()]).changed

    def test_a_steady_fault_is_not_a_change(self, aggregator, clock):
        aggregator.ingest([_finding()])
        clock.advance(5.0)
        assert not aggregator.ingest([_finding()]).changed

    def test_a_retirement_counts_as_a_change(self, aggregator, clock):
        aggregator.ingest([_finding()])
        aggregator.ingest([_finding(judgment=Judgment.CLEAR)])
        clock.advance(CONFIRM_S)
        assert aggregator.ingest([_finding(judgment=Judgment.CLEAR)]).changed

    def test_the_outcome_reports_what_stands_afterwards(self, aggregator):
        outcome = aggregator.ingest([_finding()])
        assert outcome.active == aggregator.active

    def test_the_aggregator_decides_no_mode_of_its_own(self, aggregator):
        """Section 5.6 escalates on several faults at once, which is a judgment
        about the system rather than about any one fault. That belongs to the
        mode manager; each event carries only what it implies alone."""
        aggregator.ingest(
            [
                _finding(detector=DetectorId.D1_DROPOUT),
                _finding(detector=DetectorId.D2_STUCK_AT),
            ]
        )
        assert [event.mode_impact for event in aggregator.active] == [
            Mode.DEGRADED_SENSOR,
            Mode.DEGRADED_SENSOR,
        ]
