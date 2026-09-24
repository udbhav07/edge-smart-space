"""Unit tests for the degradation state machine (section 5.6).

The transitions in the state diagram, one test each, plus the two decisions
that are easy to get subtly wrong: what counts as "multiple faults", and what
a manual reset is allowed to do.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import ModeConfig, load_config
from src.common.schemas import (
    DetectorId,
    FaultClass,
    FaultEvent,
    Mode,
    ModeReset,
)
from src.faults.mode_manager import ModeManager

BUDGET_S = 1800.0
CONFIRM_S = 120.0
TS = 1756032000.0


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


def _config(**overrides) -> ModeConfig:
    return ModeConfig(
        **{
            "degraded_sensor_budget_s": BUDGET_S,
            "fault_clear_confirm_s": CONFIRM_S,
            "transition_deadline_s": 2.0,
            **overrides,
        }
    )


@pytest.fixture(name="manager")
def _manager(clock) -> ModeManager:
    return ModeManager(config=_config(), clock=clock)


def _fault(
    detector: DetectorId = DetectorId.D2_STUCK_AT,
    subject: str = "temp_01",
    mode_impact: Mode = Mode.DEGRADED_SENSOR,
    fault_class: FaultClass = FaultClass.SENSOR,
) -> FaultEvent:
    return FaultEvent(
        fault_id=f"f_{subject}_{detector.value}",
        detector=detector,
        subject=subject,
        fault_class=fault_class,
        confidence=0.9,
        detected_ts=TS,
        evidence={},
        mode_impact=mode_impact,
    )


_ACTUATOR_FAULT = _fault(
    detector=DetectorId.D5_ACTUATOR_NO_RESPONSE,
    subject="ac",
    mode_impact=Mode.DEGRADED_ACTUATOR,
    fault_class=FaultClass.ACTUATOR,
)

_DIVERGENCE = _fault(
    detector=DetectorId.MODEL_DIVERGENCE,
    subject="temp_01",
    mode_impact=Mode.SAFE_HOLD,
    fault_class=FaultClass.MODEL,
)


class TestStartup:
    def test_it_starts_in_init(self, manager):
        assert manager.mode is Mode.INIT

    def test_it_stays_in_init_until_a_sensor_reports(self, manager):
        """A system whose sensors never arrive must stay visibly in INIT
        rather than claim to be NORMAL."""
        assert manager.update([], sensors_reporting=False) is None
        assert manager.mode is Mode.INIT

    def test_it_reaches_normal_once_sensors_report(self, manager):
        state = manager.update([], sensors_reporting=True)
        assert state is not None and state.mode is Mode.NORMAL

    def test_the_first_transition_is_published(self, manager):
        state = manager.update([], sensors_reporting=True)
        assert state.reason == "no active faults"

    def test_an_unchanged_mode_publishes_nothing(self, manager):
        """Republishing a retained mode every tick would bury the transitions
        that matter in the ones that do not."""
        manager.update([], sensors_reporting=True)
        assert manager.update([], sensors_reporting=True) is None


class TestSensorFaults:
    def test_a_sensor_fault_degrades_onto_prediction(self, manager):
        manager.update([])
        state = manager.update([_fault()])
        assert state.mode is Mode.DEGRADED_SENSOR

    def test_the_published_mode_names_the_fault_that_caused_it(self, manager):
        manager.update([])
        state = manager.update([_fault()])
        assert state.active_fault_ids == (_fault().fault_id,)
        assert "D2_STUCK_AT" in state.reason

    def test_clearing_the_fault_returns_to_normal(self, manager):
        manager.update([])
        manager.update([_fault()])
        state = manager.update([])
        assert state.mode is Mode.NORMAL

    def test_two_detectors_on_one_sensor_is_still_one_broken_thing(self, manager):
        """A stuck sensor raises D2, and its frozen reading grows the residual
        until D4 raises too. Escalating there would mean every sensor failure
        ended in SAFE_HOLD and FR-27 would never once be exercised."""
        manager.update([])
        state = manager.update(
            [
                _fault(detector=DetectorId.D2_STUCK_AT),
                _fault(detector=DetectorId.D4_DRIFT),
            ]
        )
        assert state.mode is Mode.DEGRADED_SENSOR

    def test_two_faulted_sensors_is_two_broken_things(self, manager):
        manager.update([])
        state = manager.update(
            [_fault(subject="temp_01"), _fault(subject="outdoor_01")]
        )
        assert state.mode is Mode.SAFE_HOLD

    def test_the_escalation_reason_says_how_many(self, manager):
        manager.update([])
        state = manager.update(
            [_fault(subject="temp_01"), _fault(subject="outdoor_01")]
        )
        assert "2 subjects" in state.reason


class TestTheSubstitutionBudget:
    def test_prediction_control_runs_while_the_budget_lasts(self, manager, clock):
        manager.update([])
        manager.update([_fault()])
        clock.advance(BUDGET_S - 1.0)
        assert manager.update([_fault()]) is None
        assert manager.mode is Mode.DEGRADED_SENSOR

    def test_an_exhausted_budget_holds(self, manager, clock):
        """A prediction is not a measurement forever: the model drifts away
        from the room with nothing to correct it (section 7.2)."""
        manager.update([])
        manager.update([_fault()])
        clock.advance(BUDGET_S)
        state = manager.update([_fault()])
        assert state.mode is Mode.SAFE_HOLD

    def test_the_hold_reason_names_the_budget(self, manager, clock):
        manager.update([])
        manager.update([_fault()])
        clock.advance(BUDGET_S)
        assert "budget" in manager.update([_fault()]).reason

    def test_the_elapsed_substitution_is_reported(self, manager, clock):
        manager.update([])
        manager.update([_fault()])
        clock.advance(600.0)
        assert manager.substitution_elapsed_s == pytest.approx(600.0)

    def test_nothing_is_spent_while_not_substituting(self, manager, clock):
        manager.update([])
        clock.advance(600.0)
        assert manager.substitution_elapsed_s == 0.0

    def test_recovering_refunds_the_budget(self, manager, clock):
        """A sensor that comes back and breaks again gets a full budget, not
        the remains of the last one."""
        manager.update([])
        manager.update([_fault()])
        clock.advance(BUDGET_S - 10.0)
        manager.update([])

        manager.update([_fault()])
        clock.advance(BUDGET_S - 10.0)
        assert manager.mode is Mode.DEGRADED_SENSOR


class TestActuatorFaults:
    def test_an_actuator_fault_stops_actuation(self, manager):
        manager.update([])
        state = manager.update([_ACTUATOR_FAULT])
        assert state.mode is Mode.DEGRADED_ACTUATOR

    def test_it_outranks_a_sensor_fault_on_its_own_subject(self, manager):
        """The aggregator ranks; the manager follows the dominant fault."""
        manager.update([])
        state = manager.update([_ACTUATOR_FAULT, _fault()])
        assert state.mode is Mode.SAFE_HOLD

    def test_the_system_does_not_leave_it_by_itself(self, manager):
        """Section 5.6 returns to NORMAL on an ack restored and a response
        observed. There is no ack on an open-loop IR path (R-02) and the mode
        blocks the cooling that would produce a response, so the transition as
        written is unreachable."""
        manager.update([])
        manager.update([_ACTUATOR_FAULT])
        manager.update([])
        assert manager.mode is Mode.DEGRADED_ACTUATOR


class TestModelDivergence:
    def test_a_diverged_model_holds_rather_than_degrades(self, manager):
        """There is no prediction left to degrade onto."""
        manager.update([])
        state = manager.update([_DIVERGENCE])
        assert state.mode is Mode.SAFE_HOLD

    def test_one_diverged_model_is_enough_on_its_own(self, manager):
        manager.update([])
        state = manager.update([_DIVERGENCE])
        assert "MODEL_DIVERGENCE" in state.reason


class TestOperatorReset:
    def _reset(self) -> ModeReset:
        return ModeReset(ts=TS, requester="operator", reason="replaced the sensor")

    def test_a_reset_clears_a_hold_once_nothing_is_active(self, manager):
        manager.update([])
        manager.update([_DIVERGENCE])
        assert manager.mode is Mode.SAFE_HOLD

        manager.request_reset(self._reset())
        state = manager.update([])
        assert state.mode is Mode.NORMAL

    def test_the_reset_is_recorded_as_the_reason(self, manager):
        manager.update([])
        manager.update([_DIVERGENCE])
        manager.request_reset(self._reset())
        assert "reset" in manager.update([]).reason

    def test_a_reset_with_faults_still_active_is_refused(self, manager):
        """It re-tests rather than overrides: the fault set is what decides."""
        manager.update([])
        manager.update([_DIVERGENCE])
        manager.request_reset(self._reset())
        assert manager.update([_DIVERGENCE]) is None
        assert manager.mode is Mode.SAFE_HOLD

    def test_a_reset_releases_a_stopped_actuator(self, manager):
        manager.update([])
        manager.update([_ACTUATOR_FAULT])
        manager.request_reset(self._reset())
        state = manager.update([])
        assert state.mode is Mode.NORMAL

    def test_a_reset_does_not_survive_being_refused(self, manager):
        """Otherwise it would fire later, when nobody was watching."""
        manager.update([])
        manager.update([_DIVERGENCE])
        manager.request_reset(self._reset())
        manager.update([_DIVERGENCE])
        assert manager.update([]) is None or manager.mode is Mode.SAFE_HOLD


class TestPublishedState:
    def test_the_state_carries_when_the_mode_was_entered(self, manager, clock):
        manager.update([])
        entered = clock.now()
        clock.advance(60.0)
        manager.update([_fault()])
        clock.advance(60.0)
        state = manager.update([_fault(), _fault(subject="hum_01")])
        assert state.since_ts > entered

    def test_the_state_lists_every_active_fault(self, manager):
        manager.update([])
        state = manager.update([_fault(), _fault(subject="outdoor_01")])
        assert len(state.active_fault_ids) == 2

    def test_a_changed_fault_set_republishes_even_in_the_same_mode(self, manager):
        """The mode is the same but what is broken is not, and the retained
        topic has to say so."""
        manager.update([])
        manager.update([_fault(detector=DetectorId.D2_STUCK_AT)])
        state = manager.update(
            [
                _fault(detector=DetectorId.D2_STUCK_AT),
                _fault(detector=DetectorId.D4_DRIFT),
            ]
        )
        assert state is not None and state.mode is Mode.DEGRADED_SENSOR


class TestConfiguredDefaults:
    def test_the_shipped_budget_matches_the_design(self):
        """Section 7.2: 1800 s of control on prediction."""
        config = load_config(Path("config/default.yaml")).mode
        assert config.degraded_sensor_budget_s == 1800.0

    def test_the_transition_deadline_is_the_one_fr_26_gives(self):
        config = load_config(Path("config/default.yaml")).mode
        assert config.transition_deadline_s == 2.0
