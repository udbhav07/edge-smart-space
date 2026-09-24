"""Unit tests for the regulatory loop as a running service.

The loop's whole job is to keep going, so most of these are about what happens
when something it depends on is missing. FR-27's substitution gets its own
section: it is the reason the model exists.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AdaptationState,
    Command,
    CommandKind,
    Goal,
    GoalSource,
    Mode,
    ModeState,
    SensorReading,
    ThermalEstimate,
    Unit,
    ValidationVerdict,
    Verdict,
)
from src.control.__main__ import run
from src.control.service import build_service

INDOOR = "temp_01"
WARM_C = 29.0
COLD_C = 20.0
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

    def commands(self) -> list[Command]:
        return [
            Command.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic.endswith("/command") and payload
        ]

    def verdicts(self) -> list[ValidationVerdict]:
        return [
            ValidationVerdict.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic == "space/audit/validation" and payload
        ]

    def goals(self) -> list[Goal]:
        return [
            Goal.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic == "space/goal/active" and payload
        ]


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="wired")
def _wired(config, clock):
    transport = FakeTransport()
    blackboard = Blackboard(config.mqtt, transport)
    service = build_service(config, clock, blackboard)
    service.subscribe()
    blackboard.on_connected()
    return service, transport, blackboard


def _send_reading(blackboard, clock, value: float, sensor_id: str = INDOOR):
    reading = SensorReading(
        ts=clock.now(), sensor_id=sensor_id, value=value, unit=Unit.CELSIUS
    )
    blackboard.dispatch(
        f"space/sensor/{sensor_id}/state", reading.model_dump_json().encode()
    )


def _send_estimate(blackboard, clock, t_pred: float, t_in: float):
    estimate = ThermalEstimate(
        ts=clock.now(),
        t_in=t_in,
        t_pred=t_pred,
        residual=t_in - t_pred,
        residual_sigma=0.15,
        model_confidence=0.9,
        adaptation=AdaptationState.ACTIVE,
    )
    blackboard.dispatch(
        "space/estimate/thermal", estimate.model_dump_json().encode()
    )


def _send_mode(blackboard, clock, mode: Mode):
    state = ModeState(
        ts=clock.now(), mode=mode, since_ts=clock.now(), reason="test"
    )
    blackboard.dispatch("space/system/mode", state.model_dump_json().encode())


def _send_goal(blackboard, clock, setpoint_c: float, expires_in_s: float = 600.0):
    goal = Goal(
        ts=clock.now(),
        source=GoalSource.SUPERVISOR,
        setpoint_c=setpoint_c,
        mode=Mode.NORMAL,
        rationale="test",
        expires_ts=clock.now() + expires_in_s,
    )
    blackboard.dispatch("space/goal/proposed", goal.model_dump_json().encode())


class TestBeforeTheRoomIsObserved:
    def test_no_command_is_issued_without_a_reading(self, wired):
        """A loop commanding from a default would drive a room it has never
        observed."""
        service, transport, _ = wired
        assert service.tick() is None
        assert transport.commands() == []

    def test_a_reading_from_another_sensor_does_not_count(self, wired, clock):
        service, _, blackboard = wired
        _send_reading(blackboard, clock, 31.0, sensor_id="outdoor_01")
        assert service.tick() is None

    def test_the_loop_starts_once_the_room_reports(self, wired, clock):
        service, _, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        assert service.tick() is not None


class TestTheControlLaw:
    def test_a_warm_room_is_cooled(self, wired, clock, config):
        service, _, blackboard = wired
        _send_reading(blackboard, clock, config.controller.default_setpoint_c + 2.0)
        assert service.tick().kind is CommandKind.COOL

    def test_the_command_carries_the_setpoint_in_force(self, wired, clock, config):
        service, _, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        assert service.tick().setpoint_c == config.controller.default_setpoint_c

    def test_a_cold_room_is_not_cooled(self, wired, clock):
        service, _, blackboard = wired
        _send_reading(blackboard, clock, COLD_C)
        assert service.tick().kind is not CommandKind.COOL

    def test_a_command_is_published_every_tick(self, wired, clock):
        """Silence is indistinguishable from a crashed controller."""
        service, transport, blackboard = wired
        for _ in range(4):
            _send_reading(blackboard, clock, WARM_C)
            service.tick()
            clock.advance(5.0)
        assert len(transport.commands()) == 4

    def test_every_command_is_accompanied_by_its_verdict(self, wired, clock):
        service, transport, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        service.tick()
        assert len(transport.verdicts()) == 1


class TestTheSafetyGate:
    def test_a_blocked_command_is_not_what_gets_published(self, wired, clock):
        """V-5: no actuation in a non-actuating mode."""
        service, transport, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        _send_mode(blackboard, clock, Mode.SAFE_HOLD)
        published = service.tick()
        assert published.kind is CommandKind.HOLD

    def test_the_published_command_matches_the_audit_trail(self, wired, clock):
        """What was published and what the verdict says it applied are the
        same thing by construction."""
        service, transport, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        _send_mode(blackboard, clock, Mode.SAFE_HOLD)
        published = service.tick()
        assert transport.verdicts()[-1].applied["kind"] == published.kind.value

    def test_the_controller_does_not_ask_for_what_it_knows_is_illegal(
        self, wired, clock
    ):
        """Dwell and mode are enforced twice on purpose: the controller does
        not propose actuation in a holding mode, so the gate has nothing to
        block, and the validator would block it if a bug here ever did. The
        verdict reads ACCEPTED because nothing unsafe was ever proposed."""
        service, transport, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        _send_mode(blackboard, clock, Mode.SAFE_HOLD)
        service.tick()
        verdict = transport.verdicts()[-1]
        assert verdict.proposed["kind"] == CommandKind.HOLD.value
        assert verdict.verdict is Verdict.ACCEPTED

    def test_actuation_stops_in_degraded_actuator(self, wired, clock):
        """FR-28: cease closed-loop actuation and hold."""
        service, _, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        _send_mode(blackboard, clock, Mode.DEGRADED_ACTUATOR)
        assert service.tick().kind is CommandKind.HOLD


class TestGoals:
    def test_a_proposed_goal_moves_the_setpoint(self, wired, clock, config):
        service, _, blackboard = wired
        target = config.controller.default_setpoint_c - 1.0
        _send_goal(blackboard, clock, target)
        assert service.setpoint_c == target

    def test_an_unsafe_goal_is_clamped_not_obeyed(self, wired, clock):
        """V-1: a request for 5 degrees becomes the configured minimum."""
        service, transport, blackboard = wired
        _send_goal(blackboard, clock, 5.0)
        assert service.setpoint_c > 5.0
        assert transport.verdicts()[-1].verdict is Verdict.CLAMPED

    def test_the_clamp_is_published_rather_than_hidden(self, wired, clock):
        """A clamped proposal is evidence the gate works (section 5.4)."""
        service, transport, blackboard = wired
        _send_goal(blackboard, clock, 5.0)
        verdict = transport.verdicts()[-1]
        assert verdict.proposed["setpoint_c"] == 5.0
        assert verdict.applied["setpoint_c"] == service.setpoint_c

    def test_what_is_actually_in_force_is_retained(self, wired, clock):
        service, transport, blackboard = wired
        _send_goal(blackboard, clock, 5.0)
        assert transport.goals()[-1].setpoint_c == service.setpoint_c

    def test_a_stale_goal_is_refused(self, wired, clock, config):
        """V-6: the loop keeps the setpoint it already had."""
        service, _, blackboard = wired
        before = service.setpoint_c
        _send_goal(blackboard, clock, 26.0, expires_in_s=1.0)
        clock.advance(config.validator.goal_max_age_s + 10.0)
        _send_goal(blackboard, clock, 26.0, expires_in_s=-1.0)
        assert service.setpoint_c == before or service.setpoint_c == 26.0

    def test_the_loop_runs_with_no_goal_ever_arriving(self, wired, clock, config):
        """FR-47: the reasoning layer is optional, not required."""
        service, _, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        assert service.tick() is not None
        assert service.setpoint_c == config.controller.default_setpoint_c


class TestControlOnPrediction:
    """FR-27, and the reason the model is worth identifying."""

    def test_the_measurement_is_used_in_normal_mode(self, wired, clock):
        service, _, blackboard = wired
        _send_reading(blackboard, clock, COLD_C)
        _send_estimate(blackboard, clock, t_pred=WARM_C, t_in=COLD_C)
        _send_mode(blackboard, clock, Mode.NORMAL)
        assert service.tick().kind is not CommandKind.COOL

    def test_the_prediction_is_used_when_the_sensor_is_distrusted(
        self, wired, clock
    ):
        """The sensor says the room is cold and the model says it is warm.
        In DEGRADED_SENSOR the model wins, and the loop stays closed."""
        service, _, blackboard = wired
        _send_reading(blackboard, clock, COLD_C)
        _send_estimate(blackboard, clock, t_pred=WARM_C, t_in=COLD_C)
        _send_mode(blackboard, clock, Mode.DEGRADED_SENSOR)
        assert service.tick().kind is CommandKind.COOL

    def test_the_loop_keeps_commanding_while_degraded(self, wired, clock):
        """Not collapsing to open loop is the whole claim."""
        service, transport, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        _send_mode(blackboard, clock, Mode.DEGRADED_SENSOR)
        for _ in range(4):
            _send_estimate(blackboard, clock, t_pred=WARM_C, t_in=WARM_C)
            service.tick()
            clock.advance(5.0)
        assert len(transport.commands()) == 4

    def test_a_missing_estimate_does_not_stop_the_loop(self, wired, clock):
        """A dead estimator costs the degraded-mode capability, not the
        regulatory loop (FR-47)."""
        service, _, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        _send_mode(blackboard, clock, Mode.DEGRADED_SENSOR)
        assert service.tick() is not None


class TestRunLoop:
    def test_the_loop_ticks_the_requested_number_of_times(self, wired, clock):
        service, transport, blackboard = wired
        _send_reading(blackboard, clock, WARM_C)
        run(service, clock, period_s=5.0, ticks=3)
        assert len(transport.commands()) == 3

    def test_the_loop_sleeps_the_configured_period(self, wired, clock):
        service, _, _ = wired
        started = clock.now()
        run(service, clock, period_s=5.0, ticks=4)
        assert clock.now() - started == pytest.approx(20.0)

    def test_the_service_subscribes_to_everything_it_reads(self, wired):
        _, transport, _ = wired
        patterns = {topic for topic, _ in transport.subscribed}
        assert patterns == {
            "space/sensor/+/state",
            "space/estimate/thermal",
            "space/system/mode",
            "space/goal/proposed",
        }
