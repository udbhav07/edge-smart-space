"""Unit tests for the fault injector (FR-31).

The behaviour worth pinning is the refusal: a magnitude that would be ignored,
or missing where it carries the fault's whole content, must not be accepted
silently. Both look exactly like a fault that failed to take effect.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import InjectionCommand
from src.faults.injector import FaultInjector

SUBJECT = "temp_01"
STUCK_VALUE = 27.0


class FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []

    def connect(self, host, port, keepalive):
        pass

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos):
        pass

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def commands(self) -> list[InjectionCommand]:
        return [
            InjectionCommand.model_validate_json(payload)
            for _, payload, _, _ in self.published
        ]


@pytest.fixture(name="wired")
def _wired():
    transport = FakeTransport()
    config = load_config(Path("config/default.yaml"))
    blackboard = Blackboard(config.mqtt, transport)
    return FaultInjector(blackboard, SimClock()), transport


class TestInjecting:
    def test_a_stuck_sensor_is_published_with_its_frozen_value(self, wired):
        injector, transport = wired
        injector.inject(SUBJECT, InjectedFault.STUCK_AT, STUCK_VALUE)
        command = transport.commands()[0]
        assert command.kind is InjectedFault.STUCK_AT
        assert command.magnitude == STUCK_VALUE

    def test_the_command_goes_to_the_subjects_own_topic(self, wired):
        injector, transport = wired
        injector.inject(SUBJECT, InjectedFault.DROPOUT)
        assert transport.published[0][0] == f"space/inject/{SUBJECT}"

    def test_the_command_is_retained(self, wired):
        """So it answers what is injected now, and a restarted adapter resumes
        what the operator asked for."""
        injector, transport = wired
        injector.inject(SUBJECT, InjectedFault.DROPOUT)
        assert transport.published[0][3] is True

    def test_the_published_command_is_returned(self, wired):
        """A caller reports what it asked for, not what it meant to ask for."""
        injector, _ = wired
        command = injector.inject(SUBJECT, InjectedFault.DRIFT, 0.01)
        assert (command.subject, command.magnitude) == (SUBJECT, 0.01)

    def test_the_requester_is_recorded(self, wired):
        injector, transport = wired
        injector.inject(SUBJECT, InjectedFault.DROPOUT)
        assert transport.commands()[0].requester == "operator"

    def test_a_named_requester_is_carried_through(self):
        transport = FakeTransport()
        config = load_config(Path("config/default.yaml"))
        injector = FaultInjector(
            Blackboard(config.mqtt, transport), SimClock(), requester="e4_run"
        )
        injector.inject(SUBJECT, InjectedFault.DROPOUT)
        assert transport.commands()[0].requester == "e4_run"

    def test_injecting_twice_replaces_rather_than_accumulates(self, wired):
        injector, transport = wired
        injector.inject(SUBJECT, InjectedFault.DROPOUT)
        injector.inject(SUBJECT, InjectedFault.STUCK_AT, STUCK_VALUE)
        assert transport.commands()[-1].kind is InjectedFault.STUCK_AT


class TestMagnitudeDiscipline:
    @pytest.mark.parametrize(
        "kind",
        [InjectedFault.STUCK_AT, InjectedFault.OUT_OF_RANGE, InjectedFault.DRIFT],
    )
    def test_a_fault_defined_by_its_magnitude_needs_one(self, wired, kind):
        injector, transport = wired
        with pytest.raises(ValueError):
            injector.inject(SUBJECT, kind)
        assert transport.published == []

    @pytest.mark.parametrize(
        "kind", [InjectedFault.NONE, InjectedFault.DROPOUT]
    )
    def test_a_magnitude_that_would_be_ignored_is_refused(self, wired, kind):
        """Accepted silently, it looks exactly like a fault that failed to
        take effect."""
        injector, transport = wired
        with pytest.raises(ValueError):
            injector.inject(SUBJECT, kind, STUCK_VALUE)
        assert transport.published == []

    def test_a_frozen_value_of_zero_is_a_real_request(self, wired):
        """Zero degrees is a legal frozen value, and refusing it would make
        the sentinel do the wrong job."""
        injector, transport = wired
        injector.inject(SUBJECT, InjectedFault.STUCK_AT, 0.0)
        assert transport.commands()[0].magnitude == 0.0


class TestClearing:
    def test_clearing_publishes_an_injection_of_none(self, wired):
        injector, transport = wired
        injector.clear(SUBJECT)
        assert transport.commands()[0].kind is InjectedFault.NONE

    def test_clearing_carries_no_magnitude(self, wired):
        injector, transport = wired
        injector.clear(SUBJECT)
        assert transport.commands()[0].magnitude == 0.0

    def test_clearing_states_it_rather_than_withdrawing_the_message(self, wired):
        """A withdrawal leaves a restarted adapter nothing to read; a retained
        NONE says plainly that somebody decided nothing is injected."""
        injector, transport = wired
        injector.clear(SUBJECT)
        assert transport.published[0][1] != b""

    def test_a_bad_subject_is_refused_before_publishing(self, wired):
        injector, transport = wired
        with pytest.raises(ValueError):
            injector.inject("a/b", InjectedFault.DROPOUT)
        assert transport.published == []
