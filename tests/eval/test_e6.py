"""E6's scoring, checked against scripted models that are right and wrong.

The real run needs a model server. What is tested here is that the three
verdicts mean what they say: a model that is right on every count scores
right, and each way of being wrong is caught by the verdict that names it and
by no other.
"""

import json
from pathlib import Path

import pytest

from eval.experiments import e6_tool_selection as e6
from src.common.config import load_config
from src.common.schemas import Intent, TariffBand
from src.common.tools import ASSISTANCE_TOOLS
from tests.reasoning.fakes import ScriptedClient, calls, text

READS = (
    ("get_thermal_state", {}),
    ("get_occupancy", {}),
    ("get_tariff_state", {}),
    ("get_active_faults", {}),
)


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


def _case(config, occupied=True, tariff=TariffBand.NORMAL, fault=False):
    return next(
        case
        for case in e6.supervisor_cases(config)
        if case.occupied is occupied and case.tariff is tariff and case.fault is fault
    )


def _propose(setpoint, mode="NORMAL"):
    return calls(
        ("propose_setpoint", {"setpoint_c": setpoint, "mode": mode, "rationale": "policy"})
    )


def _hint(intent="service", subject="calendar"):
    return text(
        json.dumps(
            {
                "intent": intent,
                "subject": subject,
                "comfort": "unchanged",
                "target_c": None,
                "rationale": "x",
                "spoken_reply": "ok",
            }
        )
    )


class TestTheScenarioSet:
    def test_there_are_about_fifty(self, config):
        """Section 8.3: ~50 hand-built scenarios."""
        total = len(e6.supervisor_cases(config)) + len(e6.ASSIST_CASES)
        assert 45 <= total <= 60

    def test_every_expected_tool_is_declared(self):
        declared = {spec.name for spec in ASSISTANCE_TOOLS}
        assert {case.tool for case in e6.ASSIST_CASES if case.tool} <= declared

    def test_every_tool_and_no_tool_is_exercised(self):
        expected = {case.tool for case in e6.ASSIST_CASES}
        assert expected == {spec.name for spec in ASSISTANCE_TOOLS} | {None}

    def test_the_bands_follow_the_policy_in_config(self, config):
        """Change the comfort target and what counts as right moves with it."""
        warmer = config.model_copy(
            update={
                "controller": config.controller.model_copy(
                    update={"default_setpoint_c": 25.0}
                )
            }
        )
        assert _case(warmer).low_c == _case(config).low_c + 1.0


class TestSupervisorScoring:
    def test_a_model_that_is_right_scores_right(self, config):
        client = ScriptedClient([calls(*READS), _propose(24.0)])
        score = e6.run_supervisor_case(config, _case(config), client)
        assert (score.schema_valid, score.selected_right, score.plausible) == (True, True, True)

    def test_proposing_without_reading_is_a_selection_miss(self, config):
        client = ScriptedClient([_propose(24.0)])
        score = e6.run_supervisor_case(config, _case(config), client)
        assert score.plausible and not score.selected_right

    def test_an_implausible_setpoint_is_a_plausibility_miss(self, config):
        """27 C for an occupied room at normal tariff: well-formed and wrong."""
        client = ScriptedClient([calls(*READS), _propose(27.0)])
        score = e6.run_supervisor_case(config, _case(config), client)
        assert score.selected_right and score.schema_valid and not score.plausible

    def test_peak_tariff_moves_the_band_up(self, config):
        client = ScriptedClient([calls(*READS), _propose(25.0)])
        score = e6.run_supervisor_case(
            config, _case(config, tariff=TariffBand.PEAK), client
        )
        assert score.plausible

    def test_a_fault_means_hold(self, config):
        client = ScriptedClient([calls(*READS), _propose(25.0, mode="DEGRADED_SENSOR")])
        score = e6.run_supervisor_case(config, _case(config, fault=True), client)
        assert not score.plausible

    def test_a_malformed_call_is_a_schema_miss(self, config):
        client = ScriptedClient([calls(("get_occupancy", "{not json")), calls(*READS), _propose(24.0)])
        score = e6.run_supervisor_case(config, _case(config), client)
        assert not score.schema_valid and score.selected_right

    def test_an_unreachable_server_is_not_counted_as_wrong(self, config):
        client = ScriptedClient([ConnectionError("refused")])
        score = e6.run_supervisor_case(config, _case(config), client)
        assert not score.reached_model
        assert e6.summarise([score]).reached_model == 0


class TestAssistScoring:
    MEETING = e6.ASSIST_CASES[0]

    def test_a_model_that_is_right_scores_right(self, config):
        client = ScriptedClient(
            [
                calls(("schedule_event", {"starts_at": "2025-08-28T15:00:00", "subject": "design review"})),
                _hint(),
            ]
        )
        score = e6.run_assist_case(config, self.MEETING, client)
        assert (score.schema_valid, score.selected_right, score.plausible) == (True, True, True)

    def test_the_wrong_day_is_a_plausibility_miss(self, config):
        client = ScriptedClient(
            [
                calls(("schedule_event", {"starts_at": "2025-08-27T15:00:00", "subject": "design review"})),
                _hint(),
            ]
        )
        score = e6.run_assist_case(config, self.MEETING, client)
        assert score.selected_right and not score.plausible

    def test_the_wrong_tool_is_a_selection_miss(self, config):
        client = ScriptedClient(
            [
                calls(("get_events", {"from_time": "2025-08-28T00:00:00", "to_time": "2025-08-28T23:00:00"})),
                _hint(),
            ]
        )
        assert not e6.run_assist_case(config, self.MEETING, client).selected_right

    def test_a_bad_argument_is_a_schema_miss(self, config):
        client = ScriptedClient(
            [calls(("schedule_event", {"starts_at": "thursday", "subject": "review"})), _hint()]
        )
        assert not e6.run_assist_case(config, self.MEETING, client).schema_valid

    def test_a_request_needing_no_tool_is_right_with_none(self, config):
        case = next(c for c in e6.ASSIST_CASES if c.tool is None and c.intent is Intent.ENVIRONMENT)
        client = ScriptedClient([_hint(intent="environment", subject="temperature")])
        score = e6.run_assist_case(config, case, client)
        assert score.selected_right and score.plausible

    def test_a_tool_where_none_was_needed_is_a_selection_miss(self, config):
        case = next(c for c in e6.ASSIST_CASES if c.tool is None and c.intent is Intent.NONE)
        client = ScriptedClient(
            [calls(("get_events", {"from_time": "2025-08-24T00:00:00", "to_time": "2025-08-24T23:00:00"})), _hint()]
        )
        assert not e6.run_assist_case(config, case, client).selected_right

    def test_a_booking_is_never_made_during_the_experiment(self, config):
        """The recorder answers a commit as the executor would: with a question."""
        case = next(c for c in e6.ASSIST_CASES if c.tool == "book_travel")
        client = ScriptedClient(
            [
                calls(("book_travel", {"kind": "flight", "destination": "Delhi", "origin": "Hyderabad", "depart_on": "2025-08-29T09:00:00"})),
                _hint(subject="booking"),
            ]
        )
        score = e6.run_assist_case(config, case, client)
        assert score.plausible


class TestTheSummary:
    def test_the_three_rates_are_reported_separately(self):
        scores = [
            e6.CaseScore("supervisor", "a", True, True, False, True, 1.0, ""),
            e6.CaseScore("supervisor", "b", True, False, False, True, 1.0, ""),
        ]
        summary = e6.summarise(scores)
        assert (summary.schema_validity, summary.selection_accuracy, summary.argument_plausibility) == (1.0, 0.5, 0.0)

    def test_the_report_says_schema_validity_is_not_a_result(self):
        scores = [e6.CaseScore("supervisor", "a", True, True, True, True, 1.0, "")]
        assert "not a result" in e6.report(scores)
