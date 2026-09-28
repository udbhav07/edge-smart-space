"""E6's scenarios and their labels; the model itself is not called here."""

from pathlib import Path

from eval.experiments.e6_tool_selection import Scenario, run_scenario, scenarios
from src.common.config import load_config
from src.common.schemas import TariffBand
from tests.reasoning.stubs import ScriptedChat, call, calling

CONFIG = load_config(Path("config/default.yaml"))


def test_there_are_about_fifty_scenarios():
    """Section 8.3 asks for around fifty."""
    assert len(scenarios()) == 48


def test_a_fault_keeps_the_setpoint_in_force():
    assert Scenario("occupied", True, TariffBand.NORMAL, 26.0).expected_c(CONFIG) == 25.0


def test_a_long_vacancy_sets_back():
    assert Scenario("vacant 30 min", False, TariffBand.PEAK, 26.0).expected_c(CONFIG) == CONFIG.supervisor.vacant_setpoint_c


def test_a_short_vacancy_does_not():
    assert Scenario("vacant 2 min", False, TariffBand.NORMAL, 26.0).expected_c(CONFIG) == CONFIG.supervisor.occupied_setpoint_c


def test_a_perfect_cycle_scores_on_every_count():
    chat = ScriptedChat(
        calling(call("get_thermal_state"), call("get_occupancy")),
        calling(call("get_tariff_state"), call("get_active_faults")),
        calling(call("propose_setpoint", setpoint_c=24.0, mode="NORMAL", rationale="occupied")),
    )
    outcome = run_scenario(CONFIG, chat, Scenario("occupied", False, TariffBand.NORMAL, 26.0))
    assert outcome.schema_valid and outcome.tools_right and outcome.argument_right


def test_proposing_without_reading_is_a_tool_selection_failure():
    chat = ScriptedChat(calling(call("propose_setpoint", setpoint_c=24.0, mode="NORMAL", rationale="guess")))
    outcome = run_scenario(CONFIG, chat, Scenario("occupied", False, TariffBand.NORMAL, 26.0))
    assert outcome.argument_right and not outcome.tools_right
