"""Unit tests for checking a classification against the words (FR-44, FR-53).

Every case is one the user test (docs/user-testing/2026-10-08-findings.md)
saw the served model get wrong.
"""

import pytest

from src.common.schemas import Comfort, ContextTurn, Intent, PreferenceHint
from src.reasoning.understanding import memory_request, understood


def _hint(intent=Intent.ENVIRONMENT, comfort=Comfort.UNCHANGED, target_c=None, subject=""):
    return PreferenceHint(ts=1.0, intent=intent, comfort=comfort, target_c=target_c, subject=subject)


CALENDAR_TURN = ContextTurn(
    ts=1.0, said="put a meeting with Ravi on Thursday at 3", replied="Added Meeting with Ravi at 15:00."
)
TEMPERATURE_TURN = ContextTurn(ts=1.0, said="make it 22 degrees", replied="Done. The room is now set to 22 degrees.")


class TestMemory:
    """U-07: memory requests were dropped as asking for nothing."""

    @pytest.mark.parametrize(
        "said",
        [
            "please remember that I am vegetarian",
            "remember that I like tea",
            "remember I like meetings after 10am",
            "forget that I am vegetarian",
            "forget what I said about meetings",
            "don't forget my mother's birthday",
        ],
    )
    def test_a_memory_request_is_a_service_request(self, said):
        hint = understood(_hint(Intent.NONE), said)
        assert hint.intent is Intent.SERVICE and hint.subject == "memory"

    @pytest.mark.parametrize("said", ["what do you know about me", "what have you remembered", "do you remember anything"])
    def test_asking_what_is_remembered_is_a_question(self, said):
        assert understood(_hint(Intent.NONE), said).intent is Intent.QUESTION

    def test_a_memory_request_carries_no_temperature(self):
        """'I prefer the room at 23, remember that' is kept, not acted on."""
        hint = understood(_hint(target_c=23.0), "I prefer the room at 23 degrees, remember that")
        assert hint.target_c is None and hint.comfort is Comfort.UNCHANGED


class TestQuestions:
    """U-06, U-08, U-25: questions and greetings are answered, never acted on."""

    @pytest.mark.parametrize(
        "said",
        ["why is it so warm", "what's the temperature in here", "is everything working ok", "how warm is it"],
    )
    def test_a_question_is_a_question(self, said):
        hint = understood(_hint(comfort=Comfort.COOLER), said)
        assert hint.intent is Intent.QUESTION and hint.comfort is Comfort.UNCHANGED

    def test_a_question_never_moves_the_setpoint(self):
        hint = understood(_hint(target_c=21.0, comfort=Comfort.COOLER), "why is it so warm")
        assert hint.target_c is None

    @pytest.mark.parametrize("said", ["hello", "hi there", "good morning", "thank you", "how are you?"])
    def test_a_greeting_is_answered(self, said):
        assert understood(_hint(Intent.NONE), said).intent is Intent.QUESTION

    def test_a_request_phrased_as_a_question_stays_a_request(self):
        hint = understood(_hint(comfort=Comfort.COOLER), "can you make it cooler")
        assert hint.intent is Intent.ENVIRONMENT and hint.comfort is Comfort.COOLER

    def test_how_about_is_a_suggestion_not_a_question(self):
        hint = understood(_hint(target_c=22.0), "how about 22 degrees")
        assert hint.intent is Intent.ENVIRONMENT

    def test_a_preference_with_nothing_in_it_is_a_remark(self):
        assert understood(_hint(), "the room").intent is Intent.QUESTION


class TestComfortWords:
    """U-08: 'stuffy' warmed the room."""

    @pytest.mark.parametrize("said", ["it's a bit stuffy in here", "so muggy today", "I'm sweating", "it's too warm"])
    def test_words_for_too_warm_mean_cooler(self, said):
        assert understood(_hint(comfort=Comfort.WARMER), said).comfort is Comfort.COOLER

    @pytest.mark.parametrize("said", ["I'm freezing", "it's chilly", "too cold in here"])
    def test_words_for_too_cold_mean_warmer(self, said):
        assert understood(_hint(comfort=Comfort.COOLER), said).comfort is Comfort.WARMER

    def test_a_model_answer_with_no_comfort_word_stands(self):
        assert understood(_hint(comfort=Comfort.WARMER), "warm it up a little").comfort is Comfort.WARMER


class TestNamedTemperature:
    """U-16: 'make it 40' was labelled cooler."""

    def test_a_named_temperature_carries_no_invented_direction(self):
        hint = understood(_hint(target_c=40.0, comfort=Comfort.COOLER), "make it 40 degrees")
        assert hint.comfort is Comfort.UNCHANGED and hint.target_c == 40.0

    def test_a_named_temperature_keeps_a_direction_that_was_said(self):
        hint = understood(_hint(target_c=22.0, comfort=Comfort.WARMER), "too warm, make it 22 degrees")
        assert hint.comfort is Comfort.COOLER


class TestFollowUps:
    """U-01: 'move it to 4' after a meeting set the room to 4 C."""

    def test_a_bare_number_after_a_calendar_exchange_is_a_follow_up(self):
        hint = understood(_hint(target_c=4.0), "move it to 4", recent=(CALENDAR_TURN,))
        assert hint.intent is Intent.SERVICE and hint.target_c is None

    def test_a_bare_number_after_a_temperature_exchange_is_a_temperature(self):
        hint = understood(_hint(target_c=21.0), "make it 21", recent=(TEMPERATURE_TURN,))
        assert hint.intent is Intent.ENVIRONMENT and hint.target_c == 21.0

    def test_a_bare_number_with_no_conversation_is_a_temperature(self):
        """The README's own 'make it 22' must still work on its own."""
        assert understood(_hint(target_c=22.0), "make it 22").target_c == 22.0

    def test_a_temperature_named_as_one_is_never_a_follow_up(self):
        hint = understood(_hint(target_c=22.0), "make it 22 degrees", recent=(CALENDAR_TURN,))
        assert hint.intent is Intent.ENVIRONMENT


class TestNothingIsInvented:
    def test_a_service_request_passes_unchanged(self):
        hint = _hint(Intent.SERVICE, subject="calendar")
        assert understood(hint, "put a meeting on Thursday at 3") == hint

    def test_an_empty_transcript_changes_nothing(self):
        hint = _hint(comfort=Comfort.COOLER)
        assert understood(hint, "  ") == hint


class TestMemoryRequest:
    """U-07: which memory tool the words plainly ask for."""

    @pytest.mark.parametrize(
        "said, expected",
        [
            ("please remember that I am vegetarian", ("remember", "Is vegetarian.")),
            ("remember I like meetings after 10am", ("remember", "Likes meetings after 10am.")),
            ("I prefer the room at 23 degrees, remember that", ("remember", "Prefers the room at 23 degrees.")),
            ("don't forget my birthday is in May", ("remember", "Their birthday is in May.")),
            ("forget that I am vegetarian", ("forget", "vegetarian")),
            ("forget what I said about meetings", ("forget", "meetings")),
        ],
    )
    def test_the_tool_and_its_text(self, said, expected):
        assert memory_request(said) == expected

    @pytest.mark.parametrize("said", ["what do you know about me", "make it cooler", "hello"])
    def test_anything_else_is_not_a_memory_request(self, said):
        assert memory_request(said) is None


class TestANamedTemperature:
    """Live: "500 degrees" came back with no target, so the gate never answered."""

    def test_the_number_the_model_dropped_is_restored(self):
        hint = understood(_hint(), "can you make the temperature go to 500 degrees centigrade")
        assert hint.intent is Intent.ENVIRONMENT and hint.target_c == 500.0

    def test_a_request_filed_as_a_question_is_a_request(self):
        hint = understood(_hint(intent=Intent.QUESTION), "set it to 22 °C please")
        assert hint.intent is Intent.ENVIRONMENT and hint.target_c == 22.0

    def test_a_question_about_degrees_stays_a_question(self):
        hint = understood(_hint(intent=Intent.QUESTION), "what is 20 degrees in fahrenheit")
        assert hint.intent is Intent.QUESTION and hint.target_c is None

    def test_the_models_own_number_is_kept(self):
        hint = understood(_hint(target_c=23.0), "make it 23 degrees")
        assert hint.target_c == 23.0


class TestATimeIsNotATemperature:
    """S-03, live: "actually make it half past two" moved the setpoint 21 -> 21.5."""

    def test_a_time_read_as_warmer_is_a_follow_up(self):
        hint = understood(_hint(comfort=Comfort.WARMER), "actually make it half past two", (CALENDAR_TURN,))
        assert hint.intent is Intent.SERVICE and hint.comfort is Comfort.UNCHANGED and hint.target_c is None

    def test_even_with_no_conversation_to_go_on(self):
        hint = understood(_hint(comfort=Comfort.WARMER), "make it half past eleven")
        assert hint.intent is Intent.SERVICE

    def test_a_time_read_as_a_temperature_is_a_follow_up(self):
        hint = understood(_hint(target_c=21.5), "actually make it 10:30")
        assert hint.intent is Intent.SERVICE and hint.target_c is None

    def test_a_comfort_word_beside_a_time_still_wins(self):
        hint = understood(_hint(comfort=Comfort.COOLER), "make it cooler at six")
        assert hint.intent is Intent.ENVIRONMENT and hint.comfort is Comfort.COOLER

    def test_a_named_temperature_beside_a_time_still_wins(self):
        hint = understood(_hint(target_c=22.0), "make it 22 degrees at four")
        assert hint.intent is Intent.ENVIRONMENT and hint.target_c == 22.0
