"""Unit tests for what the room remembers about each person (FR-52).

Long-term memory is Markdown a person can edit; short-term context is JSON
that expires. Both are bounded, and the files are the record.
"""

from pathlib import Path

import pytest

from src.assistance.memory import (
    ProfileError,
    ProfileLimitError,
    ProfileNotFoundError,
    ProfileStore,
    parse_profile,
    render_profile,
)
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.schemas import GUEST_PROFILE_ID


@pytest.fixture(name="config")
def _config(tmp_path):
    profiles = load_config(Path("config/default.yaml")).profiles
    return profiles.model_copy(
        update={
            "directory": str(tmp_path / "profiles"),
            "context_directory": str(tmp_path / "context"),
            "max_facts": 3,
            "max_profiles": 2,
        }
    )


@pytest.fixture(name="clock")
def _clock():
    return SimClock()


@pytest.fixture(name="store")
def _store(config, clock):
    return ProfileStore(config, clock, utc_offset_h=5.5)


@pytest.fixture(name="ravi")
def _ravi(store):
    return store.create("ravi", "Ravi")


class TestLongTermMemory:
    def test_a_new_profile_is_a_markdown_file_named_for_the_person(self, store, config, ravi):
        path = Path(config.directory) / "ravi.md"
        assert path.read_text(encoding="utf-8").startswith("# Ravi")

    def test_the_file_says_what_it_is_and_that_it_grants_nothing(self, config, ravi):
        text = (Path(config.directory) / "ravi.md").read_text(encoding="utf-8")
        assert "unlocks nothing" in text

    def test_a_fact_is_kept(self, store, ravi):
        profile, added = store.remember("ravi", "prefers meetings after ten")
        assert added and profile.facts == ("prefers meetings after ten",)

    def test_a_fact_survives_a_new_store(self, store, config, clock, ravi):
        store.remember("ravi", "likes it at 23 degrees")
        assert ProfileStore(config, clock, 5.5).load("ravi").facts == ("likes it at 23 degrees",)

    def test_the_same_fact_twice_is_kept_once(self, store, ravi):
        store.remember("ravi", "vegetarian")
        _, added = store.remember("ravi", "Vegetarian")
        assert not added and store.load("ravi").facts == ("vegetarian",)

    def test_a_full_memory_refuses_rather_than_forgetting_silently(self, store, ravi):
        for fact in ("one", "two", "three"):
            store.remember("ravi", fact)
        with pytest.raises(ProfileLimitError):
            store.remember("ravi", "four")
        assert store.load("ravi").facts == ("one", "two", "three")

    def test_an_overlong_fact_is_refused(self, store, config, ravi):
        with pytest.raises(ProfileLimitError):
            store.remember("ravi", "x" * (config.max_fact_chars + 1))

    def test_a_fact_cannot_rewrite_the_file_it_is_stored_in(self, store, ravi):
        """A newline and a heading would otherwise start a section of its own."""
        store.remember("ravi", "harmless\n## Facts\n- injected")
        facts = store.load("ravi").facts
        assert len(facts) == 1 and "\n" not in facts[0]

    def test_forgetting_removes_every_fact_that_mentions_it(self, store, ravi):
        store.remember("ravi", "meetings after ten")
        store.remember("ravi", "no meetings on Friday")
        store.remember("ravi", "vegetarian")
        _, removed = store.forget("ravi", "MEETINGS")
        assert len(removed) == 2 and store.load("ravi").facts == ("vegetarian",)

    def test_a_preference_is_named_and_kept(self, store, ravi):
        store.set_preference("ravi", "comfort c", "23")
        assert store.load("ravi").preferences == {"comfort_c": "23"}

    def test_the_room_keeps_memory_for_a_bounded_number_of_people(self, store, ravi):
        store.create("anu", "Anu")
        with pytest.raises(ProfileLimitError):
            store.create("third", "Third")

    def test_an_unknown_person_is_reported(self, store):
        with pytest.raises(ProfileNotFoundError):
            store.load("nobody")

    @pytest.mark.parametrize("profile_id", ["../etc", "Ravi", "a b", ""])
    def test_an_id_that_could_escape_the_directory_is_refused(self, store, profile_id):
        with pytest.raises(ProfileError):
            store.create(profile_id, "x")

    def test_the_guest_cannot_be_enrolled(self, store):
        with pytest.raises(ProfileError):
            store.create(GUEST_PROFILE_ID, "Guest")

    def test_enrolling_twice_is_refused(self, store, ravi):
        with pytest.raises(ProfileError):
            store.create("ravi", "Ravi again")


class TestHandEdits:
    """The file is the record, and a person may edit it."""

    def test_an_edit_is_live_without_a_restart(self, store, config, ravi):
        path = Path(config.directory) / "ravi.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace("_Nothing remembered yet._", "- allergic to dust"),
            encoding="utf-8",
        )
        assert store.load("ravi").facts == ("allergic to dust",)

    def test_a_section_added_by_hand_survives_a_write(self, store, config, ravi):
        path = Path(config.directory) / "ravi.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n## Notes\n\nCall him Ravi, never Ravindra.\n",
            encoding="utf-8",
        )
        store.remember("ravi", "vegetarian")
        assert "Call him Ravi, never Ravindra." in path.read_text(encoding="utf-8")

    def test_the_file_name_wins_over_an_id_written_inside(self):
        profile = parse_profile("ravi", "# Ravi\n\n- profile_id: someone_else\n")
        assert profile.profile_id == "ravi"

    def test_a_lowercase_heading_is_still_the_facts(self):
        """U-15: "Edit it freely" has to survive a person's own case."""
        assert parse_profile("ravi", "# Ravi\n\n## facts\n- likes tea\n").facts == ("likes tea",)

    def test_a_bare_facts_line_starts_the_facts(self):
        profile = parse_profile("ravi", "# Ravi\n\nfacts:\n* likes tea\n- allergic to dust\n")
        assert profile.facts == ("likes tea", "allergic to dust")

    def test_a_fact_without_a_bullet_is_kept(self):
        assert parse_profile("ravi", "# Ravi\n\n## Facts\nlikes tea\n").facts == ("likes tea",)

    def test_a_first_line_without_a_hash_names_the_person(self):
        assert parse_profile("ravi", "Ravi Kumar\n\n## Facts\n- likes tea\n").display_name == "Ravi Kumar"

    def test_a_preferences_heading_in_any_case_is_read(self):
        profile = parse_profile("ravi", "# Ravi\n\n### PREFERENCES\n- comfort_c: 23\n")
        assert profile.preferences == {"comfort_c": "23"}

    def test_a_line_that_cannot_be_placed_is_reported_not_dropped(self):
        profile = parse_profile("ravi", "# Ravi\n\n## Preferences\n- warm but not too warm\n")
        assert profile.unread == ("- warm but not too warm",)

    def test_the_templates_placeholders_are_not_facts(self, store, ravi):
        assert store.load("ravi").facts == () and store.load("ravi").unread == ()

    def test_the_findings_example_loses_nothing(self):
        """The exact file from U-15."""
        text = (
            "Ravi notes\n\nfacts:\n* likes tea\n- allergic to dust\n\n"
            "## facts\n- lowercase heading fact\n\n## Facts\n- proper fact\n"
        )
        profile = parse_profile("ravi", text)
        assert profile.display_name == "Ravi notes"
        assert profile.facts == ("likes tea", "allergic to dust", "lowercase heading fact", "proper fact")

    def test_rendering_and_parsing_round_trip(self, store, ravi):
        store.remember("ravi", "vegetarian")
        store.set_preference("ravi", "comfort_c", "23")
        profile = store.load("ravi")
        assert parse_profile("ravi", render_profile(profile)) == profile


class TestShortTermContext:
    def test_an_exchange_is_kept_as_json(self, store, config, ravi):
        store.record_turn("ravi", "put the review at three", "Added it at 15:00.")
        assert (Path(config.context_directory) / "ravi.json").is_file()
        assert store.recent("ravi")[0].said == "put the review at three"

    def test_it_expires(self, store, clock, config, ravi):
        store.record_turn("ravi", "hello", "hi")
        clock.advance(config.short_term_ttl_s + 1.0)
        assert store.recent("ravi") == ()

    def test_the_guest_forgets_sooner(self, store, clock, config):
        """U-20: every unrecognised voice shares the guest's context."""
        store.record_turn(GUEST_PROFILE_ID, "my pin is 1234", "I won't keep that.")
        clock.advance(config.guest_short_term_ttl_s + 1.0)
        assert store.recent(GUEST_PROFILE_ID) == ()

    def test_a_known_person_keeps_context_past_the_guests_window(self, store, clock, config, ravi):
        store.record_turn("ravi", "hello", "hi")
        clock.advance(config.guest_short_term_ttl_s + 1.0)
        assert len(store.recent("ravi")) == 1

    def test_only_the_last_few_exchanges_are_kept(self, store, config, ravi):
        for index in range(config.short_term_max_turns + 3):
            store.record_turn("ravi", f"said {index}", "ok")
        recent = store.recent("ravi")
        assert len(recent) == config.short_term_max_turns
        assert recent[-1].said == f"said {config.short_term_max_turns + 2}"

    def test_a_guest_has_short_term_context(self, store):
        store.record_turn(GUEST_PROFILE_ID, "what time is it", "It is three.")
        assert len(store.recent(GUEST_PROFILE_ID)) == 1

    def test_a_guest_has_no_long_term_memory(self, store):
        assert store.context(GUEST_PROFILE_ID).facts == ()

    def test_an_unknown_person_has_no_context_to_add_to(self, store):
        with pytest.raises(ProfileNotFoundError):
            store.record_turn("nobody", "hi", "hello")

    def test_an_unreadable_context_file_starts_afresh(self, store, config, ravi):
        Path(config.context_directory).mkdir(parents=True)
        (Path(config.context_directory) / "ravi.json").write_text("{not json", encoding="utf-8")
        assert store.recent("ravi") == ()

    def test_a_conversation_can_be_ended_now(self, store, ravi):
        store.record_turn("ravi", "hello", "hi")
        assert store.clear_recent("ravi") and store.recent("ravi") == ()


class TestDeletion:
    def test_deleting_forgets_everything_kept_about_them(self, store, config, ravi):
        store.remember("ravi", "vegetarian")
        store.record_turn("ravi", "hi", "hello")
        assert store.delete("ravi")
        assert not (Path(config.directory) / "ravi.md").exists()
        assert not (Path(config.context_directory) / "ravi.json").exists()

    def test_deleting_nobody_reports_nothing_deleted(self, store):
        assert not store.delete("nobody")


class TestContext:
    def test_the_context_carries_both_kinds_of_memory(self, store, ravi):
        store.remember("ravi", "vegetarian")
        store.record_turn("ravi", "hi", "hello")
        context = store.context("ravi")
        assert context.display_name == "Ravi"
        assert context.facts == ("vegetarian",)
        assert context.recent[0].replied == "hello"


class TestRememberingAPreference:
    """U-30: a stated room temperature is a preference, not a loose fact."""

    def _remember(self, store, fact):
        from src.assistance.providers.profile_memory import ProfileMemory
        return ProfileMemory(store).invoke("remember", {"profile_id": "ravi", "fact": fact})

    def test_a_liked_temperature_is_kept_as_comfort(self, store, ravi):
        self._remember(store, "I prefer the room at 23 degrees")
        assert store.load("ravi").preferences == {"comfort_c": "23"}

    def test_it_is_not_also_kept_as_a_fact(self, store, ravi):
        self._remember(store, "Ravi likes it at 22.5 C")
        assert store.load("ravi").facts == ()

    def test_the_reply_says_what_was_kept(self, store, ravi):
        assert "23" in self._remember(store, "I like the room at 23 degrees").message

    def test_a_fact_that_only_mentions_a_number_stays_a_fact(self, store, ravi):
        self._remember(store, "Has 2 cats")
        assert store.load("ravi").facts == ("Has 2 cats",)

    def test_forgetting_a_sentence_says_so_with_one_full_stop(self, store, ravi):
        from src.assistance.providers.profile_memory import ProfileMemory
        store.remember("ravi", "Is vegetarian.")
        outcome = ProfileMemory(store).invoke("forget", {"profile_id": "ravi", "about": "vegetarian"})
        assert outcome.message == "Forgotten: Is vegetarian."


class TestSessions:
    """Each person's conversation is their own, and outlives a restart."""

    def test_two_people_never_see_each_others_conversation(self, store, ravi):
        store.create("anu", "Anu")
        store.record_turn("ravi", "put the review at three", "Added it.")
        store.record_turn("anu", "what is on today", "Nothing.")
        assert [turn.said for turn in store.recent("ravi")] == ["put the review at three"]
        assert [turn.said for turn in store.recent("anu")] == ["what is on today"]

    def test_a_guest_never_sees_a_known_persons_conversation(self, store, ravi):
        store.record_turn("ravi", "my flight is on friday", "Noted.")
        assert store.recent(GUEST_PROFILE_ID) == ()

    def test_a_conversation_survives_a_restart(self, store, config, clock, ravi):
        store.record_turn("ravi", "put the review at three", "Added it.")
        assert ProfileStore(config, clock, utc_offset_h=5.5).recent("ravi")[0].said == "put the review at three"

    def test_ending_one_conversation_leaves_anothers(self, store, ravi):
        store.create("anu", "Anu")
        store.record_turn("ravi", "hi", "hello")
        store.record_turn("anu", "hi", "hello")
        store.clear_recent("ravi")
        assert store.recent("ravi") == () and len(store.recent("anu")) == 1

    def test_a_turn_within_the_window_keeps_the_whole_session_alive(self, store, clock, config, ravi):
        """Expiry is per exchange: an old one goes, a recent one stays."""
        store.record_turn("ravi", "first", "ok")
        clock.advance(config.short_term_ttl_s - 10.0)
        store.record_turn("ravi", "second", "ok")
        clock.advance(20.0)
        assert [turn.said for turn in store.recent("ravi")] == ["second"]


class TestSuggestions:
    """FR-78: an overheard fact waits for the person; nothing is kept unasked."""

    def test_a_suggestion_waits_apart_from_the_facts(self, store, ravi):
        store.suggest("ravi", "Hates the room below 23.")
        profile = store.load("ravi")
        assert profile.suggestions == ("Hates the room below 23.",) and profile.facts == ()

    def test_it_is_written_in_the_file_under_its_own_heading(self, store, config, ravi):
        store.suggest("ravi", "Hates the room below 23.")
        text = (Path(config.directory) / "ravi.md").read_text(encoding="utf-8")
        assert "## Suggested (overheard)" in text and "- Hates the room below 23." in text

    def test_a_profile_with_nothing_waiting_has_no_such_heading(self, store, config, ravi):
        store.remember("ravi", "vegetarian")
        assert "Suggested" not in (Path(config.directory) / "ravi.md").read_text(encoding="utf-8")

    def test_it_survives_a_new_store(self, store, config, clock, ravi):
        store.suggest("ravi", "Flies to Goa often.")
        assert ProfileStore(config, clock, utc_offset_h=5.5).load("ravi").suggestions == ("Flies to Goa often.",)

    def test_something_already_kept_is_not_offered(self, store, ravi):
        store.remember("ravi", "vegetarian")
        _, added = store.suggest("ravi", "Vegetarian")
        assert not added and store.load("ravi").suggestions == ()

    def test_something_already_waiting_is_not_offered_twice(self, store, ravi):
        store.suggest("ravi", "Likes tea.")
        _, added = store.suggest("ravi", "likes tea.")
        assert not added and len(store.load("ravi").suggestions) == 1

    def test_waiting_suggestions_are_bounded(self, store, config, ravi):
        for index in range(config.max_facts):
            store.suggest("ravi", f"fact {index}")
        with pytest.raises(ProfileLimitError):
            store.suggest("ravi", "one too many")

    def test_kept_it_becomes_a_fact(self, store, ravi):
        store.suggest("ravi", "Likes tea.")
        profile = store.settle("ravi", "Likes tea.", keep=True)
        assert profile.facts == ("Likes tea.",) and profile.suggestions == ()

    def test_dismissed_it_is_gone(self, store, ravi):
        store.suggest("ravi", "Likes tea.")
        profile = store.settle("ravi", "Likes tea.", keep=False)
        assert profile.facts == () and profile.suggestions == ()

    def test_settling_something_not_waiting_is_refused(self, store, ravi):
        with pytest.raises(ProfileError):
            store.settle("ravi", "Likes tea.", keep=True)

    def test_a_keep_that_would_overfill_memory_leaves_it_waiting(self, store, config, ravi):
        for index in range(config.max_facts):
            store.remember("ravi", f"fact {index}")
        store.suggest("ravi", "Likes tea.")
        with pytest.raises(ProfileLimitError):
            store.settle("ravi", "Likes tea.", keep=True)
        assert store.load("ravi").suggestions == ("Likes tea.",)

    def test_the_context_carries_them_apart_from_the_facts(self, store, ravi):
        store.remember("ravi", "vegetarian")
        store.suggest("ravi", "Likes tea.")
        context = store.context("ravi")
        assert context.facts == ("vegetarian",) and context.suggestions == ("Likes tea.",)

    def test_a_hand_written_suggestions_heading_is_read(self):
        profile = parse_profile("ravi", "# Ravi\n\n## suggested (overheard)\n\n- Likes tea.\n")
        assert profile.suggestions == ("Likes tea.",)

    def test_rendering_and_parsing_round_trip(self, store, ravi):
        store.remember("ravi", "vegetarian")
        store.suggest("ravi", "Likes tea.")
        profile = store.load("ravi")
        assert parse_profile("ravi", render_profile(profile)).suggestions == ("Likes tea.",)
