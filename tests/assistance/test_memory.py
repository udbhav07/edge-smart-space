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
