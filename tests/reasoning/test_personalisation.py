"""Unit tests for answering a recognised person (FR-52).

The executor, the memory store and the keeper are real and talk over an
in-process blackboard; only the model is scripted. What is asserted is what
recognition may and may not change: who the reply is for, what the model is
told, and which memory a fact lands in -- never what anyone is permitted.
"""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_keeper, build_service, build_store
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import GUEST_PROFILE_ID, AssistantReply, Comfort, Intent, PreferenceHint
from src.reasoning.__main__ import build_assistant
from tests.reasoning.stubs import ScriptedChat, call, calling, saying


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    assistance = config.assistance.model_copy(
        update={"calendar_path": str(tmp_path / "calendar.json"), "result_timeout_s": 1.0}
    )
    profiles = config.profiles.model_copy(
        update={"directory": str(tmp_path / "profiles"), "context_directory": str(tmp_path / "context")}
    )
    return config.model_copy(update={"assistance": assistance, "profiles": profiles})


class Room:
    def __init__(self, config, chat) -> None:
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, self.transport) for _ in range(4)]
        for board in boards:
            self.transport.attach(board)
        self.store = build_store(config, self.clock)
        self.store.create("ravi", "Ravi")
        self.store.remember("ravi", "prefers meetings after ten")
        self.executor = build_service(config, self.clock, boards[0])
        self.executor.subscribe()
        self.keeper = build_keeper(config, self.clock, boards[1])
        self.keeper.subscribe()
        self.assistant = build_assistant(config, self.clock, boards[2], chat)
        self.assistant.subscribe()
        self.keeper.publish_all()
        self.caller = boards[3]
        self.chat = chat

    def say(self, transcript: str, profile_id: str = "ravi") -> AssistantReply:
        hint = PreferenceHint(
            ts=self.clock.now(),
            intent=Intent.SERVICE,
            comfort=Comfort.UNCHANGED,
            subject="memory",
            transcript=transcript,
            profile_id=profile_id,
        )
        self.caller.publish(topics.CONTEXT_PREFERENCE, hint)
        return self.assistant.process_pending()[-1]

    def system_prompt(self, request: int = 0) -> str:
        return self.chat.requests[request][0][0]["content"]

    def tools_offered(self, request: int = 0) -> set[str]:
        return {tool["function"]["name"] for tool in self.chat.requests[request][1]}


class TestWhatTheModelIsTold:
    def test_it_is_told_who_it_is_speaking_with(self, config):
        room = Room(config, ScriptedChat(saying("Hello Ravi.")))
        room.say("hello")
        assert "You are speaking with Ravi." in room.system_prompt()

    def test_it_is_told_what_they_asked_to_be_remembered(self, config):
        room = Room(config, ScriptedChat(saying("Noted.")))
        room.say("when should we meet")
        assert "prefers meetings after ten" in room.system_prompt()

    def test_it_is_told_the_conversation_so_far(self, config):
        room = Room(config, ScriptedChat(saying("Hello."), saying("At three.")))
        room.say("hello")
        room.say("and when is the review")
        assert "They said: hello" in room.system_prompt(request=1)

    def test_a_guest_is_told_nothing_about_anyone(self, config):
        room = Room(config, ScriptedChat(saying("Hello.")))
        room.say("hello", profile_id="")
        assert "Ravi" not in room.system_prompt()
        assert "meetings after ten" not in room.system_prompt()


class TestMemoryTools:
    def test_a_person_is_offered_memory(self, config):
        room = Room(config, ScriptedChat(saying("ok")))
        room.say("hello")
        assert {"remember", "forget"} <= room.tools_offered()

    def test_a_guest_is_not(self, config):
        room = Room(config, ScriptedChat(saying("ok")))
        room.say("hello", profile_id="")
        assert not {"remember", "forget"} & room.tools_offered()

    def test_a_fact_lands_in_the_speakers_own_memory(self, config):
        room = Room(config, ScriptedChat(calling(call("remember", fact="vegetarian"))))
        reply = room.say("remember that I am vegetarian")
        assert "vegetarian" in room.store.load("ravi").facts
        assert "remember" in reply.reply.lower()

    def test_the_model_cannot_choose_whose_memory(self, config):
        """A model naming another profile is overruled by the voice."""
        room = Room(config, ScriptedChat(calling(call("remember", fact="likes it cold", profile_id="anu"))))
        room.store.create("anu", "Anu")
        room.say("remember that I like it cold")
        assert "likes it cold" in room.store.load("ravi").facts
        assert room.store.load("anu").facts == ()

    def test_forgetting_through_the_model(self, config):
        room = Room(config, ScriptedChat(calling(call("forget", about="meetings"))))
        room.say("forget what I said about meetings")
        assert room.store.load("ravi").facts == ()


class TestTheWordsDecideTheMemoryTool:
    """U-07, measured live: the model claimed actions it never took."""

    def test_a_remember_the_model_left_undone_is_carried_out(self, config):
        """It replied "I'll remember that you like tea" and called nothing."""
        room = Room(config, ScriptedChat(saying("I'll remember that you like tea, Ravi.")))
        room.say("remember that I like tea")
        assert "Likes tea." in room.store.load("ravi").facts

    def test_the_reply_is_the_providers_not_the_models_claim(self, config):
        room = Room(config, ScriptedChat(saying("I've forgotten that you are vegetarian, Ravi.")))
        reply = room.say("forget that I am vegetarian")
        assert "nothing" in reply.reply.lower()

    def test_forget_called_for_a_remember_is_replaced(self, config):
        """'Please remember that I am vegetarian' was answered with forget."""
        room = Room(config, ScriptedChat(calling(call("forget", about="vegetarian"))))
        room.say("please remember that I am vegetarian")
        assert "Is vegetarian." in room.store.load("ravi").facts

    def test_the_models_phrasing_is_kept_when_it_chose_right(self, config):
        room = Room(config, ScriptedChat(calling(call("remember", fact="Likes green tea."))))
        room.say("remember that I like green tea")
        assert "Likes green tea." in room.store.load("ravi").facts

    def test_a_guest_gets_no_memory_from_the_words_either(self, config):
        room = Room(config, ScriptedChat(saying("I can't keep that.")))
        room.say("remember that I like tea", profile_id="")
        assert "Likes tea." not in room.store.load("ravi").facts


class TestRecognitionPermitsNothing:
    def test_the_first_reply_says_recognition_only_personalises(self, config):
        room = Room(config, ScriptedChat(saying("Hello.")))
        reply = room.say("hello")
        assert "recognised your voice as Ravi" in reply.reply
        assert "doesn't unlock anything" in reply.reply

    def test_it_is_said_once_per_conversation(self, config):
        room = Room(config, ScriptedChat(saying("Hello."), saying("Fine.")))
        room.say("hello")
        assert "recognised your voice" not in room.say("how are you").reply

    def test_a_guest_is_not_told_they_were_recognised(self, config):
        room = Room(config, ScriptedChat(saying("Hello.")))
        assert "recognised" not in room.say("hello", profile_id="").reply

    def test_a_booking_still_needs_confirmation(self, config):
        """FR-54 and FR-52: a recognised voice confirms nothing."""
        room = Room(
            config,
            ScriptedChat(
                calling(call("book_travel", kind="flight", destination="Delhi", depart_on="2026-10-01T09:00:00"))
            ),
        )
        reply = room.say("book me a flight to Delhi")
        assert reply.awaiting_confirmation
        assert "Nothing is booked yet" in reply.reply

    def test_the_reply_names_whose_it_is(self, config):
        room = Room(config, ScriptedChat(saying("Hello.")))
        assert room.say("hello").profile_id == "ravi"

    def test_the_exchange_becomes_their_short_term_context(self, config):
        room = Room(config, ScriptedChat(saying("Hello.")))
        room.say("hello")
        assert room.store.recent("ravi")[-1].said == "hello"

    def test_a_guests_exchange_is_the_guests(self, config):
        room = Room(config, ScriptedChat(saying("Hello.")))
        room.say("hello", profile_id="")
        assert room.store.recent(GUEST_PROFILE_ID)[-1].said == "hello"


def _ask(room, transcript, profile_id="ravi", typed=False):
    hint = PreferenceHint(
        ts=room.clock.now(), intent=Intent.QUESTION, comfort=Comfort.UNCHANGED,
        subject="memory", transcript=transcript, profile_id=profile_id, typed=typed,
    )
    room.caller.publish(topics.CONTEXT_PREFERENCE, hint)
    return room.assistant.process_pending()[-1]


class TestWhatIsKnownIsReadFromTheProfile:
    """S-05 and S-06, as met live."""

    def test_the_answer_is_the_profile_as_it_stands(self, config):
        room = Room(config, ScriptedChat())
        reply = _ask(room, "what do you know about me?")
        assert "prefers meetings after ten" in reply.reply.lower() or "prefer meetings after ten" in reply.reply.lower()
        assert room.chat.requests == []

    def test_a_forgotten_fact_is_never_recited(self, config):
        """Live: after 'forget that I'm vegetarian' the reply still said 'and are vegetarian'."""
        room = Room(config, ScriptedChat(calling(call("forget", about="vegetarian")), saying("ok")))
        room.store.remember("ravi", "Is vegetarian.")
        room.keeper.publish_all()
        room.say("forget that I'm vegetarian")
        assert "vegetarian" not in _ask(room, "what do you know about me?").reply

    def test_a_kept_fact_is_included_at_once(self, config):
        room = Room(config, ScriptedChat())
        room.store.suggest("ravi", "Can never work if the room is noisy.")
        room.store.settle("ravi", "Can never work if the room is noisy.", keep=True)
        room.keeper.publish_all()
        assert "can never work if the room is noisy" in _ask(room, "what do you know about me?").reply

    def test_a_guest_is_told_nothing_is_kept_about_them(self, config):
        """Live: a guest was told about 'a meeting with Ravi' as something known about them."""
        reply = _ask(Room(config, ScriptedChat()), "what do you know about me?", profile_id="")
        assert "keep nothing about you" in reply.reply and "Ravi" not in reply.reply

    def test_overheard_facts_waiting_are_mentioned_not_recited(self, config):
        room = Room(config, ScriptedChat())
        room.store.suggest("ravi", "Hates the cold.")
        room.keeper.publish_all()
        reply = _ask(room, "what do you remember about me?")
        assert "waiting for you" in reply.reply and "cold" not in reply.reply

    def test_it_is_said_in_the_second_person(self):
        from src.reasoning.assistant import _as_you

        assert _as_you("Likes tea.") == "you like tea"
        assert _as_you("Is allergic to peanuts.") == "you are allergic to peanuts"


class TestTypedIsNotAVoice:
    """S-12: words typed as Ravi were answered 'I recognised your voice'."""

    def test_typed_says_chatting_as(self, config):
        reply = _ask(Room(config, ScriptedChat()), "what do you know about me?", typed=True)
        assert "chatting as Ravi" in reply.reply and "voice" not in reply.reply.split("(")[-1]

    def test_spoken_still_says_recognised(self, config):
        reply = _ask(Room(config, ScriptedChat()), "what do you know about me?")
        assert "recognised your voice as Ravi" in reply.reply
