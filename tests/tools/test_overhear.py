"""The terminal stand-in for speech overheard without the wake word (FR-76).

Run on the in-process bus against the real miner and executor, so a typed
line is shown to arrive exactly where an overheard one would, and its
outcome to be the calendar's and the profile's own.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_service as build_executor
from src.assistance.__main__ import build_store
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.localtime import epoch_of
from src.common.mqtt_client import Blackboard
from src.common.schemas import AmbientUtterance
from src.reasoning.__main__ import build_miner
from tests.reasoning.stubs import ScriptedChat, saying
from tools.overhear import Overhearing, main

SATURDAY_AFTERNOON = epoch_of(
    datetime(2026, 9, 26, 15, 0), load_config(Path("config/default.yaml")).site.utc_offset_h
)
DENTIST = {"subject": "dentist", "starts_at": "2026-10-01T16:00:00"}


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    assistance = config.assistance.model_copy(update={"calendar_path": str(tmp_path / "calendar.json")})
    profiles = config.profiles.model_copy(
        update={"directory": str(tmp_path / "profiles"), "context_directory": str(tmp_path / "context")}
    )
    return config.model_copy(update={"assistance": assistance, "profiles": profiles})


class Room:
    def __init__(self, config, answer: str) -> None:
        self.clock = SimClock(start_epoch_s=SATURDAY_AFTERNOON)
        self.transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, self.transport) for _ in range(3)]
        for board in boards:
            self.transport.attach(board)
        build_executor(config, self.clock, boards[0]).subscribe()
        self.miner = build_miner(config, self.clock, boards[1], ScriptedChat(saying(answer)))
        self.miner.subscribe()
        build_store(config, self.clock).create("ravi", "Ravi")
        self.boards = boards

    def overhearing(self, speaker: str = "") -> Overhearing:
        overhearing = Overhearing(self.clock, self.boards[2], speaker)
        overhearing.subscribe()
        return overhearing


def test_the_line_is_published_as_overheard_speech(config):
    room = Room(config, json.dumps({"plans": [], "facts": []}))
    room.overhearing("ravi").say("dentist   on thursday at four")
    (payload,) = [p for t, p, _, _ in room.transport.published if t == topics.CONTEXT_AMBIENT.pattern]
    heard = AmbientUtterance.model_validate_json(payload)
    assert heard.transcript == "dentist on thursday at four" and heard.profile_id == "ravi"


def test_a_pencilled_plan_is_reported(config):
    room = Room(config, json.dumps({"plans": [DENTIST], "facts": []}))
    overhearing = room.overhearing("ravi")
    overhearing.say("I've got the dentist on Thursday at four")
    room.miner.process_pending()
    lines = overhearing.outcome(record_timeout_s=0.1, result_timeout_s=0.1)
    assert "1 plan(s) pencilled" in lines[0] and "tentatively" in lines[1]


def test_an_offered_fact_is_reported(config):
    room = Room(config, json.dumps({"plans": [], "facts": ["Hates the cold."]}))
    overhearing = room.overhearing("ravi")
    overhearing.say("honestly I hate the cold so much")
    room.miner.process_pending()
    lines = overhearing.outcome(record_timeout_s=0.1, result_timeout_s=0.1)
    assert any("suggest_fact" in line and "Hates the cold." in line for line in lines)


def test_nothing_reading_it_is_said_plainly(config):
    room = Room(config, json.dumps({"plans": [], "facts": []}))
    overhearing = room.overhearing()
    overhearing.say("nobody is listening to this at all")
    assert "reasoning service" in overhearing.outcome(record_timeout_s=0.05)[0]


def test_a_malformed_speaker_is_refused(capsys):
    assert main(["hello there everyone", "--as", "Not A Profile"]) == 2
    assert "not a profile id" in capsys.readouterr().out
