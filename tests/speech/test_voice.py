"""Unit tests for the room speaking its replies (FR-79, FR-82, FR-83).

A recording speaker stands in for Piper and the loudspeaker, so what is said
is asserted on and nothing is played.
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.localtime import epoch_of
from src.common.schemas import AssistantReply, Intent, Mode, ModeState, SpeechSettings
from src.speech.voice import Voice, build_speaker

OFFSET_H = load_config(Path("config/default.yaml")).site.utc_offset_h


class RecordingSpeaker:
    name = "recording"

    def __init__(self, clock=None, takes_s=0.0):
        self.said: list[tuple[str, float]] = []
        self.clock, self.takes_s = clock, takes_s

    def say(self, text, volume):
        self.said.append((text, volume))
        if self.clock is not None:
            self.clock.advance(self.takes_s)


def _at(hour, minute=0) -> SimClock:
    return SimClock(start_epoch_s=epoch_of(datetime(2026, 10, 26, hour, minute), OFFSET_H))


def _voice(clock=None, speaker=None, tail_s=0.4):
    clock = clock or _at(12)
    speaker = speaker or RecordingSpeaker()
    return Voice(speaker, clock, OFFSET_H, tail_s), speaker, clock


def _reply(text="Done. The room is now set to 23 degrees.", spoken=True):
    return AssistantReply(ts=1.0, transcript="make it 23", intent=Intent.ENVIRONMENT, reply=text, spoken=spoken)


def _said(voice, speaker):
    while voice.speak_next(timeout_s=0.0):
        pass
    return [text for text, _ in speaker.said]


class TestWhatIsSpoken:
    def test_a_reply_to_a_spoken_request_is_said(self):
        voice, speaker, _ = _voice()
        voice.on_reply("t", _reply())
        assert _said(voice, speaker) == ["Done. The room is now set to 23 degrees."]

    def test_a_typed_reply_is_not_said_by_default(self):
        voice, speaker, _ = _voice()
        voice.on_reply("t", _reply(spoken=False))
        assert _said(voice, speaker) == []

    def test_unless_the_occupant_asks_for_it(self):
        voice, speaker, clock = _voice()
        voice.on_settings("t", SpeechSettings(ts=clock.now(), speak_typed=True))
        voice.on_reply("t", _reply(spoken=False))
        assert len(_said(voice, speaker)) == 1

    def test_replies_can_be_turned_off(self):
        voice, speaker, clock = _voice()
        voice.on_settings("t", SpeechSettings(ts=clock.now(), speak_replies=False))
        voice.on_reply("t", _reply())
        assert _said(voice, speaker) == []

    def test_it_is_said_at_the_chosen_volume(self):
        voice, speaker, clock = _voice()
        voice.on_settings("t", SpeechSettings(ts=clock.now(), volume=0.3))
        voice.on_reply("t", _reply())
        _said(voice, speaker)
        assert speaker.said[0][1] == 0.3


class TestAlerts:
    def _mode(self, clock, mode):
        return ModeState(ts=clock.now(), mode=mode, since_ts=clock.now())

    def test_the_mode_the_room_is_already_in_is_not_announced(self):
        voice, speaker, clock = _voice()
        voice.on_mode("t", self._mode(clock, Mode.NORMAL))
        assert _said(voice, speaker) == []

    def test_a_change_is_said_in_plain_words(self):
        voice, speaker, clock = _voice()
        voice.on_mode("t", self._mode(clock, Mode.NORMAL))
        voice.on_mode("t", self._mode(clock, Mode.DEGRADED_ACTUATOR))
        (said,) = _said(voice, speaker)
        assert said.startswith("The air conditioner isn't cooling the room") and "press Reset" in said

    def test_the_same_mode_twice_is_said_once(self):
        voice, speaker, clock = _voice()
        for mode in (Mode.NORMAL, Mode.DEGRADED_SENSOR, Mode.DEGRADED_SENSOR):
            voice.on_mode("t", self._mode(clock, mode))
        assert len(_said(voice, speaker)) == 1

    def test_alerts_can_be_turned_off(self):
        voice, speaker, clock = _voice()
        voice.on_settings("t", SpeechSettings(ts=clock.now(), speak_alerts=False))
        voice.on_mode("t", self._mode(clock, Mode.NORMAL))
        voice.on_mode("t", self._mode(clock, Mode.DEGRADED_SENSOR))
        assert _said(voice, speaker) == []


class TestQuietHours:
    """FR-83: quiet hours silence what nobody asked for, not what they did."""

    @pytest.mark.parametrize("hour, quiet", [(23, True), (3, True), (7, False), (12, False), (22, True)])
    def test_across_midnight(self, hour, quiet):
        voice, _, clock = _voice(clock=_at(hour))
        voice.on_settings("t", SpeechSettings(ts=clock.now(), quiet_from="22:00", quiet_to="07:00"))
        assert voice.quiet_now() is quiet

    def test_an_alert_waits_out_quiet_hours(self):
        voice, speaker, clock = _voice(clock=_at(23))
        voice.on_settings("t", SpeechSettings(ts=clock.now(), quiet_from="22:00", quiet_to="07:00"))
        voice.on_mode("t", ModeState(ts=clock.now(), mode=Mode.NORMAL, since_ts=clock.now()))
        voice.on_mode("t", ModeState(ts=clock.now(), mode=Mode.DEGRADED_SENSOR, since_ts=clock.now()))
        assert _said(voice, speaker) == []

    def test_a_requested_reply_is_still_said(self):
        voice, speaker, clock = _voice(clock=_at(23))
        voice.on_settings("t", SpeechSettings(ts=clock.now(), quiet_from="22:00", quiet_to="07:00"))
        voice.on_reply("t", _reply())
        assert len(_said(voice, speaker)) == 1

    def test_no_quiet_hours_means_never_quiet(self):
        voice, _, _ = _voice(clock=_at(3))
        assert voice.quiet_now() is False


class TestTheRoomNeverHearsItself:
    """FR-79: the microphone is ignored while speaking and for a tail after."""

    def test_busy_while_a_reply_waits(self):
        voice, _, _ = _voice()
        voice.on_reply("t", _reply())
        assert voice.busy

    def test_busy_for_the_tail_after_speaking(self):
        clock = _at(12)
        voice, speaker, _ = _voice(clock=clock, speaker=RecordingSpeaker(clock, takes_s=2.0), tail_s=0.4)
        voice.on_reply("t", _reply())
        voice.speak_next(timeout_s=0.0)
        assert voice.busy
        clock.advance(0.5)
        assert not voice.busy

    def test_reports_speaking_while_it_speaks(self):
        told = []
        clock = _at(12)
        voice = Voice(RecordingSpeaker(), clock, OFFSET_H, 0.4, on_speaking=told.append)
        voice.on_reply("t", _reply())
        voice.speak_next(timeout_s=0.0)
        assert told == [True, False]


class TestNoVoice:
    def test_configured_off_is_silent_and_never_busy(self):
        config = load_config(Path("config/default.yaml")).speech.model_copy(update={"voice_engine": "none"})
        voice = Voice(build_speaker(config), _at(12), OFFSET_H, 0.4)
        voice.on_reply("t", _reply())
        assert voice.engine == "" and not voice.busy

    def test_a_missing_piper_model_falls_back_rather_than_failing(self, monkeypatch):
        import src.speech.voice as module

        class Fallback:
            name = "pyttsx3"

        monkeypatch.setattr(module, "SystemSpeaker", Fallback)
        config = load_config(Path("config/default.yaml")).speech.model_copy(update={"voice_model_path": "no/such/voice.onnx"})
        assert build_speaker(config).name == "pyttsx3"

    def test_a_backlog_drops_the_oldest(self):
        voice, speaker, _ = _voice()
        for n in range(10):
            voice.say(f"sentence {n}")
        said = _said(voice, speaker)
        assert len(said) <= 4 and said[-1] == "sentence 9"
