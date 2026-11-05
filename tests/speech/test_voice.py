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


class _Player:
    def __init__(self, fails=False):
        self.played: list[tuple[int, int]] = []
        self.fails = fails

    def __call__(self, pcm, rate_hz):
        if self.fails:
            raise OSError("no output device")
        self.played.append((len(pcm), rate_hz))


def _with_player(player=None, **kwargs):
    clock = _at(12)
    speaker = RecordingSpeaker()
    return Voice(speaker, clock, OFFSET_H, 0.4, player=player, **kwargs), speaker, clock


def _played(player, timeout_s=5.0):
    import time

    deadline = time.monotonic() + timeout_s
    while not player.played and time.monotonic() < deadline:
        time.sleep(0.01)
    return player.played


class TestAReplyIsATurn:
    """FR-93: after the room replies, it listens for the answer without the wake word."""

    def test_after_a_spoken_reply_a_window_is_offered_once(self):
        voice, speaker, clock = _with_player()
        voice.on_reply("t", _reply("Sorry, could you say that again?"))
        _said(voice, speaker)
        clock.advance(0.5)
        assert voice.take_follow_up() is True and voice.take_follow_up() is False

    def test_not_while_the_room_is_still_in_its_tail(self):
        voice, speaker, clock = _with_player()
        voice.on_reply("t", _reply())
        _said(voice, speaker)
        assert voice.take_follow_up() is False
        clock.advance(0.5)
        assert voice.take_follow_up() is True

    def test_an_alert_is_not_a_turn(self):
        voice, speaker, clock = _with_player()
        voice.on_mode("t", ModeState(ts=1.0, mode=Mode.NORMAL, since_ts=1.0))
        voice.on_mode("t", ModeState(ts=2.0, mode=Mode.SAFE_HOLD, since_ts=2.0))
        _said(voice, speaker)
        clock.advance(1.0)
        assert voice.take_follow_up() is False

    def test_a_reply_not_spoken_opens_nothing(self):
        voice, speaker, clock = _with_player()
        voice.on_reply("t", _reply(spoken=False))  # typed, and typed replies are not spoken
        _said(voice, speaker)
        clock.advance(1.0)
        assert voice.take_follow_up() is False

    def test_turned_off_it_opens_nothing(self):
        voice, speaker, clock = _with_player(follow_up=False)
        voice.on_reply("t", _reply())
        _said(voice, speaker)
        clock.advance(1.0)
        assert voice.take_follow_up() is False


class TestTheChime:
    def test_it_plays_and_the_microphone_ignores_it_meanwhile(self):
        player = _Player()
        voice, _, clock = _with_player(player=player)
        voice.chime()
        assert voice.chiming is True and _played(player)[0][1] == 22050
        clock.advance(1.0)
        assert voice.chiming is False

    def test_turned_off_it_plays_nothing(self):
        player = _Player()
        voice, _, _ = _with_player(player=player, chime=False)
        voice.chime()
        assert voice.chiming is False and player.played == []

    def test_no_audio_output_costs_only_the_chime(self, caplog):
        import time

        voice, _, _ = _with_player(player=_Player(fails=True))
        voice.chime()
        deadline = time.monotonic() + 5
        while "could not play the chime" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "could not play the chime" in caplog.text

    def test_it_is_two_short_notes_that_never_click(self):
        import numpy as np

        from src.speech.voice import chime_pcm, chime_seconds

        samples = np.frombuffer(chime_pcm(1.0), dtype=np.int16)
        assert len(samples) == int(22050 * 0.09) * 2 and chime_seconds() < 0.25
        assert abs(int(samples[0])) < 50 and abs(int(samples[-1])) < 50  # faded at both ends

    def test_it_follows_the_occupants_volume(self):
        import numpy as np

        from src.speech.voice import chime_pcm

        loud = np.abs(np.frombuffer(chime_pcm(1.0), dtype=np.int16)).max()
        soft = np.abs(np.frombuffer(chime_pcm(0.25), dtype=np.int16)).max()
        assert soft < loud / 3


class TestARemoteReply:
    def test_is_never_said_in_the_room(self):
        """FR-94: the person asking from their phone hears it there; the room stays silent."""
        voice, speaker, _ = _voice()
        voice.on_settings("t", SpeechSettings(ts=1.0, speak_typed=True))
        voice.on_reply("t", _reply().model_copy(update={"remote": True}))
        assert _said(voice, speaker) == []

    def test_opens_no_window_in_the_room(self):
        voice, speaker, clock = _voice()
        voice.on_reply("t", _reply().model_copy(update={"remote": True}))
        _said(voice, speaker)
        clock.advance(1.0)
        assert voice.take_follow_up() is False
