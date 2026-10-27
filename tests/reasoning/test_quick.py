"""Short device commands, understood without a model (FR-97).

What matters: "stop", "next song", "louder", "turn off the fan" become the
device call they plainly mean, in no time; anything longer or less plain --
or a bare "stop" with nothing playing, which may be about something else --
is left to the model; and the call is still run through the executor.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.devices import DeviceCapability, DeviceClass, DeviceKind, DeviceManifest, DeviceState
from src.common.schemas import Intent
from src.common.tools import ParameterType, ToolEffect, ToolParameter
from src.reasoning.devices import KnownDevices
from src.reasoning.quick import quick_command
from src.reasoning.single_shot import PersonalContext

CONFIG = load_config(Path("config/default.yaml"))


def _capability(name, *parameters, device_class=DeviceClass.MEDIA):
    return DeviceCapability(name=name, purpose=f"{name}.", effect=ToolEffect.WRITE, device_class=device_class, parameters=parameters)


VOLUME = ToolParameter(name="volume", type=ParameterType.INTEGER, description="percent", minimum=0, maximum=100)
MUSIC = DeviceManifest(
    ts=1.0, device_id="music", name="Music", kind=DeviceKind.MEDIA, aliases=("the music", "song", "speaker"), simulated=False,
    capabilities=tuple(_capability(name) for name in ("play", "pause", "resume", "stop", "next")) + (_capability("set_volume", VOLUME),),
)
FAN = DeviceManifest(
    ts=1.0, device_id="fan", name="Ceiling fan", kind=DeviceKind.FAN, aliases=("the fan",), simulated=True,
    capabilities=(_capability("turn_on", device_class=DeviceClass.COMFORT), _capability("turn_off", device_class=DeviceClass.COMFORT)),
)


def _known(playing=True, paused=False, volume=60):
    known = KnownDevices()
    known._on_manifest("", MUSIC)
    known._on_manifest("", FAN)
    known._on_state("", DeviceState(ts=1.0, device_id="music", simulated=False, values={"playing": playing, "paused": paused, "volume": volume}))
    return known


def _call(text, **state):
    command = quick_command(text, _known(**state))
    return None if command is None else (command.device_id, command.capability, command.arguments)


class TestRecognised:
    @pytest.mark.parametrize(
        ("text", "capability"),
        [
            ("Hey Jarvis, stop the music.", "stop"),
            ("stop", "stop"),
            ("Pause it please", "pause"),
            ("next song", "next"),
            ("play the next song", "next"),
            ("skip this one", "next"),
            ("Can you carry on?", "resume"),
            ("continue the music", "resume"),
        ],
    )
    def test_the_music_while_it_plays(self, text, capability):
        assert _call(text) == ("music", capability, {})

    def test_only_what_follows_the_wake_word_counts(self):
        assert _call("so anyway. Hey Jarvis, next song") == ("music", "next", {})

    def test_continue_works_once_paused(self):
        assert _call("continue", playing=False, paused=True) == ("music", "resume", {})

    @pytest.mark.parametrize(("text", "volume"), [("louder", 75), ("turn it up", 75), ("quieter", 45), ("turn the music down", 45)])
    def test_louder_and_quieter_move_from_where_it_is(self, text, volume):
        assert _call(text) == ("music", "set_volume", {"volume": volume})

    def test_louder_stops_at_the_top(self):
        assert _call("louder", volume=95) == ("music", "set_volume", {"volume": 100})

    @pytest.mark.parametrize(("text", "volume"), [("volume 30", 30), ("set the volume to 30 percent", 30), ("turn it up to 80", 80)])
    def test_a_level_said_is_the_level(self, text, volume):
        assert _call(text) == ("music", "set_volume", {"volume": volume})

    @pytest.mark.parametrize(("text", "capability"), [("turn off the fan", "turn_off"), ("Hey Jarvis switch on the ceiling fan", "turn_on"), ("fan off", "turn_off")])
    def test_a_named_device_on_or_off(self, text, capability):
        assert _call(text) == ("fan", capability, {})


class TestLeftToTheModel:
    @pytest.mark.parametrize(
        "text",
        [
            "stop the music in ten minutes",
            "play something by Queen",
            "turn it off",                      # which?
            "next",                             # nothing playing: see below
            "30",                               # a number is not a volume
            "stop the fan from making that noise when I sleep tonight",
            "what is playing",
        ],
    )
    def test_anything_more_than_the_command(self, text):
        assert _call(text, playing=False) is None

    def test_a_bare_stop_with_nothing_playing_may_be_about_something_else(self):
        assert _call("stop", playing=False, paused=False) is None

    def test_a_device_that_cannot_do_it(self):
        assert _call("pause the fan") is None

    def test_louder_without_knowing_how_loud(self):
        known = KnownDevices()
        known._on_manifest("", MUSIC)
        known._on_state("", DeviceState(ts=1.0, device_id="music", simulated=False, values={"playing": True}))
        assert quick_command("louder", known) is None

    def test_an_offline_device_is_not_meant(self):
        from src.common.devices import DeviceAvailability

        known = _known()
        known._on_availability("", DeviceAvailability(ts=1.0, device_id="fan", online=False))
        assert quick_command("turn off the fan", known) is None


class _NeverCalled:
    def __init__(self):
        self.chat = self

    @property
    def completions(self):
        raise AssertionError("a quick command must not reach the model")


class TestTheExtraction:
    def test_no_model_is_asked_and_the_call_rides_on_the_hint(self):
        hint = PersonalContext(CONFIG.reasoning, SimClock(), client=_NeverCalled(), devices=_known()).extract("Hey Jarvis, next song")
        assert hint.intent is Intent.SERVICE and hint.subject == "music" and hint.transcript == "Hey Jarvis, next song"
        assert (hint.command.device_id, hint.command.capability, dict(hint.command.arguments)) == ("music", "next", {})
