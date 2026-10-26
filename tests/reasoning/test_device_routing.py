"""Routing a request to one device, for a 7B model (FR-89).

What matters: the extraction is shown one line per device and never their
tools; a request naming a device is filed as being about it, and can only be
moved away from the plant by that; the assistant is shown that device's tools
and no others, however many devices are plugged in; and a request about the
fan really turns the fan on, end to end.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_service as build_executor
from src.assistance.devices import DeviceDirectory
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.devices import DeviceAvailability, DeviceCapability, DeviceClass, DeviceKind, DeviceManifest, air_conditioner_manifest
from src.common.mqtt_client import Blackboard
from src.common.schemas import Comfort, Intent, PreferenceHint
from src.common.tools import ToolCatalogue, ToolEffect, ToolSpec
from src.devices.fan import build as build_fan
from src.devices.service import DeviceService
from src.reasoning.__main__ import build_assistant
from src.reasoning.devices import KnownDevices
from src.reasoning.single_shot import PersonalContext, routed
from tests.reasoning.stubs import ScriptedChat, call, calling

CONFIG = load_config(Path("config/default.yaml"))


def _manifest(device_id, name, *aliases, plant=False, capabilities=("turn_on",)):
    device_class = DeviceClass.PLANT if plant else DeviceClass.COMFORT
    return DeviceManifest(
        ts=1.0,
        device_id=device_id,
        name=name,
        kind=DeviceKind.OTHER,
        aliases=aliases,
        simulated=True,
        capabilities=tuple(DeviceCapability(name=c, purpose=f"{c} it.", effect=ToolEffect.WRITE, device_class=device_class) for c in capabilities),
    )


def _known(*manifests, offline=()):
    known = KnownDevices()
    for manifest in manifests:
        known._on_manifest("", manifest)
    for device_id in offline:
        known._on_availability("", DeviceAvailability(ts=1.0, device_id=device_id, online=False))
    return known


def _hint(intent=Intent.ENVIRONMENT, **fields):
    return PreferenceHint(ts=1.0, intent=intent, comfort=fields.pop("comfort", Comfort.UNCHANGED), **fields)


class TestKnownDevices:
    def test_one_line_per_device_with_its_aliases(self):
        known = _known(_manifest("fan", "Ceiling fan", "the fan"))
        assert known.lines() == ["- fan: Ceiling fan (also: the fan)"]

    def test_an_offline_device_is_not_listed(self):
        assert _known(_manifest("fan", "Ceiling fan"), offline=("fan",)).lines() == []

    def test_the_air_conditioner_is_never_listed(self):
        assert _known(air_conditioner_manifest(1.0, True)).lines() == []

    def test_named_by_alias_in_whole_words(self):
        known = _known(_manifest("fan", "Ceiling fan", "the fan"))
        assert known.named_in("could you turn the fan on") == "fan"
        assert known.named_in("that was fantastic") is None

    def test_the_longest_name_said_wins(self):
        known = _known(_manifest("lamp", "Lamp"), _manifest("desk_lamp", "Desk lamp"))
        assert known.named_in("switch off the desk lamp") == "desk_lamp"

    def test_the_id_is_a_name_too(self):
        assert _known(_manifest("desk_lamp", "Reading light")).named_in("desk lamp off") == "desk_lamp"


class TestRouting:
    def test_naming_a_device_files_the_request_about_it(self):
        hint = routed(_hint(Intent.QUESTION, subject="room"), "is the fan on?", _known(_manifest("fan", "Ceiling fan")))
        assert (hint.intent, hint.subject) == (Intent.SERVICE, "fan")

    def test_a_number_said_about_a_device_is_never_a_temperature(self):
        """'Set the fan to 3' must not reach the goal path as 3 degrees."""
        hint = routed(_hint(target_c=3.0, subject="temperature"), "set the fan to 3", _known(_manifest("fan", "Ceiling fan")))
        assert hint.target_c is None and hint.intent is Intent.SERVICE

    def test_a_memory_request_stays_one(self):
        hint = _hint(Intent.SERVICE, subject="memory")
        assert routed(hint, "remember that I like the fan on", _known(_manifest("fan", "Ceiling fan"))) == hint

    def test_no_device_named_leaves_the_request_alone(self):
        hint = _hint(comfort=Comfort.COOLER, subject="temperature")
        assert routed(hint, "it's too warm", _known(_manifest("fan", "Ceiling fan"))) == hint

    def test_naming_the_ac_is_still_about_the_temperature(self):
        hint = _hint(target_c=23.0, subject="temperature")
        assert routed(hint, "set the ac to 23", _known(air_conditioner_manifest(1.0, True))) == hint

    def test_without_a_device_list_nothing_changes(self):
        hint = _hint(target_c=3.0)
        assert routed(hint, "set the fan to 3", None) == hint


class _Client:
    def __init__(self, content):
        self.content, self.calls = content, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class TestTheExtractionIsShownTheDevices:
    REPLY = json.dumps({"intent": "service", "subject": "fan", "comfort": "unchanged", "target_c": None, "rationale": "fan on"})

    def test_the_devices_are_listed_one_line_each(self):
        client = _Client(self.REPLY)
        PersonalContext(CONFIG.reasoning, SimClock(), client=client, devices=_known(_manifest("fan", "Ceiling fan", "the fan"))).extract("turn on the fan")
        prompt = client.calls[0]["messages"][0]["content"]
        assert "- fan: Ceiling fan (also: the fan)" in prompt and "tools" not in client.calls[0]

    def test_no_devices_no_list(self):
        client = _Client(self.REPLY)
        PersonalContext(CONFIG.reasoning, SimClock(), client=client, devices=KnownDevices()).extract("turn on the fan")
        assert "Devices in the room" not in client.calls[0]["messages"][0]["content"]

    def test_the_routing_rule_runs_on_the_extraction(self):
        wrong = json.dumps({"intent": "environment", "subject": "temperature", "comfort": "unchanged", "target_c": 3, "rationale": "x"})
        hint = PersonalContext(CONFIG.reasoning, SimClock(), client=_Client(wrong), devices=_known(_manifest("fan", "Ceiling fan"))).extract("set the fan to 3")
        assert (hint.intent, hint.subject, hint.target_c) == (Intent.SERVICE, "fan", None)


def _catalogue(*device_ids, per_device=5):
    specs = [
        ToolSpec(name=f"{device_id}_do{n}", purpose=f"{device_id} {n}", effect=ToolEffect.WRITE, device=device_id)
        for device_id in device_ids
        for n in range(per_device)
    ]
    return ToolCatalogue(ts=1.0, tools=(ToolSpec(name="get_events", purpose="Calendar.", effect=ToolEffect.READ), *specs))


class TestTheAssistantIsShownOneDevice:
    def _assistant(self, catalogue, max_device_tools=6):
        assistant = build_assistant(CONFIG, SimClock(), Blackboard(CONFIG.mqtt, LoopbackTransport()), ScriptedChat())
        assistant._max_device_tools = max_device_tools
        assistant._on_catalogue("", catalogue)
        return assistant

    def test_only_the_named_devices_tools(self):
        assistant = self._assistant(_catalogue(*(f"dev{n}" for n in range(20))))
        names = {tool["function"]["name"] for tool in assistant._tools_for(_hint(Intent.SERVICE, subject="dev7"))}
        assert names == {f"dev7_do{n}" for n in range(5)}

    def test_never_more_than_the_cap(self):
        assistant = self._assistant(_catalogue("big", per_device=12), max_device_tools=6)
        assert len(assistant._tools_for(_hint(Intent.SERVICE, subject="big"))) == 6

    def test_a_request_about_no_device_sees_no_device_tools(self):
        assistant = self._assistant(_catalogue(*(f"dev{n}" for n in range(20))))
        names = {tool["function"]["name"] for tool in assistant._tools_for(_hint(Intent.SERVICE, subject="calendar"))}
        assert names == {"get_events"}

    def test_the_context_does_not_grow_with_the_room(self):
        small = self._assistant(_catalogue("dev0"))._tools_for(_hint(Intent.SERVICE, subject="dev0"))
        large = self._assistant(_catalogue(*(f"dev{n}" for n in range(50))))._tools_for(_hint(Intent.SERVICE, subject="dev0"))
        assert json.dumps(small) == json.dumps(large)


class TestEndToEnd:
    def test_turn_on_the_fan(self, tmp_path):
        config = CONFIG.model_copy(update={"assistance": CONFIG.assistance.model_copy(update={"calendar_path": str(tmp_path / "c.json"), "result_timeout_s": 1.0})})
        clock = SimClock(start_epoch_s=1_790_000_000.0)
        transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, transport) for _ in range(3)]
        for board in boards:
            transport.attach(board)
        executor = build_executor(config, clock, boards[0])
        executor.subscribe()
        DeviceDirectory(clock, boards[0], executor.registry, 1.0, on_change=executor.publish_catalogue).subscribe()
        chat = ScriptedChat(calling(call("fan_turn_on")))
        assistant = build_assistant(config, clock, boards[1], chat, KnownDevices())
        assistant.subscribe()
        fan = build_fan(config, clock)
        DeviceService(fan.manifest, fan.handlers, fan.state, clock, boards[2]).start()

        boards[2].publish(topics.CONTEXT_PREFERENCE, PreferenceHint(ts=clock.now(), intent=Intent.SERVICE, comfort=Comfort.UNCHANGED, subject="fan", transcript="turn on the fan"))
        (reply,) = assistant.process_pending()

        assert fan.state()["on"] is True and "fan is on" in reply.reply
        shown = {tool["function"]["name"] for tool in chat.requests[0][1]}
        assert shown == {"fan_turn_on", "fan_turn_off", "fan_set_speed", "fan_status"}
        assert "ceiling fan" in chat.requests[0][0][0]["content"]
