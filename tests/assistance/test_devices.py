"""Devices as tools (FR-87, FR-88): discovered from the board, run through the
registry's gates, withdrawn when they go away, and never the thermal plant.

The fan service, the directory and the executor share one in-process
blackboard, so a call travels the real path: proposed -> registry -> device
command -> device result -> assist result.
"""

import queue
import threading
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.devices import DeviceDirectory
from src.assistance.executor import AssistanceExecutor, Worker
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.devices import DeviceAvailability, air_conditioner_manifest
from src.common.mqtt_client import Blackboard
from src.common.tools import ToolCatalogue, ToolInvocation, ToolRegistry, ToolRequester, ToolResult, ToolStatus
from src.devices.fan import build as build_fan
from src.devices.service import DeviceService

CONFIG = load_config(Path("config/default.yaml"))


class _Room:
    def __init__(self, timeout_s=2.0):
        self.clock = SimClock(start_epoch_s=1000.0)
        self.transport = LoopbackTransport()
        self.board = Blackboard(CONFIG.mqtt, self.transport)
        self.transport.attach(self.board)
        self.registry = ToolRegistry(self.clock)
        self.executor = AssistanceExecutor(self.clock, self.board, self.registry)
        self.executor.subscribe()
        self.directory = DeviceDirectory(self.clock, self.board, self.registry, timeout_s, on_change=self.executor.publish_catalogue)
        self.directory.subscribe()
        self.fan_parts = build_fan(CONFIG, self.clock)
        self.fan = DeviceService(self.fan_parts.manifest, self.fan_parts.handlers, self.fan_parts.state, self.clock, self.board)

    def ask(self, tool, **arguments):
        self.clock.advance(5.0)
        invocation = ToolInvocation(ts=self.clock.now(), invocation_id=f"a{self.clock.now()}", tool=tool, arguments=arguments, requester=ToolRequester.CONSOLE, expires_ts=self.clock.now() + 60)
        self.board.publish(topics.ASSIST_PROPOSED, invocation)
        payloads = [p for t, p, _, _ in self.transport.published if t == topics.ASSIST_RESULT.pattern]
        return ToolResult.model_validate_json(payloads[-1])

    def catalogue(self):
        payloads = [p for t, p, _, _ in self.transport.published if t == topics.ASSIST_CATALOGUE.pattern]
        return ToolCatalogue.model_validate_json(payloads[-1]) if payloads else None

    def offline(self, device_id="fan"):
        self.board.publish(topics.DEVICE_AVAILABILITY, DeviceAvailability(ts=self.clock.now(), device_id=device_id, online=False), device_id=device_id)


@pytest.fixture(name="room")
def _room():
    room = _Room()
    room.fan.start()
    return room


class TestDiscovery:
    def test_an_online_device_becomes_tools(self, room):
        assert {"fan_turn_on", "fan_turn_off", "fan_set_speed", "fan_status"} <= set(room.registry.names)

    def test_each_tool_says_which_device_it_reaches(self, room):
        assert room.registry.spec("fan_set_speed").device == "fan"

    def test_the_tool_is_described_with_the_devices_name(self, room):
        assert room.registry.spec("fan_turn_on").purpose.startswith("Ceiling fan: ")

    def test_the_catalogue_is_republished_with_the_device(self, room):
        assert "fan_set_speed" in {spec.name for spec in room.catalogue().tools}

    def test_offline_withdraws_the_tools_and_republishes(self, room):
        room.offline()
        assert not any(name.startswith("fan_") for name in room.registry.names)
        assert not any(spec.name.startswith("fan_") for spec in room.catalogue().tools)

    def test_back_online_declares_them_again(self, room):
        room.offline()
        room.fan.start()
        assert "fan_turn_on" in room.registry.names

    def test_the_air_conditioner_is_never_a_tool(self, room):
        room.board.publish(topics.DEVICE_MANIFEST, air_conditioner_manifest(room.clock.now(), True), device_id="ac")
        assert not any(name.startswith("ac_") for name in room.registry.names)

    def test_a_name_already_taken_is_skipped_not_fatal(self):
        room = _Room()
        from src.common.tools import ToolEffect, ToolSpec

        room.registry.declare(ToolSpec(name="fan_status", purpose="something else", effect=ToolEffect.READ))
        room.fan.start()
        assert room.registry.spec("fan_status").purpose == "something else" and "fan_turn_on" in room.registry.names


class TestCalling:
    def test_a_call_travels_to_the_device_and_back(self, room):
        result = room.ask("fan_set_speed", speed=2)
        assert result.status is ToolStatus.OK and result.provider == "fan" and result.simulated is True
        assert room.fan_parts.state()["speed"] == 2

    def test_out_of_bounds_is_refused_before_the_device_is_asked(self, room):
        result = room.ask("fan_set_speed", speed=9)
        assert result.status is ToolStatus.BAD_ARGUMENTS
        assert room.transport.count_on("space/device/fan/command") == 0

    def test_the_device_refusing_is_said_in_its_words(self, room):
        """Twice at once is inside the fan's switching interval; the device says so."""
        for invocation_id in ("first", "again"):
            invocation = ToolInvocation(ts=room.clock.now(), invocation_id=invocation_id, tool="fan_turn_on", arguments={}, requester=ToolRequester.CONSOLE, expires_ts=room.clock.now() + 60)
            room.board.publish(topics.ASSIST_PROPOSED, invocation)
        result = ToolResult.model_validate_json([p for t, p, _, _ in room.transport.published if t == topics.ASSIST_RESULT.pattern][-1])
        assert result.status is ToolStatus.FAILED and "too soon" in result.message

    def test_a_device_that_never_answers_is_said_plainly(self):
        room = _Room(timeout_s=0.05)
        room.board.publish(topics.DEVICE_MANIFEST, room.fan_parts.manifest, device_id="fan")  # announced, but nobody serving
        result = room.ask("fan_turn_on")
        assert result.status is ToolStatus.FAILED and "didn't answer" in result.message

    def test_after_going_offline_the_tool_is_unknown(self, room):
        room.offline()
        assert room.ask("fan_turn_on").status is ToolStatus.UNKNOWN_TOOL

    def test_the_command_names_the_executor_as_its_requester(self, room):
        room.ask("fan_turn_on")
        command = ToolInvocation.model_validate_json([p for t, p, _, _ in room.transport.published if t == "space/device/fan/command"][-1])
        assert command.requester is ToolRequester.ASSISTANCE and command.tool == "turn_on"


class TestWorker:
    def test_jobs_run_off_the_calling_thread_in_order(self):
        worker = Worker()
        worker.start()
        seen, done = [], threading.Event()
        for n in range(3):
            worker(lambda n=n: seen.append((n, threading.current_thread().name)))
        worker(done.set)
        assert done.wait(5) and [n for n, _ in seen] == [0, 1, 2]
        assert all(name == "assistance-worker" for _, name in seen)

    def test_a_crashing_job_does_not_stop_the_worker(self):
        worker = Worker()
        worker.start()
        done = threading.Event()
        worker(lambda: 1 / 0)
        worker(done.set)
        assert done.wait(5)

    def test_a_flood_is_refused_with_a_result(self):
        clock, transport = SimClock(start_epoch_s=1000.0), LoopbackTransport()
        board = Blackboard(CONFIG.mqtt, transport)
        transport.attach(board)

        def full(_job):
            raise queue.Full

        executor = AssistanceExecutor(clock, board, ToolRegistry(clock), dispatch=full)
        executor.subscribe()
        board.publish(topics.ASSIST_PROPOSED, ToolInvocation(ts=clock.now(), invocation_id="x", tool="get_events", arguments={}, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + 60))
        result = ToolResult.model_validate_json([p for t, p, _, _ in transport.published if t == topics.ASSIST_RESULT.pattern][-1])
        assert result.status is ToolStatus.UNAVAILABLE and "try again" in result.message


class TestRegistryWithdraw:
    def test_withdrawing_an_unknown_tool_says_so(self):
        assert ToolRegistry(SimClock()).withdraw("nothing") is False

    def test_withdrawing_removes_the_provider_too(self):
        registry = ToolRegistry(SimClock())
        registry.withdraw("get_events")
        assert "get_events" not in registry.names
