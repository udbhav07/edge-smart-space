"""The home robot's errands, walked over the house on a simulated clock (FR-100, FR-15)."""

import math
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.devices import DeviceClass, DeviceKind
from src.common.mqtt_client import Blackboard
from src.common.tools import ToolEffect, ToolInvocation, ToolRequester, ToolResult, ToolStatus
from src.devices.robot import build
from src.devices.service import DeviceService

CONFIG = load_config(Path("config/default.yaml"))
ROBOT = CONFIG.actuators.robot
DOCK = CONFIG.house.spots[ROBOT.dock_spot]


class _Robot:
    def __init__(self) -> None:
        self.clock = SimClock(start_epoch_s=1000.0)
        self.parts = build(CONFIG, self.clock)[0]
        transport = LoopbackTransport()
        self.blackboard = Blackboard(CONFIG.mqtt, transport)
        transport.attach(self.blackboard)
        self.proposed: list[ToolInvocation] = []
        self.blackboard.subscribe(topics.ASSIST_PROPOSED, ToolInvocation, lambda _t, invocation: self.proposed.append(invocation))
        self.parts.subscribe(self.blackboard)
        self.blackboard.start()
        self.service = DeviceService(self.parts.manifest, self.parts.handlers, self.parts.state, self.clock, self.blackboard)

    def ask(self, tool, **arguments) -> ToolResult:
        now = self.clock.now()
        return self.service.handle(ToolInvocation(ts=now, invocation_id=f"i{now}", tool=tool, arguments=arguments, requester=ToolRequester.CONSOLE, expires_ts=now + 60))

    def run(self, seconds: float, step: float = 1.0) -> None:
        for _ in range(int(seconds / step)):
            self.clock.advance(step)
            self.parts.tick(self.service)

    @property
    def state(self):
        return self.parts.state()


@pytest.fixture(name="robot")
def _robot():
    return _Robot()


def _at(state, x, y) -> bool:
    return math.dist((state["x"], state["y"]), (x, y)) < 0.05


class TestManifest:
    def test_it_is_a_simulated_robot(self, robot):
        assert robot.parts.manifest.kind is DeviceKind.ROBOT and robot.parts.manifest.simulated

    def test_nothing_it_does_is_plant(self, robot):
        assert all(capability.device_class is DeviceClass.COMFORT for capability in robot.parts.manifest.capabilities)

    def test_status_is_a_read(self, robot):
        assert robot.parts.manifest.capability("status").effect is ToolEffect.READ

    def test_go_to_offers_the_houses_rooms(self, robot):
        (room,) = robot.parts.manifest.capability("go_to").parameters
        assert set(room.choices) == {r.room_id for r in CONFIG.house.rooms}


class TestFeedingTheDog:
    def test_it_answers_at_once_with_how_long(self, robot):
        result = robot.ask("feed_the_dog")
        assert result.status is ToolStatus.OK and result.message.startswith("On my way to feed the dog, about") and result.message.endswith("(simulated).")

    def test_it_walks_through_the_kitchen_door(self, robot):
        robot.ask("feed_the_dog")
        door = next(d for d in CONFIG.house.doors if set(d.between) == {"living_room", "kitchen"})
        legs = set()
        for _ in range(30):
            robot.run(1)
            legs.add((robot.state["leg_to_x"], robot.state["leg_to_y"]))
        assert (door.x, door.y) in legs

    def test_it_fills_the_bowl_and_comes_home(self, robot):
        robot.ask("feed_the_dog")
        robot.run(60)
        bowl = CONFIG.house.spots["dog_bowl"]
        assert robot.state["bowl_full"] is True and robot.state["docked"] is True and robot.state["task"] == ""
        assert not _at(robot.state, bowl.x, bowl.y)

    def test_it_is_at_the_bowl_while_feeding(self, robot):
        robot.ask("feed_the_dog")
        bowl = CONFIG.house.spots["dog_bowl"]
        phases = []
        for _ in range(60):
            robot.run(1)
            if robot.state["phase"] == "feeding":
                phases.append(_at(robot.state, bowl.x, bowl.y))
        assert phases and all(phases)

    def test_the_bowl_empties_again_for_the_next_demo(self, robot):
        robot.ask("feed_the_dog")
        robot.run(60 + ROBOT.bowl_empties_after_s)
        assert robot.state["bowl_full"] is False


class TestOneErrandAtATime:
    def test_a_second_errand_is_refused_in_words(self, robot):
        robot.ask("feed_the_dog")
        robot.run(3)
        result = robot.ask("clean_the_bathroom")
        assert result.status is ToolStatus.FAILED and "I'm feeding the dog" in result.message

    def test_stop_stops_it_where_it_is(self, robot):
        robot.ask("feed_the_dog")
        robot.run(4)
        where = (robot.state["x"], robot.state["y"])
        assert robot.ask("stop").message == "Stopped feeding the dog (simulated)."
        robot.run(10)
        assert (robot.state["x"], robot.state["y"]) == where and robot.state["task"] == ""

    def test_after_stop_it_takes_a_new_errand(self, robot):
        robot.ask("feed_the_dog")
        robot.run(4)
        robot.ask("stop")
        robot.run(3)
        assert robot.ask("return_to_dock").status is ToolStatus.OK


class TestCleaningTheBathroom:
    def test_it_sweeps_inside_the_bathroom(self, robot):
        robot.ask("clean_the_bathroom")
        bathroom = CONFIG.house.room("bathroom")
        cleaning = []
        for _ in range(120):
            robot.run(1)
            if robot.state["phase"] == "cleaning":
                cleaning.append(bathroom.contains(robot.state["x"], robot.state["y"]))
        assert len(cleaning) > 10 and all(cleaning)

    def test_it_is_marked_clean_and_it_docks(self, robot):
        robot.ask("clean_the_bathroom")
        robot.run(150)
        assert robot.state["bathroom_cleaned_ts"] > 0 and robot.state["docked"] is True


class TestSwitchingOnTheTv:
    def test_it_asks_the_executor_once_as_a_remote_would(self, robot):
        robot.ask("switch_on_tv")
        robot.run(60)
        (asked,) = robot.proposed
        assert asked.tool == "tv_turn_on" and asked.requester is ToolRequester.DEVICE

    def test_it_never_asks_before_reaching_the_tv(self, robot):
        robot.ask("switch_on_tv")
        robot.run(3)
        assert robot.proposed == []

    def test_it_repeats_the_tvs_own_answer(self, robot):
        robot.ask("switch_on_tv")
        while not robot.proposed:
            robot.run(1)
        (asked,) = robot.proposed
        robot.blackboard.publish(topics.ASSIST_RESULT, ToolResult(ts=robot.clock.now(), invocation_id=asked.invocation_id, tool=asked.tool, status=ToolStatus.OK, message="The TV is on, channel 1 (simulated).", provider="tv"))
        assert robot.state["last_said"] == "The TV is on, channel 1 (simulated)."

    def test_someone_elses_answer_is_not_taken_for_its_own(self, robot):
        robot.ask("switch_on_tv")
        robot.run(60)
        robot.blackboard.publish(topics.ASSIST_RESULT, ToolResult(ts=robot.clock.now(), invocation_id="other", tool="tv_turn_on", status=ToolStatus.OK, message="not mine", provider="tv"))
        assert robot.state["last_said"] != "not mine"

    def test_silence_is_reported(self, robot):
        robot.ask("switch_on_tv")
        robot.run(60 + ROBOT.result_wait_s)
        assert robot.state["last_said"] == "The TV did not answer."


class TestGoingSomewhere:
    def test_it_waits_in_the_room_it_was_sent_to(self, robot):
        robot.ask("go_to", room="bedroom")
        robot.run(60)
        assert robot.state["room"] == "bedroom" and robot.state["task"] == "" and robot.state["docked"] is False

    def test_a_room_the_house_lacks_is_refused(self, robot):
        assert robot.ask("go_to", room="garage").status is ToolStatus.BAD_ARGUMENTS

    def test_sent_home_it_docks(self, robot):
        robot.ask("go_to", room="bedroom")
        robot.run(60)
        robot.ask("return_to_dock")
        robot.run(60)
        assert robot.state["docked"] is True

    def test_on_the_dock_already_it_says_so(self, robot):
        assert "already on my dock" in robot.ask("return_to_dock").message


class TestBattery:
    def test_walking_drains_it(self, robot):
        robot.ask("go_to", room="bathroom")
        robot.run(10)
        assert robot.state["battery"] < 100

    def test_the_dock_charges_it(self, robot):
        robot.ask("go_to", room="bathroom")
        robot.run(60)
        low = robot.state["battery"]
        robot.ask("return_to_dock")
        robot.run(120)
        assert robot.state["battery"] > low

    def test_low_it_refuses_and_goes_to_charge(self, robot):
        robot.ask("go_to", room="bedroom")
        robot.run(60)
        robot.parts.tick.__self__.battery = ROBOT.low_battery_pct - 1
        result = robot.ask("feed_the_dog")
        assert result.status is ToolStatus.FAILED and "battery is low" in result.message
        robot.run(60)
        assert robot.state["docked"] is True


class TestStatus:
    def test_on_the_dock(self, robot):
        assert robot.ask("status").message == "I'm on my dock, battery 100% (simulated)."

    def test_during_an_errand(self, robot):
        robot.ask("feed_the_dog")
        robot.run(3)
        assert robot.ask("status").message.startswith("I'm feeding the dog, in the")


class TestTheLegForTheConsole:
    def test_a_walk_publishes_where_the_leg_ends_and_when(self, robot):
        robot.ask("feed_the_dog")
        robot.run(1)
        state = robot.state
        assert state["leg_end_ts"] > state["leg_start_ts"] > 0 and state["phase"] == "walking"

    def test_at_rest_there_is_no_leg(self, robot):
        assert robot.state["leg_end_ts"] == 0.0
