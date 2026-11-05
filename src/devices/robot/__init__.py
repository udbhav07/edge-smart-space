"""A home robot with scripted errands, as a device service (FR-100, FR-15).

It feeds the dog, cleans the bathroom, switches on the TV, and goes where it
is sent. None of it is real, and every sentence says so: an errand is a walk
over the floor plan in ``house`` and a pause where the work would be. That is
enough for the house to be drawn with the robot crossing it, and it is the
whole of the stub -- a real robot would replace the walking, not the contract.

**An errand answers at once and runs on.** A command has 12 s to be answered,
and feeding the dog takes half a minute, so the handler says "on my way" and
the robot's progress is its published state: where it is, where its current
leg of the walk ends and when, so the console can move it smoothly between
ticks. Time is the injected clock's.

**It uses the TV the way a person does.** ``switch_on_tv`` walks to the TV,
then asks for ``tv_turn_on`` on ``space/assist/proposed`` -- exactly what a
remote's tap publishes -- so the executor's checks and audit apply to the
robot as to anyone, and it never touches the TV's command topic. The TV's own
answer, from ``space/assist/result``, is what the robot then reports.

Handlers run on the network thread and ``tick`` on the runner's, so the
robot's state is held under a lock.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config, HouseConfig, RobotConfig
from src.common.devices import TOOL_JOINER, DeviceCapability, DeviceClass, DeviceKind, DeviceManifest
from src.common.mqtt_client import Blackboard
from src.common.tools import (
    ArgumentValue,
    ParameterType,
    ProviderOutcome,
    ToolEffect,
    ToolInvocation,
    ToolParameter,
    ToolRequester,
    ToolResult,
    ToolStatus,
)
from src.devices.runner import DeviceParts
from src.devices.service import DeviceService

LOGGER = logging.getLogger(__name__)

#: Close enough to a spot to be on it, in m.
_ON_SPOT_M = 0.05
#: A full battery, in percent.
_FULL = 100.0

Point = tuple[float, float]


@dataclass(frozen=True)
class _Step:
    """One part of an errand: a walk to a point, or work in place for a while."""

    phase: str
    duration_s: float
    to: Point | None = None
    done: Callable[[], None] | None = None


class HomeRobot:
    """The robot's place, battery and errand, advanced by the clock."""

    def __init__(self, config: RobotConfig, house: HouseConfig, clock: Clock) -> None:
        self._config = config
        self._house = house
        self._clock = clock
        self._lock = threading.RLock()
        self._blackboard: Blackboard | None = None
        dock = house.spots[config.dock_spot]
        self._dock: Point = (dock.x, dock.y)
        self.x, self.y = self._dock
        self.heading = 0.0
        self.battery = _FULL
        self._steps: deque[_Step] = deque()
        self._step_started = 0.0
        self._leg_from: Point = self._dock
        self._task = ""
        self._doing = ""
        self._task_started = 0.0
        self._task_total_s = 0.0
        self._bowl_filled_ts = 0.0
        self._cleaned_ts = 0.0
        self._last_said = ""
        self._asked: tuple[str, float] | None = None
        self._last_tick = clock.now()
        self._changed = True

    # --- what it can be asked ------------------------------------------------

    def feed_the_dog(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        bowl = self._spot("dog_bowl")
        return self._start(
            "feed_the_dog",
            "feeding the dog",
            [*self._walk(bowl), _Step("feeding", self._config.feed_s, done=self._fill_bowl), *self._walk(self._dock, start=bowl)],
            "On my way to feed the dog",
        )

    def clean_the_bathroom(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        sweep = self._sweep("bathroom")
        steps = [*self._walk(sweep[0])]
        for here, there in zip(sweep, sweep[1:]):
            steps.append(_Step("cleaning", math.dist(here, there) / self._config.speed_m_s, to=there))
        steps.append(_Step("cleaning", 0.0, done=self._cleaned))
        steps.extend(self._walk(self._dock, start=sweep[-1]))
        return self._start("clean_the_bathroom", "cleaning the bathroom", steps, "Cleaning the bathroom")

    def switch_on_tv(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        tv = self._spot("tv_front")
        return self._start(
            "switch_on_tv",
            "switching on the TV",
            [*self._walk(tv), _Step("pressing", self._config.press_s, done=self._press_tv), *self._walk(self._dock, start=tv)],
            "Going to switch on the TV",
        )

    def go_to(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        room = self._house.room(str(arguments["room"]))
        if room is None:
            raise ValueError(f"There is no {arguments['room']} in the house.")
        name = room.name.lower()
        return self._start("go_to", f"going to the {name}", self._walk(room.centre), f"Going to the {name}")

    def return_to_dock(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            if self._at(self._dock) and not self._steps:
                return ProviderOutcome(message=f"I'm already on my dock, battery {self.battery:.0f}% (simulated).", detail=self._values())
            self._begin("return_to_dock", "going back to my dock", self._walk(self._dock))
            return ProviderOutcome(message="Going back to my dock (simulated).", detail=self._values())

    def stop(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            self._advance(self._clock.now())
            was = self._doing
            self._steps.clear()
            self._task = self._doing = ""
            self._changed = True
            said = f"Stopped {was} (simulated)." if was else "I wasn't doing anything."
            return ProviderOutcome(message=said, detail=self._values())

    def status(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            self._advance(self._clock.now())
            where = self._room_name()
            if self._doing:
                said = f"I'm {self._doing}, in the {where}; battery {self.battery:.0f}%"
            elif self._at(self._dock):
                said = f"I'm on my dock, battery {self.battery:.0f}%"
            else:
                said = f"I'm in the {where}, battery {self.battery:.0f}%"
            return ProviderOutcome(message=f"{said} (simulated).", detail=self._values())

    # --- the clock and the board ---------------------------------------------

    def follow(self, blackboard: Blackboard) -> None:
        """Hear other devices' answers to what the robot asked of them."""
        self._blackboard = blackboard
        blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)

    def tick(self, service: DeviceService) -> None:
        """Walk on, drain or charge the battery, and publish what changed."""
        with self._lock:
            now = self._clock.now()
            before = round(self.battery)
            active = bool(self._steps)
            self._charge(now)
            self._advance(now)
            if self._asked is not None and now - self._asked[1] > self._config.result_wait_s:
                self._asked = None
                self._say("The TV did not answer.")
            if self._bowl_filled_ts and now - self._bowl_filled_ts > self._config.bowl_empties_after_s:
                self._bowl_filled_ts = 0.0
                self._changed = True
            publish = active or self._changed or round(self.battery) != before
            self._changed = False
        if publish:
            service.publish_state()

    def state(self) -> dict[str, ArgumentValue]:
        with self._lock:
            return self._values()

    # --- inside ----------------------------------------------------------------

    def _start(self, task: str, doing: str, steps: list[_Step], said: str) -> ProviderOutcome:
        with self._lock:
            self._advance(self._clock.now())
            if self._doing:
                raise ValueError(f"I'm {self._doing}; ask again when I'm done, or say stop.")
            if self.battery < self._config.low_battery_pct:
                if not self._at(self._dock):
                    self._begin("return_to_dock", "going back to my dock", self._walk(self._dock))
                raise ValueError("My battery is low; I'm going to charge.")
            self._begin(task, doing, steps)
            return ProviderOutcome(message=f"{said}, about {self._rounded(self._task_total_s)} (simulated).", detail=self._values())

    def _begin(self, task: str, doing: str, steps: list[_Step]) -> None:
        now = self._clock.now()
        self._steps = deque(steps)
        self._task, self._doing = task, doing
        self._task_started = self._step_started = now
        self._task_total_s = sum(step.duration_s for step in steps)
        self._leg_from = (self.x, self.y)
        self._changed = True

    def _advance(self, now: float) -> None:
        """Move through the errand's steps up to now, carrying the time over between them."""
        while self._steps:
            step = self._steps[0]
            elapsed = now - self._step_started
            if step.to is not None:
                share = 1.0 if step.duration_s <= 0 else min(1.0, elapsed / step.duration_s)
                (x0, y0), (x1, y1) = self._leg_from, step.to
                if (x1, y1) != (x0, y0):
                    self.heading = math.degrees(math.atan2(y1 - y0, x1 - x0))
                self.x, self.y = x0 + (x1 - x0) * share, y0 + (y1 - y0) * share
            if elapsed < step.duration_s:
                return
            self._steps.popleft()
            self._step_started += step.duration_s
            self._leg_from = (self.x, self.y)
            self._changed = True
            if step.done is not None:
                step.done()
        if self._doing:
            LOGGER.info("robot finished %s", self._doing)
            self._task = self._doing = ""
            self._changed = True

    def _charge(self, now: float) -> None:
        elapsed, self._last_tick = max(0.0, now - self._last_tick), now
        phase = self._phase()
        if phase == "walking" or phase == "cleaning":
            self.battery -= self._config.battery_drain_moving_pct_s * elapsed
        elif self._steps:
            self.battery -= self._config.battery_drain_working_pct_s * elapsed
        elif self._at(self._dock):
            self.battery += self._config.battery_charge_pct_s * elapsed
        self.battery = max(0.0, min(_FULL, self.battery))

    def _walk(self, to: Point, start: Point | None = None) -> list[_Step]:
        """The legs from a point (here, if none) to another, through the house's doors."""
        here = start if start is not None else (self.x, self.y)
        points = [*self._doors_between(here, to), to]
        steps, previous = [], here
        for point in points:
            steps.append(_Step("walking", math.dist(previous, point) / self._config.speed_m_s, to=point))
            previous = point
        return steps

    def _doors_between(self, here: Point, there: Point) -> list[Point]:
        start, goal = self._house.room_at(*here), self._house.room_at(*there)
        if start is None or goal is None or start.room_id == goal.room_id:
            return []
        came_from: dict[str, tuple[str, Point] | None] = {start.room_id: None}
        frontier = deque([start.room_id])
        while frontier:
            room = frontier.popleft()
            if room == goal.room_id:
                break
            for door in self._house.doors:
                if room in door.between:
                    other = door.between[1] if door.between[0] == room else door.between[0]
                    if other not in came_from:
                        came_from[other] = (room, (door.x, door.y))
                        frontier.append(other)
        doors, room = [], goal.room_id
        while came_from.get(room) is not None:
            room, door = came_from[room]  # type: ignore[misc]
            doors.append(door)
        return list(reversed(doors))

    def _sweep(self, room_id: str) -> list[Point]:
        """Back-and-forth passes over a room's floor, inside its margin."""
        room = self._house.room(room_id)
        assert room is not None  # the configuration requires it
        margin, spacing = self._config.sweep_margin_m, self._config.sweep_spacing_m
        left, right = room.x + margin, room.x + room.width - margin
        top, bottom = room.y + margin, room.y + room.depth - margin
        points, y, forward = [], top, True
        while y <= bottom + 1e-9:
            points.extend([(left, y), (right, y)] if forward else [(right, y), (left, y)])
            y, forward = y + spacing, not forward
        return points

    def _spot(self, name: str) -> Point:
        spot = self._house.spots[name]
        return (spot.x, spot.y)

    def _fill_bowl(self) -> None:
        self._bowl_filled_ts = self._clock.now()
        self._say("The dog's bowl is full.")

    def _cleaned(self) -> None:
        self._cleaned_ts = self._clock.now()
        self._say("The bathroom is clean.")

    def _press_tv(self) -> None:
        tool = f"{self._config.tv_device_id}{TOOL_JOINER}turn_on"
        if self._blackboard is None:
            self._say("I pressed the TV's button, but nothing heard me.")
            return
        now = self._clock.now()
        invocation = ToolInvocation(
            ts=now,
            invocation_id=f"{self._config.device_id}-{now:.3f}",
            tool=tool,
            arguments={},
            requester=ToolRequester.DEVICE,
            rationale=f"The {self._config.name.lower()} pressed the TV's power button.",
            expires_ts=now + self._config.result_wait_s,
        )
        self._asked = (invocation.invocation_id, now)
        self._say("Pressing the TV's power button.")
        self._blackboard.publish(topics.ASSIST_PROPOSED, invocation)

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        with self._lock:
            if self._asked is None or result.invocation_id != self._asked[0]:
                return
            self._asked = None
            said = result.message if result.status is ToolStatus.OK else f"The TV didn't come on: {result.message}"
            self._say(said)

    def _say(self, sentence: str) -> None:
        self._last_said = sentence
        self._changed = True

    def _phase(self) -> str:
        if self._steps:
            return self._steps[0].phase
        if self._at(self._dock):
            return "charging" if self.battery < _FULL else "docked"
        return "idle"

    def _at(self, point: Point) -> bool:
        return math.dist((self.x, self.y), point) <= _ON_SPOT_M

    def _room_name(self) -> str:
        room = self._house.room_at(self.x, self.y)
        return room.name.lower() if room else "house"

    def _values(self) -> dict[str, ArgumentValue]:
        leg = self._steps[0] if self._steps and self._steps[0].to is not None else None
        now = self._clock.now()
        progress = 100 if not self._task_total_s else min(100, round(100 * (now - self._task_started) / self._task_total_s))
        room = self._house.room_at(self.x, self.y)
        values: dict[str, ArgumentValue] = {
            "task": self._task,
            "doing": self._doing,
            "phase": self._phase(),
            "room": room.room_id if room else "",
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "heading": round(self.heading, 1),
            "battery": round(self.battery),
            "progress": progress if self._doing else 0,
            "docked": self._at(self._dock),
            "speed_m_s": self._config.speed_m_s,
            # The leg being walked, so the console can move the robot between ticks.
            "leg_from_x": round(self._leg_from[0], 3) if leg else round(self.x, 3),
            "leg_from_y": round(self._leg_from[1], 3) if leg else round(self.y, 3),
            "leg_to_x": round(leg.to[0], 3) if leg and leg.to else round(self.x, 3),
            "leg_to_y": round(leg.to[1], 3) if leg and leg.to else round(self.y, 3),
            "leg_start_ts": round(self._step_started, 3) if leg else 0.0,
            "leg_end_ts": round(self._step_started + leg.duration_s, 3) if leg else 0.0,
            "bowl_full": bool(self._bowl_filled_ts),
            "bowl_filled_ts": round(self._bowl_filled_ts, 3),
            "bathroom_cleaned_ts": round(self._cleaned_ts, 3),
            "last_said": self._last_said,
        }
        return values

    @staticmethod
    def _rounded(seconds: float) -> str:
        if seconds >= 90:
            return f"{round(seconds / 60)} minutes"
        if seconds >= 50:
            return "a minute"
        return f"{max(5, int(5 * round(seconds / 5)))} s"


def manifest(config: RobotConfig, house: HouseConfig, now: float) -> DeviceManifest:
    """What the robot is and what it can be sent to do."""

    def errand(name: str, purpose: str, *parameters: ToolParameter) -> DeviceCapability:
        return DeviceCapability(
            name=name,
            purpose=purpose,
            effect=ToolEffect.WRITE,
            device_class=DeviceClass.COMFORT,
            parameters=parameters,
            min_interval_s=config.min_interval_s,
        )

    rooms = tuple(room.room_id for room in house.rooms)
    room = ToolParameter(
        name="room",
        type=ParameterType.STRING,
        description="Which room to go to: " + ", ".join(rooms),
        choices=rooms,
    )
    dock = house.spots[config.dock_spot]
    return DeviceManifest(
        ts=now,
        device_id=config.device_id,
        name=config.name,
        kind=DeviceKind.ROBOT,
        room=dock.room,
        aliases=tuple(config.aliases),
        simulated=True,
        capabilities=(
            errand("feed_the_dog", "Walk to the kitchen and fill the dog's bowl."),
            errand("clean_the_bathroom", "Clean the bathroom floor."),
            errand("switch_on_tv", "Walk to the TV and switch it on."),
            errand("go_to", "Go to a room of the house and wait there.", room),
            errand("return_to_dock", "Go back to the dock and charge."),
            DeviceCapability(name="stop", purpose="Stop what the robot is doing, where it is.", effect=ToolEffect.WRITE, device_class=DeviceClass.COMFORT),
            DeviceCapability(name="status", purpose="Say what the robot is doing, where, and its battery.", effect=ToolEffect.READ, device_class=DeviceClass.COMFORT),
        ),
    )


def build(config: Config, clock: Clock) -> list[DeviceParts]:
    """The robot, or nothing if it is switched off."""
    if not config.actuators.robot.enabled:
        return []
    robot = HomeRobot(config.actuators.robot, config.house, clock)
    handlers = {
        "feed_the_dog": robot.feed_the_dog,
        "clean_the_bathroom": robot.clean_the_bathroom,
        "switch_on_tv": robot.switch_on_tv,
        "go_to": robot.go_to,
        "return_to_dock": robot.return_to_dock,
        "stop": robot.stop,
        "status": robot.status,
    }
    return [
        DeviceParts(
            manifest=manifest(config.actuators.robot, config.house, clock.now()),
            handlers=handlers,
            state=robot.state,
            tick=robot.tick,
            subscribe=robot.follow,
        )
    ]
