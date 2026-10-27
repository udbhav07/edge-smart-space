"""The room as the assistant must describe it to an occupant (FR-28, FR-60).

A user test (docs/user-testing/2026-10-08) found the assistant answering as
if the room were always healthy: "make it cooler" got "Done" while the air
conditioner was on hold; "set to 22" was said while the room was held at 23
for peak tariff; and "what's the temperature?" or "is everything ok?" could
not be answered at all, while the fault's explanation sat in a log.

This holds what an honest answer needs, read off the blackboard like
everything else and never from another process: the mode, the active
faults and their explanations, what the loop is holding the room to, and
the temperature -- the model's estimate when the sensor cannot be trusted.
Nothing here can change the room; it only says what is true of it.
"""

from __future__ import annotations

from collections import deque

from src.common import topics
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    ActuatorState,
    CommandKind,
    FaultDiagnosis,
    Mode,
    ModeState,
    SensorReading,
    ThermalEstimate,
    TrackingState,
)

#: Explanations held; bounded by the faults that can be active at once.
_DIAGNOSES_KEPT = 16
#: Estimates remembered for the trend: at one every 5 s, the last five minutes.
_HISTORY_KEPT = 60

#: Modes in which the air conditioner is not being driven at all.
_NOT_COOLING_MODES = frozenset({Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD})

_MODE_TEXT = {
    Mode.INIT: "starting up",
    Mode.NORMAL: "working normally",
    Mode.DEGRADED_SENSOR: (
        "the temperature sensor is not working, so the room is being controlled "
        "on an estimate"
    ),
    Mode.DEGRADED_ACTUATOR: (
        "the air conditioner is not responding and is on hold until someone checks it"
    ),
    Mode.SAFE_HOLD: (
        "the system has stopped the air conditioner and is holding safe until "
        "someone checks it and resets it"
    ),
}


class RoomState:
    """What is true of the room now, for the assistant to say."""

    def __init__(self, outdoor_sensor_id: str = "") -> None:
        """
        :param outdoor_sensor_id: the outdoor temperature sensor, so a
            question about slow cooling is answered with the heat outside.
        """
        self._outdoor_sensor_id = outdoor_sensor_id
        self._outdoor_c: float | None = None
        self._actuator: ActuatorState | None = None
        #: (ts, indoor temperature), the last few minutes of it: whether it is falling.
        self._history: deque[tuple[float, float]] = deque(maxlen=_HISTORY_KEPT)
        self._mode: ModeState | None = None
        self._estimate: ThermalEstimate | None = None
        self._tracking: TrackingState | None = None
        self._diagnoses: dict[str, FaultDiagnosis] = {}

    def subscribe(self, blackboard: Blackboard) -> None:
        blackboard.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)
        blackboard.subscribe(topics.ESTIMATE_THERMAL, ThermalEstimate, self._on_estimate)
        blackboard.subscribe(topics.CONTROL_TRACKING, TrackingState, self._on_tracking)
        blackboard.subscribe(topics.DIAGNOSIS, FaultDiagnosis, self._on_diagnosis)
        blackboard.subscribe(topics.ACTUATOR_STATE, ActuatorState, self._on_actuator)
        if self._outdoor_sensor_id:
            blackboard.subscribe(topics.SENSOR_STATE, SensorReading, self._on_reading)

    def _on_actuator(self, _topic: str, state: ActuatorState) -> None:
        self._actuator = state

    def _on_reading(self, _topic: str, reading: SensorReading) -> None:
        if reading.sensor_id == self._outdoor_sensor_id:
            self._outdoor_c = reading.value

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        self._mode = state
        live = set(state.active_fault_ids)
        self._diagnoses = {key: value for key, value in self._diagnoses.items() if key in live}

    def _on_estimate(self, _topic: str, estimate: ThermalEstimate) -> None:
        self._estimate = estimate
        self._history.append((estimate.ts, estimate.t_in))

    def _on_tracking(self, _topic: str, state: TrackingState) -> None:
        self._tracking = state

    def _on_diagnosis(self, _topic: str, diagnosis: FaultDiagnosis) -> None:
        if diagnosis.fault_id not in self._diagnoses and len(self._diagnoses) >= _DIAGNOSES_KEPT:
            self._diagnoses.pop(next(iter(self._diagnoses)))
        self._diagnoses[diagnosis.fault_id] = diagnosis

    # --- what can be said ---------------------------------------------

    @property
    def mode(self) -> Mode | None:
        return self._mode.mode if self._mode else None

    @property
    def cooling_on_hold(self) -> bool:
        return self.mode in _NOT_COOLING_MODES

    @property
    def sensor_out(self) -> bool:
        return self.mode is Mode.DEGRADED_SENSOR

    def explanations(self) -> tuple[str, ...]:
        """What each active fault's diagnosis tells the occupant."""
        return tuple(diagnosis.user_message for diagnosis in self._diagnoses.values())

    def status(self) -> str:
        """The system's state in one sentence, with any fault explained."""
        if self._mode is None:
            return "The system's state is not known yet."
        text = _MODE_TEXT[self._mode.mode]
        # "The system is working normally" reads; "The system is the
        # temperature sensor is not working" did not (S-13, re-verified live).
        if self._mode.mode in (Mode.INIT, Mode.NORMAL):
            sentence = f"The system is {text}."
        else:
            sentence = f"{text[:1].upper()}{text[1:]}."
        explained = " ".join(self.explanations())
        return f"{sentence} {explained}".strip()

    def temperature_c(self) -> float | None:
        """The room's temperature: the estimate whenever the sensor is out."""
        if self._estimate is None:
            return None
        return self._estimate.t_pred if self.sensor_out else self._estimate.t_in

    def held_c(self, setpoint_c: float) -> tuple[float, str]:
        """What a setpoint will actually be held at, and why, if different."""
        if self._tracking is None or not self._tracking.reason:
            return setpoint_c, ""
        shift = self._tracking.tracked_c - self._tracking.setpoint_c
        return setpoint_c + shift, self._tracking.reason

    def summary(self) -> str:
        """The room now, as the model is told it, so a question is answered from fact."""
        lines = ["The room now:"]
        temperature = self.temperature_c()
        if temperature is not None:
            estimated = " (an estimate: the sensor is not working)" if self.sensor_out else ""
            lines.append(f"- temperature: {temperature:.1f} C{estimated}")
        if self._tracking is not None:
            held = self._tracking.tracked_c
            target = f"- target: {self._tracking.setpoint_c:g} C"
            if held != self._tracking.setpoint_c:
                target += f", held at {held:g} C ({self._tracking.reason})"
            lines.append(target)
        if self._actuator is not None:
            cooling = self._actuator.kind is CommandKind.COOL
            lines.append(f"- air conditioner: {'cooling' if cooling else self._actuator.kind.value.lower()}")
        trend = self.trend()
        if trend is not None:
            change, minutes = trend
            way = "falling" if change < -0.05 else "rising" if change > 0.05 else "steady"
            lines.append(f"- over the last {minutes:.0f} minutes: {way}, {abs(change):.1f} C ({abs(change) / minutes:.2f} C a minute)")
        if self._outdoor_c is not None:
            lines.append(f"- outside: {self._outdoor_c:.1f} C")
        lines.append(f"- status: {self.status()}")
        return "\n".join(lines)

    def trend(self) -> tuple[float, float] | None:
        """How much the room's temperature has changed, and over how many minutes; None if too little is known."""
        if len(self._history) < 2:
            return None
        (first_ts, first_c), (last_ts, last_c) = self._history[0], self._history[-1]
        minutes = (last_ts - first_ts) / 60.0
        return (last_c - first_c, minutes) if minutes >= 1.0 else None

    def caveat(self, applied_c: float) -> str:
        """What an occupant must be told beside a new setpoint, if anything."""
        if self.cooling_on_hold:
            return (
                f"But {_MODE_TEXT[self.mode]}, so the room will not be cooled until "
                f"then."
            )
        notes = []
        held, reason = self.held_c(applied_c)
        if reason:
            notes.append(f"Right now it is held at {held:g} degrees ({reason}).")
        if self.sensor_out:
            notes.append(
                "The temperature sensor is not working, so I am controlling on an estimate."
            )
        return " ".join(notes)
