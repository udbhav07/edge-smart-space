"""The Environmental Supervisor's tool surface (section 5.7.2).

Four read tools and one terminal tool, and nothing else. The supervisor is not
shown the assistance surface of section 5.7.6 -- it has no business booking a
flight -- and Personal Context is not shown this one (FR-42).

**The read tools read the blackboard, not the components.** A
:class:`RoomSnapshot` subscribes to the retained state topics and the sensor
stream and keeps the latest of each, so answering ``get_thermal_state`` is a
lookup rather than a call into the estimator. It is the same information
``mosquitto_sub`` shows an examiner, which is the point: the model sees what a
person watching the system would see, and nothing a person could not.

**``propose_setpoint`` proposes; it cannot command.** It writes a ``Goal`` to
``space/goal/proposed`` (FR-45) and returns the gate's verdict when the gate
answers in time. Before that, the proposal is checked after decoding (FR-44):
a setpoint that is not a room temperature at all, a mode that is not the mode
the system is in, or a proposal with no reason is discarded here and never
reaches the gate. The check is deliberately narrow. A request for 5 C passes
it -- it is a temperature -- and is refused by the validator, visibly, which is
the demonstration section 5.4 asks for. Filtering it here would hide the one
piece of evidence that the gate works.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.localtime import local_time
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    SETPOINT_KEY,
    Coefficients,
    FaultEvent,
    Goal,
    GoalSource,
    Mode,
    ModeState,
    SensorReading,
    TariffState,
    ThermalEstimate,
    ValidationVerdict,
)
from src.common.tools import (
    ParameterType,
    ToolArgumentError,
    ToolEffect,
    ToolParameter,
    ToolSpec,
)

LOGGER = logging.getLogger(__name__)

#: Most faults the snapshot remembers. Active ones are pruned against the
#: retained mode; this bounds the memory if the mode never arrives.
MAX_FAULTS_KEPT = 32

#: Most trigger events held between two supervisor runs.
MAX_EVENTS_KEPT = 32

#: The reading that means somebody is present (FR-02, a boolean sensor).
_OCCUPIED = 1.0


# --- The declared surface ---------------------------------------------------

GET_THERMAL_STATE = ToolSpec(
    name="get_thermal_state",
    purpose=(
        "Read the room's temperatures: indoor, outdoor, the setpoint in force, "
        "the thermal model's prediction and residual, its coefficients and how "
        "confident it is."
    ),
    effect=ToolEffect.READ,
)

GET_OCCUPANCY = ToolSpec(
    name="get_occupancy",
    purpose=(
        "Read whether somebody is in the room, when that last changed, and "
        "how long it has been empty if it is."
    ),
    effect=ToolEffect.READ,
)

GET_TARIFF_STATE = ToolSpec(
    name="get_tariff_state",
    purpose=(
        "Read the electricity tariff band (normal or peak), when it next "
        "changes, and how far the comfort band shifts up while it is peak."
    ),
    effect=ToolEffect.READ,
)

GET_ACTIVE_FAULTS = ToolSpec(
    name="get_active_faults",
    purpose=(
        "Read the system's current operating mode and every fault that is "
        "active: what failed, which detector found it, and since when."
    ),
    effect=ToolEffect.READ,
)

PROPOSE_SETPOINT = ToolSpec(
    name="propose_setpoint",
    purpose=(
        "Propose the setpoint goal for the room. This is a proposal: a safety "
        "validator decides what is applied and may clamp or refuse it, and the "
        "result tells you what it decided. Call it exactly once, last."
    ),
    effect=ToolEffect.WRITE,
    parameters=(
        ToolParameter(
            name="setpoint_c",
            type=ParameterType.NUMBER,
            description="The temperature to hold, in degrees Celsius.",
        ),
        ToolParameter(
            name="mode",
            type=ParameterType.STRING,
            description=(
                "The system's current operating mode, exactly as "
                "get_active_faults reports it. You do not choose the mode."
            ),
            choices=tuple(mode.value for mode in Mode),
        ),
        ToolParameter(
            name="rationale",
            type=ParameterType.STRING,
            description=(
                "One plain sentence saying why, citing what you read: "
                "occupancy, tariff, faults."
            ),
        ),
    ),
)

SUPERVISOR_TOOLS: tuple[ToolSpec, ...] = (
    GET_THERMAL_STATE,
    GET_OCCUPANCY,
    GET_TARIFF_STATE,
    GET_ACTIVE_FAULTS,
    PROPOSE_SETPOINT,
)


# --- What the blackboard says ----------------------------------------------


class RoomSnapshot:
    """The latest of everything the supervisor may read, kept from the topics.

    Handlers run on the MQTT thread and the supervisor on another, so every
    read and write holds one lock. Nothing blocking happens under it.
    """

    def __init__(self, config: Config, clock: Clock, blackboard: Blackboard) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._lock = threading.Lock()

        self._estimate: ThermalEstimate | None = None
        self._coefficients: Coefficients | None = None
        self._outdoor_c: float | None = None
        self._active_goal: Goal | None = None
        self._mode: ModeState | None = None
        self._tariff: TariffState | None = None
        self._occupied: bool | None = None
        self._occupancy_changed_ts: float | None = None
        self._faults: dict[str, FaultEvent] = {}
        self._events: deque[str] = deque(maxlen=MAX_EVENTS_KEPT)
        self._new_faults: deque[FaultEvent] = deque(maxlen=MAX_FAULTS_KEPT)
        self._verdict: ValidationVerdict | None = None
        self._verdict_arrived = threading.Event()

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        board = self._blackboard
        board.subscribe(topics.ESTIMATE_THERMAL, ThermalEstimate, self._on_estimate)
        board.subscribe(
            topics.ESTIMATE_COEFFICIENTS, Coefficients, self._on_coefficients
        )
        board.subscribe(topics.SENSOR_STATE, SensorReading, self._on_reading)
        board.subscribe(topics.GOAL_ACTIVE, Goal, self._on_active_goal)
        board.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)
        board.subscribe(topics.FAULT, FaultEvent, self._on_fault)
        board.subscribe(topics.CONTEXT_TARIFF, TariffState, self._on_tariff)
        board.subscribe(topics.AUDIT_VALIDATION, ValidationVerdict, self._on_verdict)

    def _on_estimate(self, _topic: str, estimate: ThermalEstimate) -> None:
        with self._lock:
            self._estimate = estimate

    def _on_coefficients(self, _topic: str, coefficients: Coefficients) -> None:
        with self._lock:
            self._coefficients = coefficients

    def _on_active_goal(self, _topic: str, goal: Goal) -> None:
        with self._lock:
            self._active_goal = goal

    def _on_reading(self, _topic: str, reading: SensorReading) -> None:
        estimator = self._config.estimator
        if reading.sensor_id == estimator.outdoor_sensor_id:
            with self._lock:
                self._outdoor_c = reading.value
        elif reading.sensor_id == estimator.occupancy_sensor_id:
            self._observe_occupancy(reading.value == _OCCUPIED, reading.ts)

    def _observe_occupancy(self, occupied: bool, ts: float) -> None:
        """Note a transition, which is one of FR-41's event triggers."""
        with self._lock:
            previous = self._occupied
            if previous is occupied:
                return
            self._occupied = occupied
            self._occupancy_changed_ts = ts
            if previous is not None:
                self._events.append(
                    "occupancy: the room became "
                    + ("occupied" if occupied else "empty")
                )

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        with self._lock:
            self._mode = state
            active = set(state.active_fault_ids)
            for fault_id in list(self._faults):
                if fault_id not in active:
                    del self._faults[fault_id]

    def _on_fault(self, _topic: str, event: FaultEvent) -> None:
        """A confirmed fault: FR-41's third trigger, and FR-25's input."""
        with self._lock:
            if event.fault_id in self._faults:
                return
            if len(self._faults) >= MAX_FAULTS_KEPT:
                self._faults.pop(next(iter(self._faults)))
            self._faults[event.fault_id] = event
            self._new_faults.append(event)
            self._events.append(
                f"fault confirmed: {event.detector.value} on {event.subject}"
            )

    def _on_tariff(self, _topic: str, state: TariffState) -> None:
        with self._lock:
            previous = self._tariff
            self._tariff = state
            if previous is not None and previous.band is not state.band:
                self._events.append(
                    f"tariff: {previous.band.value} -> {state.band.value}"
                )

    def _on_verdict(self, _topic: str, verdict: ValidationVerdict) -> None:
        """Keep setpoint verdicts; command verdicts are the loop's business."""
        if SETPOINT_KEY not in verdict.proposed:
            return
        with self._lock:
            self._verdict = verdict
        self._verdict_arrived.set()

    # --- what the service drains --------------------------------------

    def drain_events(self) -> list[str]:
        """Trigger events since the last drain (FR-41)."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def drain_new_faults(self) -> list[FaultEvent]:
        """Faults confirmed since the last drain, for diagnosis (FR-25)."""
        with self._lock:
            faults = list(self._new_faults)
            self._new_faults.clear()
        return faults

    @property
    def mode(self) -> Mode:
        with self._lock:
            return Mode.INIT if self._mode is None else self._mode.mode

    # --- the verdict on a proposal ------------------------------------

    def expect_verdict(self) -> None:
        """Forget any earlier verdict, before publishing a proposal."""
        with self._lock:
            self._verdict = None
        self._verdict_arrived.clear()

    def await_verdict(self, setpoint_c: float, timeout_s: float) -> ValidationVerdict | None:
        """The gate's answer to a proposal, if it arrives in time.

        :returns: the verdict on this setpoint, or None. None is an honest
            answer rather than an error: the control process may be down, and
            the proposal then stands on the blackboard unanswered, which is
            what the model is told.
        """
        if not self._clock.wait_for(self._verdict_arrived, timeout_s):
            return None
        with self._lock:
            verdict = self._verdict
        if verdict is None or verdict.proposed.get(SETPOINT_KEY) != setpoint_c:
            return None
        return verdict

    # --- the tool answers ---------------------------------------------

    def _local(self, ts: float | None) -> str | None:
        if ts is None:
            return None
        return local_time(ts, self._config.site.utc_offset_h).isoformat(
            timespec="minutes"
        )

    def thermal_state(self) -> dict[str, object]:
        with self._lock:
            estimate, coefficients = self._estimate, self._coefficients
            outdoor_c, goal = self._outdoor_c, self._active_goal
        state: dict[str, object] = {
            "t_in_c": None if estimate is None else round(estimate.t_in, 2),
            "t_out_c": None if outdoor_c is None else round(outdoor_c, 2),
            "t_setpoint_c": (
                self._config.controller.default_setpoint_c
                if goal is None
                else goal.setpoint_c
            ),
            "setpoint_source": "default" if goal is None else goal.source.value,
        }
        if estimate is not None:
            state.update(
                t_pred_c=round(estimate.t_pred, 2),
                residual_c=round(estimate.residual, 3),
                residual_sigma_c=round(estimate.residual_sigma, 3),
                model_confidence=round(estimate.model_confidence, 2),
                adaptation=estimate.adaptation.value,
                age_s=round(self._clock.now() - estimate.ts, 1),
            )
        if coefficients is not None:
            state["coefficients"] = {
                name: round(getattr(coefficients, name), 5)
                for name in ("a1", "a2", "a3", "a4")
            }
        return state

    def occupancy(self) -> dict[str, object]:
        with self._lock:
            occupied, changed_ts = self._occupied, self._occupancy_changed_ts
        vacancy_s = None
        if occupied is False and changed_ts is not None:
            vacancy_s = round(self._clock.now() - changed_ts, 0)
        return {
            "occupied": occupied,
            "last_transition": self._local(changed_ts),
            "vacancy_duration_s": vacancy_s,
        }

    def tariff_state(self) -> dict[str, object]:
        with self._lock:
            tariff = self._tariff
        if tariff is None:
            return {"band": "unknown", "next_transition": None, "offset_c": 0.0}
        return {
            "band": tariff.band.value,
            "next_transition": self._local(tariff.next_transition_ts),
            "offset_c": tariff.offset_c,
        }

    def active_faults(self) -> dict[str, object]:
        with self._lock:
            mode = self._mode
            faults = list(self._faults.values())
            if mode is not None:
                active = set(mode.active_fault_ids)
                faults = [event for event in faults if event.fault_id in active]
        return {
            "mode": Mode.INIT.value if mode is None else mode.mode.value,
            "faults": [
                {
                    "fault_id": event.fault_id,
                    "class": event.fault_class.value,
                    "sensor": event.subject,
                    "detector": event.detector.value,
                    "since": self._local(event.detected_ts),
                    "mode_impact": event.mode_impact.value,
                }
                for event in faults
            ],
        }


# --- Running a tool ---------------------------------------------------------


@dataclass(frozen=True)
class ToolAnswer:
    """What one tool call produced.

    ``content`` goes back to the model. The rest is for the supervisor and its
    audit record: whether a proposal went out, what the gate said, and --
    when the post-decode check discarded a proposal -- why (FR-44).
    """

    content: Mapping[str, object]
    proposed: Goal | None = None
    verdict: ValidationVerdict | None = None
    discarded: str = ""

    def as_text(self) -> str:
        return json.dumps(dict(self.content))


class SupervisorTools:
    """Runs the supervisor's tools against the snapshot."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        snapshot: RoomSnapshot,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._snapshot = snapshot
        self._specs = {spec.name: spec for spec in SUPERVISOR_TOOLS}

    def schemas(self) -> tuple[dict[str, object], ...]:
        """What the model is shown: the five declarations, nothing more."""
        return tuple(spec.as_schema() for spec in SUPERVISOR_TOOLS)

    def execute(self, name: str, raw_arguments: str) -> ToolAnswer:
        """Run one call the model asked for.

        A call to a tool that does not exist, or with arguments that do not
        parse or fit the declaration, gets an error back rather than an
        exception: the model is told what was wrong and may try again within
        its round limit. Only a proposal failing the post-decode check ends
        the run, because FR-44 says to discard it, not to negotiate.
        """
        spec = self._specs.get(name)
        if spec is None:
            return ToolAnswer({"error": f"no tool named {name!r}"})
        try:
            decoded = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return ToolAnswer({"error": f"arguments are not JSON: {exc}"})
        if not isinstance(decoded, dict):
            return ToolAnswer({"error": "arguments must be a JSON object"})
        try:
            arguments = spec.validate_arguments(decoded)
        except ToolArgumentError as exc:
            if name == PROPOSE_SETPOINT.name:
                return ToolAnswer({"error": str(exc)}, discarded=str(exc))
            return ToolAnswer({"error": str(exc)})

        if name == GET_THERMAL_STATE.name:
            return ToolAnswer(self._snapshot.thermal_state())
        if name == GET_OCCUPANCY.name:
            return ToolAnswer(self._snapshot.occupancy())
        if name == GET_TARIFF_STATE.name:
            return ToolAnswer(self._snapshot.tariff_state())
        if name == GET_ACTIVE_FAULTS.name:
            return ToolAnswer(self._snapshot.active_faults())
        return self._propose(arguments)

    def _propose(self, arguments: Mapping[str, object]) -> ToolAnswer:
        """Check, publish, and report the gate's answer (FR-44, FR-45)."""
        setpoint_c = float(arguments["setpoint_c"])
        mode = Mode(str(arguments["mode"]))
        rationale = str(arguments["rationale"]).strip()

        problem = self._semantic_problem(setpoint_c, mode, rationale)
        if problem:
            LOGGER.warning("discarding supervisor proposal: %s", problem)
            return ToolAnswer(
                {"status": "discarded", "reason": problem}, discarded=problem
            )

        now = self._clock.now()
        goal = Goal(
            ts=now,
            source=GoalSource.SUPERVISOR,
            setpoint_c=setpoint_c,
            mode=mode,
            rationale=rationale,
            expires_ts=now + self._config.reasoning.supervisor_goal_lifetime_s,
        )
        self._snapshot.expect_verdict()
        self._blackboard.publish(topics.GOAL_PROPOSED, goal)
        verdict = self._snapshot.await_verdict(
            setpoint_c, self._config.reasoning.verdict_timeout_s
        )
        if verdict is None:
            return ToolAnswer(
                {
                    "status": "submitted",
                    "note": "no verdict from the safety validator yet; the "
                    "proposal stands on the blackboard",
                },
                proposed=goal,
            )
        return ToolAnswer(
            {
                "status": "decided",
                "verdict": verdict.verdict.value,
                "reason": verdict.reason.value,
                "applied_setpoint_c": verdict.applied.get(SETPOINT_KEY),
            },
            proposed=goal,
            verdict=verdict,
        )

    def _semantic_problem(self, setpoint_c: float, mode: Mode, rationale: str) -> str:
        """The post-decode check (FR-44). Empty when the proposal passes.

        Three things only, and each is a sign the model has misunderstood the
        task rather than made a choice the gate should judge.
        """
        plausible = self._config.reasoning.plausible_setpoint_c
        if not math.isfinite(setpoint_c) or not plausible.contains(setpoint_c):
            return (
                f"{setpoint_c} C is not a room temperature "
                f"(plausible {plausible.low} to {plausible.high} C)"
            )
        current = self._snapshot.mode
        if mode is not current:
            return (
                f"mode {mode.value} is not the system's mode ({current.value}); "
                f"the supervisor reports the mode, it does not choose it"
            )
        if not rationale:
            return "a proposal must say why"
        return ""
