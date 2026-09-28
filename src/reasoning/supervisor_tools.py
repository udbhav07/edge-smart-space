"""The Environmental Supervisor's view of the room (section 5.7.2).

Four read tools and one proposal, and nothing else. The read tools answer from
state this module caches off the blackboard; they never reach a sensor, a
process or a file, so what the supervisor can know is exactly what anyone
running ``mosquitto_sub`` can see (FR-60). The proposal publishes a
:class:`Goal` to ``space/goal/proposed`` -- never an actuator topic (FR-45) --
and the validator judges it like any other.

Tool declarations reuse :class:`~src.common.tools.ToolSpec`, so the surface
the model is shown and the surface its arguments are checked against are one
object, as for the assistance tools.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Coefficients,
    FaultEvent,
    Goal,
    Mode,
    ModeState,
    SensorReading,
    TariffState,
    ThermalEstimate,
)
from src.common.tools import (
    ParameterType,
    ToolArgumentError,
    ToolEffect,
    ToolParameter,
    ToolSpec,
)

GET_THERMAL_STATE = ToolSpec(
    name="get_thermal_state",
    purpose=(
        "Read the room: indoor and outdoor temperature, the setpoint in force, "
        "the model's latest prediction error, and its identified coefficients."
    ),
    effect=ToolEffect.READ,
)
GET_OCCUPANCY = ToolSpec(
    name="get_occupancy",
    purpose="Read whether the room is occupied, and for how long it has been empty.",
    effect=ToolEffect.READ,
)
GET_TARIFF_STATE = ToolSpec(
    name="get_tariff_state",
    purpose=(
        "Read the electricity price band (normal or peak) and when it next "
        "changes. During peak the system already raises the setpoint by itself; "
        "do not add that again."
    ),
    effect=ToolEffect.READ,
)
GET_ACTIVE_FAULTS = ToolSpec(
    name="get_active_faults",
    purpose="Read the faults currently active and the operating mode.",
    effect=ToolEffect.READ,
)
PROPOSE_SETPOINT = ToolSpec(
    name="propose_setpoint",
    purpose=(
        "Propose the target room temperature. A safety gate may clamp or refuse "
        "it; the verdict is returned. Call it once, after reading the state."
    ),
    effect=ToolEffect.WRITE,
    parameters=(
        ToolParameter(
            name="setpoint_c",
            type=ParameterType.NUMBER,
            description="Target temperature in degrees Celsius.",
        ),
        ToolParameter(
            name="mode",
            type=ParameterType.STRING,
            description="The operating mode, exactly as get_active_faults reports it.",
            choices=tuple(mode.value for mode in Mode),
        ),
        ToolParameter(
            name="rationale",
            type=ParameterType.STRING,
            description="One sentence: why this temperature, now.",
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

_UNKNOWN = None


class SemanticRejection(ValueError):
    """A proposal that parsed but does not make sense (FR-44)."""


class SupervisorState:
    """What the supervisor can know, cached from the blackboard."""

    def __init__(self, config: Config, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._estimate: ThermalEstimate | None = None
        self._coefficients: Coefficients | None = None
        self._outdoor_c: float | None = None
        self._setpoint_c = config.controller.default_setpoint_c
        self._occupied: bool | None = None
        self._occupancy_changed_ts: float | None = None
        self._tariff: TariffState | None = None
        self._mode = ModeState(
            ts=clock.now(), mode=Mode.INIT, since_ts=clock.now(), reason="not yet heard"
        )
        self._faults: dict[str, FaultEvent] = {}
        self._changes: list[str] = []

    # --- wiring -------------------------------------------------------

    def subscribe(self, blackboard: Blackboard) -> None:
        blackboard.subscribe(topics.ESTIMATE_THERMAL, ThermalEstimate, self._on_estimate)
        blackboard.subscribe(
            topics.ESTIMATE_COEFFICIENTS, Coefficients, self._on_coefficients
        )
        blackboard.subscribe(topics.SENSOR_STATE, SensorReading, self._on_reading)
        blackboard.subscribe(topics.GOAL_ACTIVE, Goal, self._on_goal)
        blackboard.subscribe(topics.TARIFF_STATE, TariffState, self._on_tariff)
        blackboard.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)
        blackboard.subscribe(topics.FAULT, FaultEvent, self._on_fault)

    @property
    def thermal_known(self) -> bool:
        return self._estimate is not None

    @property
    def mode(self) -> Mode:
        return self._mode.mode

    def take_changes(self) -> list[str]:
        """Events worth an unscheduled cycle since last asked (FR-41)."""
        changes, self._changes = self._changes, []
        return changes

    def _on_estimate(self, _topic: str, estimate: ThermalEstimate) -> None:
        self._estimate = estimate

    def _on_coefficients(self, _topic: str, coefficients: Coefficients) -> None:
        self._coefficients = coefficients

    def _on_reading(self, _topic: str, reading: SensorReading) -> None:
        estimator = self._config.estimator
        if reading.sensor_id == estimator.outdoor_sensor_id:
            self._outdoor_c = reading.value
        elif reading.sensor_id == estimator.occupancy_sensor_id:
            occupied = reading.value >= 0.5
            if self._occupied is not None and occupied != self._occupied:
                self._changes.append("occupancy transition")
            if occupied != self._occupied:
                self._occupancy_changed_ts = reading.ts
            self._occupied = occupied

    def _on_goal(self, _topic: str, goal: Goal) -> None:
        self._setpoint_c = goal.setpoint_c

    def _on_tariff(self, _topic: str, state: TariffState) -> None:
        if self._tariff is not None and state.band is not self._tariff.band:
            self._changes.append("tariff transition")
        self._tariff = state

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        added = set(state.active_fault_ids) - set(self._mode.active_fault_ids)
        if added:
            self._changes.append("fault confirmation")
        self._mode = state

    def _on_fault(self, _topic: str, event: FaultEvent) -> None:
        self._faults[event.fault_id] = event
        # Bounded by what the mode still lists: a withdrawn fault arrives as
        # an empty retained payload, which is never dispatched.
        live = set(self._mode.active_fault_ids) | {event.fault_id}
        self._faults = {key: value for key, value in self._faults.items() if key in live}

    # --- the read tools -----------------------------------------------

    def read(self, tool: str) -> str:
        """Answer a read tool as JSON text, as the model receives it.

        :raises ToolArgumentError: for a name that is not a read tool here.
        """
        readers = {
            GET_THERMAL_STATE.name: self._thermal,
            GET_OCCUPANCY.name: self._occupancy,
            GET_TARIFF_STATE.name: self._tariff_state,
            GET_ACTIVE_FAULTS.name: self._active_faults,
        }
        if tool not in readers:
            raise ToolArgumentError(f"no supervisor tool named {tool!r}")
        return json.dumps(readers[tool](), sort_keys=True)

    def _thermal(self) -> dict[str, object]:
        estimate = self._estimate
        coefficients = self._coefficients
        return {
            "t_in_c": _rounded(estimate.t_in) if estimate else _UNKNOWN,
            "t_out_c": _rounded(self._outdoor_c),
            "t_setpoint_c": _rounded(self._setpoint_c),
            "prediction_residual_c": _rounded(estimate.residual) if estimate else _UNKNOWN,
            "model_confidence": _rounded(estimate.model_confidence) if estimate else _UNKNOWN,
            "coefficients": (
                {
                    name: round(getattr(coefficients, name), 5)
                    for name in ("a1", "a2", "a3", "a4")
                }
                if coefficients
                else _UNKNOWN
            ),
        }

    def _occupancy(self) -> dict[str, object]:
        vacancy_s = (
            round(self._clock.now() - self._occupancy_changed_ts)
            if self._occupied is False and self._occupancy_changed_ts is not None
            else 0
        )
        # Unknown occupancy is treated as occupied, as a failed PIR is
        # (section 7.1): conservative for comfort.
        occupied = self._occupied is not False
        setback = not occupied and vacancy_s >= self._config.supervisor.setback_after_s
        return {
            "occupied": occupied,
            "last_transition": _local(self._occupancy_changed_ts),
            "vacancy_duration_s": vacancy_s,
            "setback_applies": setback,
        }

    def _tariff_state(self) -> dict[str, object]:
        if self._tariff is None:
            return {"band": "unknown", "next_transition": _UNKNOWN}
        return {
            "band": self._tariff.band.value,
            "next_transition": _local(self._tariff.next_transition_ts),
        }

    def _active_faults(self) -> dict[str, object]:
        return {
            "mode": self._mode.mode.value,
            "faults": [
                {
                    "fault_id": event.fault_id,
                    "class": event.fault_class.value,
                    "subject": event.subject,
                    "since": _local(event.detected_ts),
                    "mode_impact": event.mode_impact.value,
                }
                for event in self._faults.values()
            ],
        }

    # --- the proposal -------------------------------------------------

    def check_proposal(self, arguments: Mapping[str, object]) -> tuple[float, Mode, str]:
        """Post-decode semantic validation of a proposal (FR-44).

        Parsing is guaranteed by the decode; sense is not. A proposal naming a
        mode other than the one in force is discarded: the supervisor may
        restate the mode, never change it -- the mode manager owns that.

        :raises ToolArgumentError: if the arguments fail the declaration.
        :raises SemanticRejection: if they parse and make no sense.
        """
        arguments = dict(arguments)
        if not str(arguments.get("mode") or "").strip():
            # The mode can only ever be restated, so a blank one is
            # unambiguous; seen live on a fault-confirmation cycle, where it
            # discarded an otherwise correct proposal.
            arguments["mode"] = self._mode.mode.value
        accepted = PROPOSE_SETPOINT.validate_arguments(arguments)
        setpoint_c = float(accepted["setpoint_c"])
        if not math.isfinite(setpoint_c):
            raise SemanticRejection(f"setpoint {setpoint_c!r} is not a temperature")
        mode = Mode(str(accepted["mode"]))
        if mode is not self._mode.mode:
            raise SemanticRejection(
                f"proposed mode {mode.value} but the system is in {self._mode.mode.value}"
            )
        rationale = str(accepted["rationale"]).strip()
        if not rationale:
            raise SemanticRejection("a proposal must say why")
        return setpoint_c, mode, rationale


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _local(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="minutes")
