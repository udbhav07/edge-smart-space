"""Message schemas for every payload on the blackboard.

These are the contracts frozen at Week 2 (R-06) so both team members can work
against them rather than against each other's code. Changing a field here is
an interface change.

Design rules applied throughout:

* Every model is frozen. A message is a value, not a mutable buffer, and one
  component must not be able to alter what another is reading.
* Every model forbids unknown fields. A typo in a publisher is caught at the
  subscriber boundary instead of being silently dropped.
* Schemas constrain what is *physically representable*, never what is
  *currently permitted*. Setpoint bounds, coefficient plausibility boxes and
  detector thresholds are policy and live in config, because they are re-tuned
  on hardware (R-04). A Coefficients message must be able to carry an
  out-of-box estimate, or FR-06's rejection logging could not describe what it
  rejected.

Payload examples: DESIGN.md section 6.2.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Enumerations ----------------------------------------------------------


class Unit(str, Enum):
    """Physical unit of a sensor reading."""

    CELSIUS = "C"
    PERCENT_RH = "%RH"
    BOOLEAN = "bool"


class Quality(str, Enum):
    """Trustworthiness of a reading, as judged by the detector bank."""

    OK = "ok"
    SUSPECT = "suspect"
    FAULTED = "faulted"


class AdaptationState(str, Enum):
    """Whether RLS is currently updating coefficients (FR-29)."""

    ACTIVE = "active"
    FROZEN = "frozen"


class Mode(str, Enum):
    """System operating mode. State machine: DESIGN.md section 5.6."""

    INIT = "INIT"
    NORMAL = "NORMAL"
    DEGRADED_SENSOR = "DEGRADED_SENSOR"
    DEGRADED_ACTUATOR = "DEGRADED_ACTUATOR"
    SAFE_HOLD = "SAFE_HOLD"


class FaultClass(str, Enum):
    """What kind of thing failed."""

    SENSOR = "sensor"
    ACTUATOR = "actuator"
    MODEL = "model"


class DetectorId(str, Enum):
    """The five detectors in DESIGN.md section 5.5, plus model divergence.

    D4 and D5 are the two that exist only because the model exists; a
    threshold controller has no expectation to compare against.
    """

    D1_DROPOUT = "D1_DROPOUT"
    D2_STUCK_AT = "D2_STUCK_AT"
    D3_OUT_OF_RANGE = "D3_OUT_OF_RANGE"
    D4_DRIFT = "D4_DRIFT"
    D5_ACTUATOR_NO_RESPONSE = "D5_ACTUATOR_NO_RESPONSE"
    MODEL_DIVERGENCE = "MODEL_DIVERGENCE"


class GoalSource(str, Enum):
    """Who proposed a setpoint. The validator treats all sources alike."""

    SUPERVISOR = "supervisor"
    PREFERENCE = "preference"
    DEFAULT = "default"
    OPERATOR = "operator"


class Verdict(str, Enum):
    """Outcome of safety validation (DESIGN.md section 5.4)."""

    ACCEPTED = "ACCEPTED"
    CLAMPED = "CLAMPED"
    BLOCKED = "BLOCKED"


class ReasonCode(str, Enum):
    """Why the validator did what it did. Rules V-1 to V-6."""

    NONE = "NONE"
    BOUND_CLAMP = "BOUND_CLAMP"
    RATE_LIMIT = "RATE_LIMIT"
    DWELL = "DWELL"
    CMD_RATE = "CMD_RATE"
    MODE_BLOCK = "MODE_BLOCK"
    STALE_GOAL = "STALE_GOAL"


class CommandKind(str, Enum):
    """What the regulatory controller asks the actuator to do."""

    COOL = "COOL"
    OFF = "OFF"
    MAINTAIN = "MAINTAIN"
    HOLD = "HOLD"


class AckStatus(str, Enum):
    """Whether a command was confirmed by the device.

    UNKNOWN is first-class and is the *correct* value for an open-loop IR
    path, where no readback exists (R-02). Code that treats a missing
    acknowledgement as success will misdiagnose D5, so there is deliberately
    no default: the driver must state which case it is in.
    """

    ACKNOWLEDGED = "ACKNOWLEDGED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class TariffBand(str, Enum):
    """Electricity pricing band (FR-16)."""

    NORMAL = "normal"
    PEAK = "peak"


# --- Base ------------------------------------------------------------------


class BlackboardMessage(BaseModel):
    """Common contract for every payload published to the blackboard."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    ts: float = Field(gt=0.0, description="Unix epoch seconds, from the injected Clock")


# --- Sensing ---------------------------------------------------------------


class SensorReading(BlackboardMessage):
    """One sample from one sensor (FR-01).

    Carries its own id and timestamp so a consumer never has to infer either
    from the topic or from arrival order. Arrival order is not trustworthy:
    A-02's uniform sampling will not survive WiFi reconnect bursts.
    """

    sensor_id: str = Field(min_length=1)
    value: float
    unit: Unit
    quality: Quality = Quality.OK


class SensorHealth(BlackboardMessage):
    """Retained per-sensor status, so a late subscriber knows what is trusted."""

    sensor_id: str = Field(min_length=1)
    quality: Quality
    last_reading_ts: float | None = Field(
        default=None, description="None until the first reading arrives"
    )
    active_fault_id: str | None = None


# --- Estimation ------------------------------------------------------------


class ThermalEstimate(BlackboardMessage):
    """One-step prediction and residual, published every tick (FR-05)."""

    t_in: float
    t_pred: float
    residual: float
    residual_sigma: float = Field(ge=0.0)
    model_confidence: float = Field(
        ge=0.0, le=1.0, description="Derived from trace(P). Not a probability."
    )
    adaptation: AdaptationState


class Coefficients(BlackboardMessage):
    """The four RC coefficients and their identification health (FR-04).

    a1 to a4 are deliberately unconstrained here. Their plausibility box is
    policy (DESIGN.md section 5.2.1) enforced by the estimator's projection
    step, and a rejected out-of-box estimate must still be publishable for
    FR-06's rejection logging to mean anything.
    """

    a1: float
    a2: float
    a3: float
    a4: float
    trace_p: float = Field(ge=0.0, description="Covariance trace; excitation health")
    steady_state_residual: float = Field(
        ge=0.0, description="|a1 + a2 - 1|; drift from steady-state consistency"
    )
    samples_since_reset: int = Field(ge=0)
    adaptation: AdaptationState


# --- Faults and mode -------------------------------------------------------


class FaultEvent(BlackboardMessage):
    """A detector's finding, with the evidence that produced it.

    evidence is a mutable mapping by necessity: each detector reports a
    different set of numeric features and the schema is shared. Treat it as
    read-only; it exists to be displayed and logged, never mutated.
    """

    fault_id: str = Field(min_length=1)
    detector: DetectorId
    subject: str = Field(min_length=1, description="Sensor or actuator id")
    fault_class: FaultClass = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
    detected_ts: float = Field(gt=0.0)
    evidence: dict[str, float] = Field(default_factory=dict)
    mode_impact: Mode


class ModeState(BlackboardMessage):
    """Retained current mode (FR-61), published before any diagnosis runs."""

    mode: Mode
    since_ts: float = Field(gt=0.0)
    active_fault_ids: tuple[str, ...] = ()
    reason: str = Field(default="", description="Human-readable transition cause")


# --- Goals and validation --------------------------------------------------


class Goal(BlackboardMessage):
    """A proposed or active setpoint goal.

    The reasoning layer's entire influence on the plant is this message
    (FR-45). It is advisory until the validator has passed it.
    """

    source: GoalSource
    setpoint_c: float
    mode: Mode
    rationale: str = Field(default="")
    expires_ts: float = Field(
        gt=0.0, description="Staleness horizon enforced by validator rule V-6"
    )


class ValidationVerdict(BlackboardMessage):
    """The audit record of one validation decision (FR-13).

    A CLAMPED verdict is evidence the gate works and is displayed as a
    finding, not hidden as an error (DESIGN.md section 5.4).
    """

    proposed_setpoint_c: float
    verdict: Verdict
    reason: ReasonCode
    applied_setpoint_c: float

    @model_validator(mode="after")
    def _accepted_verdict_must_not_alter_the_proposal(self) -> ValidationVerdict:
        if self.verdict is Verdict.ACCEPTED:
            if self.reason is not ReasonCode.NONE:
                raise ValueError("an ACCEPTED verdict must carry reason NONE")
            if self.applied_setpoint_c != self.proposed_setpoint_c:
                raise ValueError("an ACCEPTED verdict must apply the proposal unchanged")
        elif self.reason is ReasonCode.NONE:
            raise ValueError(f"a {self.verdict.value} verdict must carry a reason code")
        return self


# --- Actuation -------------------------------------------------------------


class Command(BlackboardMessage):
    """One actuator command, after validation (FR-12)."""

    actuator_id: str = Field(min_length=1)
    kind: CommandKind
    setpoint_c: float | None = Field(
        default=None, description="Present only for COOL; meaningless otherwise"
    )

    @model_validator(mode="after")
    def _setpoint_accompanies_exactly_the_cool_command(self) -> Command:
        if self.kind is CommandKind.COOL and self.setpoint_c is None:
            raise ValueError("a COOL command requires a setpoint")
        if self.kind is not CommandKind.COOL and self.setpoint_c is not None:
            raise ValueError(f"a {self.kind.value} command must not carry a setpoint")
        return self


class ActuatorState(BlackboardMessage):
    """Retained actuator state.

    simulated has no default: FR-15 requires every simulated actuator to be
    labelled in every published state message, and a default is how that
    silently stops happening.
    """

    actuator_id: str = Field(min_length=1)
    simulated: bool
    kind: CommandKind
    setpoint_c: float | None = None
    ack: AckStatus
    last_command_ts: float | None = None
