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

import math
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

#: Absolute tolerance when checking a field against the inputs it is derived
#: from. Two orders of magnitude below the resolution of any sensor in the
#: system, so it catches a genuinely wrong value without rejecting ordinary
#: floating-point round-off.
_DERIVED_FIELD_TOLERANCE = 1e-6

#: The only values a boolean-unit reading may carry (FR-02 binary occupancy).
_BOOLEAN_READING_VALUES = (0.0, 1.0)

#: Keys inside a ValidationVerdict's proposed and applied objects.
SETPOINT_KEY = "setpoint_c"
COMMAND_KIND_KEY = "kind"

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
    """Common contract for every payload published to the blackboard.

    Frozen and closed to unknown fields. Carries no timestamp of its own:
    FaultEvent times itself with ``detected_ts`` rather than ``ts``, and
    forcing both onto it would reject the payload in section 6.2.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class TimestampedMessage(BlackboardMessage):
    """A payload stamped with the moment it describes."""

    ts: float = Field(gt=0.0, description="Unix epoch seconds, from the injected Clock")


# --- Sensing ---------------------------------------------------------------


class SensorReading(TimestampedMessage):
    """One sample from one sensor (FR-01).

    Carries its own id and timestamp so a consumer never has to infer either
    from the topic or from arrival order. Arrival order is not trustworthy:
    A-02's uniform sampling will not survive WiFi reconnect bursts.

    Temperature and humidity are deliberately unbounded. Their physical limits
    are detector D3's configured range, and a reading outside it must reach
    the detector to be detected (FR-22). A boolean reading is different: a PIR
    reporting 27.4 is a malformed message, not a sensor fault, so it is
    rejected here.
    """

    sensor_id: str = Field(min_length=1)
    value: float
    unit: Unit
    quality: Quality = Quality.OK

    @model_validator(mode="after")
    def _boolean_readings_carry_only_zero_or_one(self) -> SensorReading:
        if self.unit is Unit.BOOLEAN and self.value not in _BOOLEAN_READING_VALUES:
            raise ValueError(
                f"a {Unit.BOOLEAN.value} reading must be one of "
                f"{list(_BOOLEAN_READING_VALUES)}, got {self.value!r}"
            )
        return self


class SensorHealth(TimestampedMessage):
    """Retained per-sensor status, so a late subscriber knows what is trusted."""

    sensor_id: str = Field(min_length=1)
    quality: Quality
    last_reading_ts: float | None = Field(
        default=None, description="None until the first reading arrives"
    )
    active_fault_id: str | None = None


# --- Estimation ------------------------------------------------------------


class ThermalEstimate(TimestampedMessage):
    """One-step prediction and residual, published every tick (FR-05).

    ``residual`` is derived from the other two fields, so it is checked
    against them. An inconsistent triple would corrupt D4, whose CUSUM test
    runs on this residual, and would do so silently.
    """

    t_in: float
    t_pred: float
    residual: float
    residual_sigma: float = Field(ge=0.0)
    model_confidence: float = Field(
        ge=0.0, le=1.0, description="Derived from trace(P). Not a probability."
    )
    adaptation: AdaptationState

    @model_validator(mode="after")
    def _residual_agrees_with_the_values_it_is_derived_from(self) -> ThermalEstimate:
        expected = self.t_in - self.t_pred
        if not math.isclose(
            self.residual, expected, abs_tol=_DERIVED_FIELD_TOLERANCE
        ):
            raise ValueError(
                f"residual {self.residual!r} contradicts t_in - t_pred "
                f"({expected!r})"
            )
        return self


class Coefficients(TimestampedMessage):
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

    @model_validator(mode="after")
    def _steady_state_residual_agrees_with_the_coefficients(self) -> Coefficients:
        """Section 5.2.1 makes drift from a1 + a2 = 1 a diagnostic signal.

        A value that disagrees with the coefficients it summarises would
        misreport that diagnostic, so the two are checked against each other.
        """
        expected = abs(self.a1 + self.a2 - 1.0)
        if not math.isclose(
            self.steady_state_residual, expected, abs_tol=_DERIVED_FIELD_TOLERANCE
        ):
            raise ValueError(
                f"steady_state_residual {self.steady_state_residual!r} contradicts "
                f"|a1 + a2 - 1| ({expected!r})"
            )
        return self


# --- Faults and mode -------------------------------------------------------


class FaultEvent(BlackboardMessage):
    """A detector's finding, with the evidence that produced it.

    evidence is an open mapping because each detector reports a different set
    of numeric features and the schema is shared. It is made immutable at
    validation: freezing the model alone would still leave the dictionary
    writable, and a fault's evidence is the record of why a mode transition
    happened. It serialises back to a plain JSON object.
    """

    fault_id: str = Field(min_length=1)
    detector: DetectorId
    subject: str = Field(min_length=1, description="Sensor or actuator id")
    fault_class: FaultClass = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
    detected_ts: float = Field(gt=0.0)
    evidence: Mapping[str, float] = Field(default_factory=dict)
    mode_impact: Mode

    @field_validator("evidence", mode="after")
    @classmethod
    def _freeze_evidence(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return MappingProxyType(dict(value))

    @field_serializer("evidence")
    def _serialise_evidence(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)


class ModeState(TimestampedMessage):
    """Retained current mode (FR-61), published before any diagnosis runs."""

    mode: Mode
    since_ts: float = Field(gt=0.0)
    active_fault_ids: tuple[str, ...] = ()
    reason: str = Field(default="", description="Human-readable transition cause")


# --- Goals and validation --------------------------------------------------


class Goal(TimestampedMessage):
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


class ValidationVerdict(TimestampedMessage):
    """The audit record of one validation decision (FR-13).

    A CLAMPED verdict is evidence the gate works and is displayed as a
    finding, not hidden as an error (DESIGN.md section 5.4).

    ``proposed`` and ``applied`` are open objects rather than bare setpoints.
    That is what the section 6.2 payload specifies, and it is what lets one
    schema on one topic carry both kinds of verdict the validator produces:
    section 5.4's rules act on setpoints (V-1, V-2, V-6) *and* on commands
    (V-3, V-4, V-5), and section 5.4 sends every verdict to
    ``space/audit/validation``.
    """

    proposed: Mapping[str, float | str]
    verdict: Verdict
    reason: ReasonCode
    applied: Mapping[str, float | str]

    @field_validator("proposed", "applied", mode="after")
    @classmethod
    def _freeze_decision(
        cls, value: Mapping[str, float | str]
    ) -> Mapping[str, float | str]:
        return MappingProxyType(dict(value))

    @field_serializer("proposed", "applied")
    def _serialise_decision(
        self, value: Mapping[str, float | str]
    ) -> dict[str, float | str]:
        return dict(value)

    @model_validator(mode="after")
    def _accepted_verdict_must_not_alter_the_proposal(self) -> ValidationVerdict:
        if self.verdict is Verdict.ACCEPTED:
            if self.reason is not ReasonCode.NONE:
                raise ValueError("an ACCEPTED verdict must carry reason NONE")
            if dict(self.applied) != dict(self.proposed):
                raise ValueError("an ACCEPTED verdict must apply the proposal unchanged")
        elif self.reason is ReasonCode.NONE:
            raise ValueError(f"a {self.verdict.value} verdict must carry a reason code")
        return self


# --- Actuation -------------------------------------------------------------


class Command(TimestampedMessage):
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


class Comfort(str, Enum):
    """The direction an occupant asked for, in their own terms."""

    WARMER = "warmer"
    COOLER = "cooler"
    UNCHANGED = "unchanged"


class Intent(str, Enum):
    """What kind of thing was asked for.

    These are the three branches in DESIGN.md section 5.8's sequence, named
    so an utterance about something other than temperature is representable
    rather than silently discarded:

    ``ENVIRONMENT``
        A preference about the space. Forwarded to the goal path as a
        supervisory input and gated there (FR-53).
    ``SERVICE``
        An action against an external service. Requires explicit
        confirmation before anything is invoked (FR-54), so it is never
        acted on from here.
    ``NONE``
        Nothing actionable was said. The ordinary case, not an error.
    """

    ENVIRONMENT = "environment"
    SERVICE = "service"
    NONE = "none"


class PreferenceHint(TimestampedMessage):
    """A spoken preference, forwarded as a supervisory input (FR-53).

    Deliberately not a command. It names a direction and, optionally, a
    temperature the occupant asked for, and it reaches the plant only if the
    goal path proposes a setpoint from it and the validator admits that
    setpoint. Speech is a request weighed like any other, not a shortcut past
    the gate.

    This is the payload on ``space/context/preference``. Section 6.1 names
    the message; its fields are not specified there, so they are kept to what
    the control path can actually consume.

    It carries more than a temperature. ``intent`` distinguishes the three
    branches section 5.8 already describes, ``subject`` names what the
    request was about, and ``spoken_reply`` carries what to say back. None of
    that makes the message a command: a SERVICE intent is still only a
    proposal awaiting explicit confirmation (FR-54), and an ENVIRONMENT
    intent still reaches the plant only through a setpoint the validator
    admits.
    """

    intent: Intent = Field(
        default=Intent.ENVIRONMENT, description="Which section 5.8 branch applies"
    )
    comfort: Comfort
    subject: str = Field(
        default="",
        description="What the request was about: temperature, lights, a booking",
    )
    target_c: float | None = Field(
        default=None, description="Temperature named by the occupant, if any"
    )
    rationale: str = Field(default="", description="What was said, briefly")
    spoken_reply: str = Field(
        default="", description="Plain-English answer to speak back to the occupant"
    )

    @model_validator(mode="after")
    def _nothing_actionable_carries_no_request(self) -> PreferenceHint:
        """A NONE intent must not smuggle a request through.

        Without this, an extraction that decided nothing was asked could
        still name a target temperature, and the goal path would have no way
        to tell an actual request from a discarded one.
        """
        if self.intent is not Intent.NONE:
            return self
        if self.comfort is not Comfort.UNCHANGED:
            raise ValueError("an intent of NONE must carry comfort 'unchanged'")
        if self.target_c is not None:
            raise ValueError("an intent of NONE must not name a target temperature")
        return self


class ActuatorState(TimestampedMessage):
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
