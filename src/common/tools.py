"""The tool-calling contract: how the reasoning layer asks for an action.

DESIGN.md section 5.7.6. The reasoning layer can propose a setpoint and
nothing else (FR-45), which is the right constraint for the plant and the
wrong one for everything an occupant might actually ask for. "Put that in my
calendar" is not a setpoint, and until now the system had no representation
for it at all.

This module is that representation. A tool is a *declaration*: a name, a
purpose, a typed parameter list, and whether it changes anything. A model is
shown the declaration and nothing else -- it asks for ``schedule_event`` with
a time and a subject, and it is never told whether that lands in a local
file, in Google Calendar, or in a Microsoft one. Two consequences follow, and
both are the point rather than a side effect:

* **Providers are swappable without touching the reasoning layer** (FR-71,
  FR-72). :class:`ToolProvider` is a Protocol declared here and implemented
  elsewhere; a provider change is a wiring change in the process that binds
  it. The prompt, the schemas, and the topics are untouched.
* **The model runs its own tools, up to a line it cannot cross.** Reading a
  calendar and writing to it are things it does; committing the occupant to a
  flight is not. That line is a declared property of each tool
  (:class:`ToolEffect`), not a judgement made at the call site, so it cannot
  be forgotten at one of them (FR-73, FR-74).

Where the pieces live, and why here:

* The contract, the registry, and the wire messages are all in this module
  because a registry that returns a :class:`ToolResult` cannot live in one
  file while the message lives in another that imports it back.
* :class:`ToolProvider` implementations are not here and must never be. A
  calendar client is I/O, and ``src/common/`` imports stdlib, pydantic and
  paho only.

The tool *name* is a validated string rather than an enum. An enum would put
the catalogue in the type system, and FR-72 requires adding a tool to be one
new :class:`ToolSpec` and a binding -- nothing else. Validity is decided
against the registry at the boundary instead, which is stricter than an enum
in the way that matters: an unknown name is refused with a reason, in one
place, and published (FR-73, FR-75).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from src.common.clock import Clock
from src.common.schemas import TimestampedMessage

LOGGER = logging.getLogger(__name__)

#: What a tool argument or a result detail may carry. Deliberately flat: a
#: nested structure would not render in a terminal view of the blackboard, and
#: nothing in the declared surface needs one. Same reasoning as
#: ``FaultEvent.evidence``.
ArgumentValue = bool | int | float | str | None

#: A tool name must be one lower-case identifier. It appears in a prompt, in a
#: JSON schema, and in a log line, and a name with a space or a slash in it
#: breaks at least one of those.
_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"

#: Minutes in a default calendar entry when the occupant named no duration.
DEFAULT_EVENT_DURATION_MIN = 60


class ParameterType(str, Enum):
    """The type of one tool parameter.

    ``TIMESTAMP`` is separate from ``STRING`` because of what produces these
    arguments. A model can reliably write ``2026-09-04T15:00:00``; it cannot
    reliably write ``1757000000.0``. Every other timestamp in the system is
    epoch seconds (section 6.2), and this is the one place that is wrong --
    the value here is written by a language model, not by a clock.
    """

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"


class ToolEffect(str, Enum):
    """How far the consequences of running a tool reach.

    This is what the confirmation gate is decided on (FR-74), so it is a
    declared property of the tool rather than a judgement made at the call
    site. The line is not "does it change anything" -- it is **whose** it is
    to undo:

    ``READ``
        Observes and changes nothing. Listing what is in the calendar.
    ``WRITE``
        Changes something the system itself owns. A local calendar entry is
        visible in the console, published like everything else (FR-75), and
        deletable by the person who did not want it. The model runs these.
    ``COMMIT``
        Commits the occupant to a party outside the system -- a flight, a
        hotel, money, a seat somebody else cannot have. Nothing here is ours
        to reverse, so nothing here runs without being asked (FR-54, FR-74).

    A tool declaring the wrong one of these is the single most consequential
    mistake available in this module, which is why the value is required and
    has no default.
    """

    READ = "read"
    WRITE = "write"
    COMMIT = "commit"


class ToolStatus(str, Enum):
    """What happened to an invocation.

    Every value is a distinct thing an occupant or an examiner needs to be
    able to tell apart, which is why there is no single ``ERROR``: "the model
    asked for a tool that does not exist" and "the calendar rejected the
    write" are different findings.
    """

    OK = "OK"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    BAD_ARGUMENTS = "BAD_ARGUMENTS"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    EXPIRED = "EXPIRED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


class ToolRequester(str, Enum):
    """Who asked for the invocation.

    Recorded because a tool that writes to a calendar on someone's behalf
    needs an audit trail naming what initiated it (FR-46, FR-75).
    """

    PERSONAL_CONTEXT = "personal_context"
    CONSOLE = "console"
    OPERATOR = "operator"


# --- Errors ----------------------------------------------------------------


class ToolError(Exception):
    """Base for every refusal the registry can produce.

    Each subclass maps to exactly one :class:`ToolStatus`. The registry
    translates them into a published result rather than propagating them:
    FR-75 requires the refusal to be visible on the blackboard, and an
    exception escaping to a service loop is not visible to anyone.
    """

    status: ToolStatus = ToolStatus.FAILED


class UnknownToolError(ToolError):
    """The named tool is not in the catalogue."""

    status = ToolStatus.UNKNOWN_TOOL


class ToolArgumentError(ToolError):
    """The arguments do not satisfy the declared parameter list (FR-73)."""

    status = ToolStatus.BAD_ARGUMENTS


class ConfirmationRequiredError(ToolError):
    """A COMMIT tool was invoked without confirmation (FR-74)."""

    status = ToolStatus.CONFIRMATION_REQUIRED


class InvocationExpiredError(ToolError):
    """The confirmation arrived after the invocation's window closed."""

    status = ToolStatus.EXPIRED


class ToolUnavailableError(ToolError):
    """The tool is declared but no provider is bound to it."""

    status = ToolStatus.UNAVAILABLE


class ToolFailedError(ToolError):
    """The provider was reached and did not complete the action."""

    status = ToolStatus.FAILED


# --- Declaration -----------------------------------------------------------


class ToolParameter(BaseModel):
    """One argument a tool takes.

    ``description`` is not documentation: it is the entire instruction the
    model gets about this argument, so it is written for a reader who knows
    nothing about the system.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=_NAME_PATTERN)
    type: ParameterType
    description: str = Field(min_length=1)
    required: bool = True
    choices: tuple[str, ...] = Field(
        default=(), description="Permitted values; empty means unrestricted"
    )

    @model_validator(mode="after")
    def _only_a_string_has_choices(self) -> ToolParameter:
        if self.choices and self.type is not ParameterType.STRING:
            raise ValueError(
                f"{self.name!r}: choices apply to a "
                f"{ParameterType.STRING.value} parameter, not "
                f"{self.type.value}"
            )
        return self

    def as_schema(self) -> dict[str, object]:
        """This parameter as a JSON-schema property."""
        schema: dict[str, object] = dict(_JSON_TYPES[self.type])
        schema["description"] = self.description
        if self.choices:
            schema["enum"] = list(self.choices)
        return schema


#: JSON-schema fragment per declared type. A timestamp is a formatted string,
#: which is what an OpenAI-compatible tool schema can express.
_JSON_TYPES: Mapping[ParameterType, Mapping[str, str]] = MappingProxyType(
    {
        ParameterType.STRING: MappingProxyType({"type": "string"}),
        ParameterType.NUMBER: MappingProxyType({"type": "number"}),
        ParameterType.INTEGER: MappingProxyType({"type": "integer"}),
        ParameterType.BOOLEAN: MappingProxyType({"type": "boolean"}),
        ParameterType.TIMESTAMP: MappingProxyType(
            {"type": "string", "format": "date-time"}
        ),
    }
)


def _coerce(parameter: ToolParameter, value: ArgumentValue) -> ArgumentValue:
    """Check one argument against its declared type.

    :returns: the value, normalised where the declaration allows it -- an
        integer parameter accepts ``3.0`` because JSON has one number type.
    :raises ToolArgumentError: if the value cannot be that type.
    """
    kind = parameter.type
    if kind is ParameterType.BOOLEAN:
        if not isinstance(value, bool):
            raise ToolArgumentError(f"{parameter.name!r} must be true or false")
        return value

    # bool is a subclass of int, so it would otherwise pass as a number.
    if isinstance(value, bool):
        raise ToolArgumentError(
            f"{parameter.name!r} must be a {kind.value}, got a boolean"
        )

    if kind is ParameterType.INTEGER:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ToolArgumentError(f"{parameter.name!r} must be a whole number")

    if kind is ParameterType.NUMBER:
        if isinstance(value, (int, float)):
            return float(value)
        raise ToolArgumentError(f"{parameter.name!r} must be a number")

    if not isinstance(value, str) or not value:
        raise ToolArgumentError(f"{parameter.name!r} must be a non-empty string")

    if kind is ParameterType.TIMESTAMP:
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ToolArgumentError(
                f"{parameter.name!r} must be an ISO-8601 date and time, "
                f"got {value!r}"
            ) from exc
        return value

    if parameter.choices and value not in parameter.choices:
        raise ToolArgumentError(
            f"{parameter.name!r} must be one of {list(parameter.choices)}, "
            f"got {value!r}"
        )
    return value


class ToolSpec(BaseModel):
    """One tool, as declared to a model and as validated against.

    The same object serves both directions, deliberately. If the surface the
    model is shown were built separately from the surface the arguments are
    checked against, the two would drift and the drift would look like the
    model hallucinating.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=_NAME_PATTERN)
    purpose: str = Field(min_length=1, description="What the model is told it does")
    effect: ToolEffect
    parameters: tuple[ToolParameter, ...] = ()

    @model_validator(mode="after")
    def _parameter_names_are_unique(self) -> ToolSpec:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name!r} declares a parameter twice: {names}")
        return self

    @property
    def requires_confirmation(self) -> bool:
        """Whether FR-74's gate applies to this tool.

        Only ``COMMIT`` tools. Prompting for everything is the failure mode
        that looks like caution: a person asked to approve each calendar
        entry learns to approve without reading, and then the one prompt that
        mattered gets the same reflex.
        """
        return self.effect is ToolEffect.COMMIT

    def as_schema(self) -> dict[str, object]:
        """The tool in OpenAI-compatible function-calling form.

        This is what the model is shown -- all of it. Nothing about the
        provider, the transport, or the confirmation flow appears here,
        because none of it is the model's business (FR-70, FR-71).
        """
        required = [
            parameter.name for parameter in self.parameters if parameter.required
        ]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.purpose,
                "parameters": {
                    "type": "object",
                    "properties": {
                        parameter.name: parameter.as_schema()
                        for parameter in self.parameters
                    },
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def validate_arguments(
        self, arguments: Mapping[str, ArgumentValue]
    ) -> Mapping[str, ArgumentValue]:
        """Check arguments against the declaration before any provider is reached.

        :returns: the accepted arguments, normalised, as an immutable mapping.
        :raises ToolArgumentError: naming the parameter at fault. An unknown
            argument is an error rather than a value to drop, for the reason
            every schema here sets ``extra="forbid"``: a misspelled argument
            silently ignored is a tool that quietly did the wrong thing.
        """
        declared = {parameter.name: parameter for parameter in self.parameters}

        unknown = sorted(set(arguments) - set(declared))
        if unknown:
            raise ToolArgumentError(
                f"{self.name!r} does not take {unknown}; "
                f"it takes {sorted(declared)}"
            )

        accepted: dict[str, ArgumentValue] = {}
        for name, parameter in declared.items():
            if name not in arguments or arguments[name] is None:
                if parameter.required:
                    raise ToolArgumentError(f"{self.name!r} requires {name!r}")
                continue
            accepted[name] = _coerce(parameter, arguments[name])
        return MappingProxyType(accepted)


# --- Wire messages ---------------------------------------------------------


class ToolInvocation(TimestampedMessage):
    """One tool call, on its way to the executor.

    Everything is published to ``space/assist/proposed``, including the calls
    the model is entitled to make on its own: the executor owns the providers,
    so nothing reaches a calendar without crossing the blackboard, and the
    audit trail is a consequence of the architecture rather than a thing each
    call site remembers to write (FR-75).

    A ``COMMIT`` invocation is the only kind that stops there. It is
    republished verbatim to ``space/assist/confirmed`` once the occupant has
    agreed (FR-74). The two topics carry the same schema on purpose:
    confirmation is an act by a person, and encoding it as a field would let a
    publisher assert its own approval.

    ``expires_ts`` is the same idea as ``Goal.expires_ts``. A confirmation
    that arrives an hour after the question was asked must not book anything,
    and a stale invocation is refused rather than run late.
    """

    invocation_id: str = Field(
        min_length=1, description="Correlates a result back to this request"
    )
    tool: str = Field(pattern=_NAME_PATTERN)
    arguments: Mapping[str, ArgumentValue] = Field(default_factory=dict)
    requester: ToolRequester
    rationale: str = Field(
        default="", description="What was asked for, in the occupant's terms"
    )
    expires_ts: float = Field(
        gt=0.0, description="After this, confirmation no longer authorises the run"
    )

    @field_validator("arguments", mode="after")
    @classmethod
    def _freeze_arguments(
        cls, value: Mapping[str, ArgumentValue]
    ) -> Mapping[str, ArgumentValue]:
        return MappingProxyType(dict(value))

    @field_serializer("arguments")
    def _serialise_arguments(
        self, value: Mapping[str, ArgumentValue]
    ) -> dict[str, ArgumentValue]:
        return dict(value)


class ToolResult(TimestampedMessage):
    """What came of an invocation, including the refusals (FR-75).

    ``provider`` names the implementation that ran, and ``simulated`` says
    whether the effect was real. Both exist for FR-55: a mock booking has to
    be identifiable as a mock in every log line and every sentence shown to
    an occupant, and a field carrying that is how it stops depending on
    someone remembering to write it.
    """

    invocation_id: str = Field(min_length=1)
    tool: str = Field(pattern=_NAME_PATTERN)
    status: ToolStatus
    message: str = Field(
        default="", description="One plain sentence for the occupant or the log"
    )
    detail: Mapping[str, ArgumentValue] = Field(default_factory=dict)
    provider: str = Field(
        default="", description="Implementation that ran it; empty if none did"
    )
    simulated: bool = False

    @field_validator("detail", mode="after")
    @classmethod
    def _freeze_detail(
        cls, value: Mapping[str, ArgumentValue]
    ) -> Mapping[str, ArgumentValue]:
        return MappingProxyType(dict(value))

    @field_serializer("detail")
    def _serialise_detail(
        self, value: Mapping[str, ArgumentValue]
    ) -> dict[str, ArgumentValue]:
        return dict(value)

    @model_validator(mode="after")
    def _something_must_have_run_for_the_result_to_be_ok(self) -> ToolResult:
        """A successful result with no provider would be a claim with nothing
        behind it, and a simulated one with no provider hides which mock
        answered (FR-55)."""
        if self.status is ToolStatus.OK and not self.provider:
            raise ValueError(f"an {ToolStatus.OK.value} result must name a provider")
        if self.simulated and not self.provider:
            raise ValueError("a simulated result must name the provider that ran")
        return self


class ToolCatalogue(TimestampedMessage):
    """The declared tool surface, retained so it can be read rather than assumed.

    Published to ``space/assist/catalogue``. The console renders its
    confirmation prompts from this, and an examiner can ask what the model is
    allowed to request without reading any code (FR-60).
    """

    tools: tuple[ToolSpec, ...] = ()

    @model_validator(mode="after")
    def _names_are_unique(self) -> ToolCatalogue:
        names = [spec.name for spec in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"a tool is declared twice: {sorted(names)}")
        return self


# --- Provider side ---------------------------------------------------------


class ProviderOutcome(BaseModel):
    """What a provider hands back when it succeeded.

    A sentence and some detail, not a :class:`ToolResult`: the provider does
    not know the invocation id, does not hold a clock, and must not be able
    to report its own status.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    message: str = Field(min_length=1, description="One sentence for the occupant")
    detail: Mapping[str, ArgumentValue] = Field(default_factory=dict)

    @field_validator("detail", mode="after")
    @classmethod
    def _freeze_detail(
        cls, value: Mapping[str, ArgumentValue]
    ) -> Mapping[str, ArgumentValue]:
        return MappingProxyType(dict(value))


@runtime_checkable
class ToolProvider(Protocol):
    """Whatever actually performs a tool.

    The only kind of object in the system permitted to know that Google
    Calendar, a local file, or a mock booking endpoint exists. One provider
    may serve several tools -- a calendar serves both writing an entry and
    reading one back -- which is why :meth:`invoke` is told the name.

    Implementations live outside ``src/common/``: this is a declaration so
    that the registry depends on the abstraction and the concrete client
    depends on it too (FR-71).
    """

    @property
    def name(self) -> str:
        """Short identifier for the log and for ``ToolResult.provider``."""

    @property
    def simulated(self) -> bool:
        """Whether the effect is real. A mock must answer True (FR-55)."""

    def invoke(
        self, tool: str, arguments: Mapping[str, ArgumentValue]
    ) -> ProviderOutcome:
        """Perform the action.

        :raises Exception: any failure. The registry wraps it; a provider is
            not expected to know how failures are reported.
        """


# --- The declared surface --------------------------------------------------

SCHEDULE_EVENT = ToolSpec(
    name="schedule_event",
    purpose=(
        "Put an entry in the occupant's calendar. Use this when they ask to "
        "schedule or add a meeting, reminder or appointment. It is their own "
        "calendar and the entry can be removed again, so do it rather than "
        "asking whether to."
    ),
    effect=ToolEffect.WRITE,
    parameters=(
        ToolParameter(
            name="starts_at",
            type=ParameterType.TIMESTAMP,
            description="When it starts, as an ISO-8601 local date and time.",
        ),
        ToolParameter(
            name="subject",
            type=ParameterType.STRING,
            description="Short title for the entry, in the occupant's own words.",
        ),
        ToolParameter(
            name="duration_min",
            type=ParameterType.INTEGER,
            description=(
                "How long it lasts, in minutes. Omit if they did not say; "
                f"{DEFAULT_EVENT_DURATION_MIN} minutes is assumed."
            ),
            required=False,
        ),
    ),
)

GET_EVENTS = ToolSpec(
    name="get_events",
    purpose=(
        "List the occupant's calendar entries between two times. Use this to "
        "answer a question about what is planned. It changes nothing."
    ),
    effect=ToolEffect.READ,
    parameters=(
        ToolParameter(
            name="from_time",
            type=ParameterType.TIMESTAMP,
            description="Start of the window, as an ISO-8601 local date and time.",
        ),
        ToolParameter(
            name="to_time",
            type=ParameterType.TIMESTAMP,
            description="End of the window, as an ISO-8601 local date and time.",
        ),
    ),
)

BOOK_TRAVEL = ToolSpec(
    name="book_travel",
    purpose=(
        "Request a flight or a hotel. You cannot complete this yourself: it "
        "is put to the occupant for confirmation first. Every booking reaches "
        "a mock endpoint and no real reservation is ever made; say so when "
        "reporting back."
    ),
    effect=ToolEffect.COMMIT,
    parameters=(
        ToolParameter(
            name="kind",
            type=ParameterType.STRING,
            description="What to book.",
            choices=("flight", "hotel"),
        ),
        ToolParameter(
            name="destination",
            type=ParameterType.STRING,
            description="Where they are going.",
        ),
        ToolParameter(
            name="depart_on",
            type=ParameterType.TIMESTAMP,
            description="Departure or check-in date, as an ISO-8601 date and time.",
        ),
        ToolParameter(
            name="origin",
            type=ParameterType.STRING,
            description="Where they are travelling from. Omit for a hotel.",
            required=False,
        ),
        ToolParameter(
            name="nights",
            type=ParameterType.INTEGER,
            description="Nights to stay. Omit for a flight.",
            required=False,
        ),
    ),
)

#: The declared assistance surface (section 5.7.6). Adding a tool is one
#: entry here plus a provider binding, and touches nothing else (FR-72).
ASSISTANCE_TOOLS: tuple[ToolSpec, ...] = (SCHEDULE_EVENT, GET_EVENTS, BOOK_TRAVEL)


# --- Registry --------------------------------------------------------------


class ToolRegistry:
    """Holds the declared tools, binds providers to them, and gates every call.

    The whole point of the class is that the two halves never meet: a model
    reads :meth:`schemas` and a provider implements :class:`ToolProvider`, and
    neither knows the other exists. Everything between them -- argument
    validation, the confirmation gate, expiry, failure reporting -- happens
    once, here, so it cannot be forgotten at one call site out of three.
    """

    def __init__(
        self, clock: Clock, specs: Iterable[ToolSpec] = ASSISTANCE_TOOLS
    ) -> None:
        self._clock = clock
        self._specs: dict[str, ToolSpec] = {}
        self._providers: dict[str, ToolProvider] = {}
        for spec in specs:
            self.declare(spec)

    def declare(self, spec: ToolSpec) -> None:
        """Add a tool to the catalogue.

        :raises ValueError: if the name is already declared. Two tools of one
            name would make which implementation runs depend on registration
            order.
        """
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name!r} is already declared")
        self._specs[spec.name] = spec

    def bind(self, name: str, provider: ToolProvider) -> None:
        """Attach the implementation that fulfils a declared tool.

        :raises UnknownToolError: if nothing declares that name. Binding a
            provider to an undeclared tool would leave it unreachable and
            look like a broken provider.
        """
        self.spec(name)
        self._providers[name] = provider

    def spec(self, name: str) -> ToolSpec:
        """The declaration for one tool.

        :raises UnknownToolError: if it is not in the catalogue.
        """
        try:
            return self._specs[name]
        except KeyError as exc:
            raise UnknownToolError(
                f"no tool named {name!r}; available: {sorted(self._specs)}"
            ) from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    def schemas(self) -> tuple[dict[str, object], ...]:
        """The tool surface as a model is shown it (FR-70)."""
        return tuple(spec.as_schema() for spec in self._specs.values())

    def catalogue(self) -> ToolCatalogue:
        """The retained catalogue message for ``space/assist/catalogue``."""
        return ToolCatalogue(ts=self._clock.now(), tools=self.specs())

    def invoke(self, invocation: ToolInvocation, *, confirmed: bool) -> ToolResult:
        """Run one invocation, or refuse it with a reason.

        Never raises for a refusal. FR-75 requires every outcome to reach the
        blackboard, and an exception escaping into a service loop reaches
        nobody; the refusal is translated into a published result instead.
        A provider crashing is contained the same way, because a broken
        calendar must not take the process down with it.

        :param confirmed: whether an occupant has agreed to this. Derived
            from the topic it arrived on, not from the message: a publisher
            must not be able to assert its own approval. Ignored for anything
            but a ``COMMIT`` tool, which is the only kind that asks.
        """
        try:
            return self._run(invocation, confirmed=confirmed)
        except ToolError as refusal:
            LOGGER.info(
                "tool %s refused (%s): %s",
                invocation.tool,
                refusal.status.value,
                refusal,
            )
            return self._refuse(invocation, refusal)

    def _run(self, invocation: ToolInvocation, *, confirmed: bool) -> ToolResult:
        spec = self.spec(invocation.tool)

        if self._clock.now() > invocation.expires_ts:
            raise InvocationExpiredError(
                f"{invocation.tool!r} expired; it was authorised until "
                f"{invocation.expires_ts}"
            )

        # Before the gate: nobody should be asked to confirm a malformed
        # request, and a bad argument is the system's fault to report.
        arguments = spec.validate_arguments(invocation.arguments)

        if spec.requires_confirmation and not confirmed:
            raise ConfirmationRequiredError(
                f"{invocation.tool!r} commits you to something outside the "
                f"system and needs confirming"
            )

        provider = self._providers.get(invocation.tool)
        if provider is None:
            raise ToolUnavailableError(
                f"{invocation.tool!r} is declared but nothing is bound to it"
            )

        try:
            outcome = provider.invoke(invocation.tool, arguments)
        except Exception as exc:
            LOGGER.warning(
                "provider %s failed on %s: %s", provider.name, invocation.tool, exc
            )
            raise ToolFailedError(f"{provider.name} could not do it: {exc}") from exc

        return ToolResult(
            ts=self._clock.now(),
            invocation_id=invocation.invocation_id,
            tool=invocation.tool,
            status=ToolStatus.OK,
            message=outcome.message,
            detail=outcome.detail,
            provider=provider.name,
            simulated=provider.simulated,
        )

    def _refuse(self, invocation: ToolInvocation, refusal: ToolError) -> ToolResult:
        return ToolResult(
            ts=self._clock.now(),
            invocation_id=invocation.invocation_id,
            tool=invocation.tool,
            status=refusal.status,
            message=str(refusal),
        )
