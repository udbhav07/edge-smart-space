"""Canonical MQTT topic definitions for the blackboard.

Every topic string in the system originates here. Nothing else may write a
topic literal, because the blackboard is the only coupling between
components (DESIGN.md section 4.4) and a typo in a topic string fails
silently: the publisher succeeds and no subscriber ever hears it.

Quality-of-service and retention are part of the contract, not a caller
decision, so they travel with the topic in :class:`TopicSpec`. Retained
topics are the ones a late-joining subscriber must be able to read
immediately in order to know current system state (FR-61).

Topic table: DESIGN.md section 6.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

#: Root of the blackboard tree. ``space/#`` subscribes to everything, which is
#: what the recorder (FR-62) and ``mosquitto_sub`` inspection (FR-60) use.
TOPIC_ROOT = "space"
ALL_TOPICS = "space/#"

#: MQTT single-level wildcard, substituted for a parameter when subscribing.
_SINGLE_LEVEL_WILDCARD = "+"

#: A topic level must not be empty and must not contain a separator or a
#: wildcard, or it would silently reshape the topic tree.
_FORBIDDEN_IN_PARAMETER = ("/", "+", "#")

_PARAMETER_PATTERN = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class Qos(IntEnum):
    """MQTT quality of service.

    ``AT_MOST_ONCE`` is used for high-rate streams where the next sample
    supersedes a lost one. ``AT_LEAST_ONCE`` is used for anything a component
    must not miss: faults, modes, goals, commands, audit records.
    """

    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1


class TopicParameterError(ValueError):
    """A topic parameter would produce a malformed or ambiguous topic."""


def _require_valid_parameter(name: str, value: str) -> None:
    """Reject a parameter that cannot legally occupy one topic level.

    :raises TopicParameterError: if empty or containing ``/``, ``+`` or ``#``.
    """
    if not value:
        raise TopicParameterError(f"topic parameter {name!r} must not be empty")
    for character in _FORBIDDEN_IN_PARAMETER:
        if character in value:
            raise TopicParameterError(
                f"topic parameter {name!r} must not contain {character!r}, "
                f"got {value!r}"
            )


@dataclass(frozen=True)
class TopicSpec:
    """One topic, with the delivery guarantees that belong to it.

    Frozen so a component cannot alter the contract for everyone else.
    """

    pattern: str
    qos: Qos
    retain: bool

    @property
    def parameter_names(self) -> frozenset[str]:
        """Names this pattern requires, in no particular order."""
        return frozenset(
            match.group(1) for match in _PARAMETER_PATTERN.finditer(self.pattern)
        )

    def format(self, **parameters: str) -> str:
        """Build a concrete topic.

        A name the pattern does not use is an error rather than a no-op.
        ``str.format`` would ignore it and hand back a plausible-looking
        topic, which is exactly the silent publisher/subscriber mismatch this
        module exists to prevent.

        :raises TopicParameterError: if a parameter is unknown, missing,
            empty, or contains a topic separator or wildcard.
        """
        expected = self.parameter_names
        supplied = frozenset(parameters)

        unknown = supplied - expected
        if unknown:
            raise TopicParameterError(
                f"{self.pattern!r} takes {sorted(expected)}, "
                f"got unknown parameter(s) {sorted(unknown)}"
            )

        missing = expected - supplied
        if missing:
            raise TopicParameterError(
                f"{self.pattern!r} requires parameter(s) {sorted(missing)}"
            )

        for name, value in parameters.items():
            _require_valid_parameter(name, value)
        return self.pattern.format(**parameters)

    def wildcard(self) -> str:
        """Subscription form, with every parameter replaced by ``+``."""
        return _PARAMETER_PATTERN.sub(_SINGLE_LEVEL_WILDCARD, self.pattern)


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """A pattern as a matcher for concrete topics.

    ``space/sensor/{sensor_id}/state`` matches ``space/sensor/temp_01/state``
    and nothing with a different shape. Each parameter stands for exactly one
    topic level, which is the same rule :func:`_require_valid_parameter`
    enforces when a topic is built.
    """
    # Split on the parameter markers and escape only the literal parts, so
    # nothing depends on which characters re.escape happens to escape.
    parts = _PARAMETER_PATTERN.split(pattern)
    rebuilt = [
        re.escape(part) if index % 2 == 0 else "[^/]+"
        for index, part in enumerate(parts)
    ]
    return re.compile("^" + "".join(rebuilt) + "$")


@lru_cache(maxsize=1)
def _declared_specs() -> tuple[tuple[TopicSpec, re.Pattern[str]], ...]:
    """Every declared topic with its matcher, built once.

    Cached because the replayer asks per message, and recompiling the whole
    table for each line of a recording would make replay cost grow with the
    size of the topic tree for no reason.
    """
    return tuple(
        (value, _pattern_regex(value.pattern))
        for value in globals().values()
        if isinstance(value, TopicSpec)
    )


def spec_for(topic: str) -> TopicSpec | None:
    """The declared contract for a concrete topic, if there is one.

    The replayer needs this: a recording holds concrete topics and encoded
    payloads, and republishing them has to use the quality of service and
    retention the topic declares rather than whatever the recorder happened to
    observe. Delivery is the topic's business even when the message is a
    recording of one.

    :returns: the matching spec, or None for a topic this build does not
        declare. None rather than a raise: a recording made by a newer build
        is a thing to report and skip, not a crash.
    """
    for spec, matcher in _declared_specs():
        if matcher.match(topic):
            return spec
    return None


# --- Sensing (FR-01, FR-02, FR-03) -----------------------------------------

SENSOR_STATE = TopicSpec(
    "space/sensor/{sensor_id}/state", Qos.AT_MOST_ONCE, retain=False
)
SENSOR_HEALTH = TopicSpec(
    "space/sensor/{sensor_id}/health", Qos.AT_LEAST_ONCE, retain=True
)

# --- Estimation (FR-04, FR-05) ---------------------------------------------

ESTIMATE_THERMAL = TopicSpec("space/estimate/thermal", Qos.AT_MOST_ONCE, retain=True)
ESTIMATE_COEFFICIENTS = TopicSpec(
    "space/estimate/coefficients", Qos.AT_LEAST_ONCE, retain=True
)

# --- Faults and mode (FR-20 to FR-31, FR-61) -------------------------------

FAULT = TopicSpec("space/fault/{fault_id}", Qos.AT_LEAST_ONCE, retain=True)
SYSTEM_MODE = TopicSpec("space/system/mode", Qos.AT_LEAST_ONCE, retain=True)

# --- Goals (FR-40, FR-45) --------------------------------------------------

GOAL_PROPOSED = TopicSpec("space/goal/proposed", Qos.AT_LEAST_ONCE, retain=False)
GOAL_ACTIVE = TopicSpec("space/goal/active", Qos.AT_LEAST_ONCE, retain=True)

# --- Actuation (FR-12, FR-15) ----------------------------------------------

ACTUATOR_COMMAND = TopicSpec(
    "space/actuator/{actuator_id}/command", Qos.AT_LEAST_ONCE, retain=False
)
ACTUATOR_STATE = TopicSpec(
    "space/actuator/{actuator_id}/state", Qos.AT_LEAST_ONCE, retain=True
)

#: The one physical actuator (DESIGN.md section 2.1). Simulated actuators use
#: the same topics under their own id and carry ``simulated: true``.
AIR_CONDITIONER_ID = "ac"

# --- Speech and context (FR-53) --------------------------------------------

CONTEXT_PREFERENCE = TopicSpec(
    "space/context/preference", Qos.AT_LEAST_ONCE, retain=False
)

#: Text addressed to the room, spoken or typed. Not retained: a restarting
#: reasoning process must not answer yesterday's request again.
CONTEXT_UTTERANCE = TopicSpec(
    "space/context/utterance", Qos.AT_LEAST_ONCE, retain=False
)

#: The pricing band in force (FR-16). Retained, because a supervisor starting
#: mid-peak has to know it is mid-peak without waiting for the next change.
CONTEXT_TARIFF = TopicSpec("space/context/tariff", Qos.AT_LEAST_ONCE, retain=True)

# --- Diagnosis (FR-25) -----------------------------------------------------

#: The Fault Diagnosis call's explanation of a fault. Published after the mode
#: has changed and never before it (FR-26). Deliberately outside
#: ``space/fault/``: a pattern there would match ``space/fault/{fault_id}``
#: and a recording would replay an explanation as a fault.
DIAGNOSIS = TopicSpec("space/diagnosis", Qos.AT_LEAST_ONCE, retain=False)

# --- Assistance tools (FR-70 to FR-75) -------------------------------------

#: Every tool call the reasoning layer makes, including the ones it is
#: entitled to run on its own: the executor owns the providers, so nothing
#: reaches a calendar without crossing the blackboard first.
ASSIST_PROPOSED = TopicSpec("space/assist/proposed", Qos.AT_LEAST_ONCE, retain=False)

#: A COMMIT invocation, republished by whoever obtained the occupant's
#: agreement (FR-74). Confirmation is an act by a person, so it is carried by
#: the topic rather than by a field a publisher could set for itself.
ASSIST_CONFIRMED = TopicSpec("space/assist/confirmed", Qos.AT_LEAST_ONCE, retain=False)

#: Every outcome, refusals included (FR-75).
ASSIST_RESULT = TopicSpec("space/assist/result", Qos.AT_LEAST_ONCE, retain=False)

#: The declared surface, retained so a late subscriber -- or an examiner --
#: can read what the reasoning layer is allowed to ask for (FR-60, FR-70).
ASSIST_CATALOGUE = TopicSpec("space/assist/catalogue", Qos.AT_LEAST_ONCE, retain=True)

#: An operator clearing SAFE_HOLD (section 5.6). Deliberately *not* retained:
#: a retained reset would be redelivered on every reconnect and re-clear a hold
#: nobody had looked at, which is the opposite of a manual acknowledgement.
SYSTEM_RESET = TopicSpec("space/system/reset", Qos.AT_LEAST_ONCE, retain=False)

# --- Fault injection (FR-31) -----------------------------------------------

#: The one topic that travels down into Layer 1. Retained, because it is the
#: answer to "what is being injected right now": an examiner can read it, and
#: a restarted adapter resumes the state the operator last asked for rather
#: than quietly healing a fault nobody cleared.
INJECT = TopicSpec("space/inject/{subject}", Qos.AT_LEAST_ONCE, retain=True)

# --- Audit (FR-46, FR-62) --------------------------------------------------

AUDIT_VALIDATION = TopicSpec("space/audit/validation", Qos.AT_LEAST_ONCE, retain=False)
AUDIT_REASONING = TopicSpec("space/audit/reasoning", Qos.AT_LEAST_ONCE, retain=False)
