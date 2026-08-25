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

#: Root of the blackboard tree. ``space/#`` subscribes to everything, which is
#: what the recorder (FR-62) and ``mosquitto_sub`` inspection (FR-60) use.
TOPIC_ROOT = "space"
ALL_TOPICS = "space/#"

#: MQTT single-level wildcard, substituted for a parameter when subscribing.
_SINGLE_LEVEL_WILDCARD = "+"

#: A topic level must not be empty and must not contain a separator or a
#: wildcard, or it would silently reshape the topic tree.
_FORBIDDEN_IN_PARAMETER = ("/", "+", "#")

_PARAMETER_PATTERN = re.compile(r"\{[a-z_]+\}")


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

    def format(self, **parameters: str) -> str:
        """Build a concrete topic.

        :raises TopicParameterError: if a parameter is empty or contains a
            topic separator or wildcard.
        :raises KeyError: if a required parameter is missing.
        """
        for name, value in parameters.items():
            _require_valid_parameter(name, value)
        return self.pattern.format(**parameters)

    def wildcard(self) -> str:
        """Subscription form, with every parameter replaced by ``+``."""
        return _PARAMETER_PATTERN.sub(_SINGLE_LEVEL_WILDCARD, self.pattern)


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

# --- Audit (FR-46, FR-62) --------------------------------------------------

AUDIT_VALIDATION = TopicSpec("space/audit/validation", Qos.AT_LEAST_ONCE, retain=False)
AUDIT_REASONING = TopicSpec("space/audit/reasoning", Qos.AT_LEAST_ONCE, retain=False)

# --- Fault injection (FR-31) -----------------------------------------------

#: Not in the DESIGN.md section 6.1 table. Added so every fault class is
#: triggerable from tools/inject.py without editing or restarting production
#: code paths, which is what FR-31 requires. Layer 1 honours these; nothing
#: above Layer 1 subscribes, so the injection path is identical in simulation
#: and on hardware.
FAULT_INJECT = TopicSpec("space/inject/{subject}", Qos.AT_LEAST_ONCE, retain=False)
