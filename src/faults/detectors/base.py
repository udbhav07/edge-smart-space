"""What every detector reports, and how it becomes a published fault.

The detector bank is five independent tests (DESIGN.md section 5.5) with
nothing in common except the shape of their answer. That shape is here so the
aggregator can treat them uniformly without any of them knowing about each
other.

Two decisions drive everything in this module.

**A detector reports a judgment, not an event.** Each detector is asked "what
do you make of this subject, now?" and answers FAULTED, CLEAR, or UNKNOWN. It
is the aggregator that notices a change of answer and turns that into a raised
or cleared fault. A detector that emitted events would have to remember what
it had already emitted, and five detectors would each get that bookkeeping
subtly different.

**UNKNOWN is not CLEAR.** A dropout detector that has never received a reading
has no silence to measure, and a variance window that is not yet full says
nothing about variance. Collapsing either into "no fault" would report a
healthy sensor the system has never actually heard from, which is precisely
the state a demonstration starts in.

Mapping a detector to a fault class and a mode is structural -- it comes from
the state machine in section 5.6, not from a threshold -- so it lives here in
code rather than in config. The numbers each detector compares against are
policy and live in config, as everything re-derived on hardware must (R-04).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from src.common.schemas import DetectorId, FaultClass, FaultEvent, Mode


class Judgment(str, Enum):
    """A detector's answer about one subject at one instant.

    UNKNOWN is first-class for the same reason ``AckStatus.UNKNOWN`` is: the
    absence of evidence is a distinct state from evidence of absence, and code
    that conflates them reports health it has not observed.
    """

    CLEAR = "CLEAR"
    FAULTED = "FAULTED"
    UNKNOWN = "UNKNOWN"


#: What each detector's finding means for fault class and mode, from the state
#: machine in section 5.6. A sensor fault degrades to running on prediction; an
#: actuator fault stops actuation; a diverged model is not something to
#: degrade around, so it holds.
_DETECTOR_IMPACT: Mapping[DetectorId, tuple[FaultClass, Mode]] = MappingProxyType(
    {
        DetectorId.D1_DROPOUT: (FaultClass.SENSOR, Mode.DEGRADED_SENSOR),
        DetectorId.D2_STUCK_AT: (FaultClass.SENSOR, Mode.DEGRADED_SENSOR),
        DetectorId.D3_OUT_OF_RANGE: (FaultClass.SENSOR, Mode.DEGRADED_SENSOR),
        DetectorId.D4_DRIFT: (FaultClass.SENSOR, Mode.DEGRADED_SENSOR),
        DetectorId.D5_ACTUATOR_NO_RESPONSE: (
            FaultClass.ACTUATOR,
            Mode.DEGRADED_ACTUATOR,
        ),
        DetectorId.MODEL_DIVERGENCE: (FaultClass.MODEL, Mode.SAFE_HOLD),
    }
)

#: Short names for fault ids, so an id stays readable in a topic listing.
_DETECTOR_SLUG: Mapping[DetectorId, str] = MappingProxyType(
    {
        DetectorId.D1_DROPOUT: "dropout",
        DetectorId.D2_STUCK_AT: "stuck",
        DetectorId.D3_OUT_OF_RANGE: "range",
        DetectorId.D4_DRIFT: "drift",
        DetectorId.D5_ACTUATOR_NO_RESPONSE: "noresponse",
        DetectorId.MODEL_DIVERGENCE: "divergence",
    }
)

#: Characters removed from a subject id when building a fault id, so the id
#: reads as three fields rather than five (section 6.2 writes ``temp_01`` as
#: ``temp01`` for exactly this reason).
_SUBJECT_SEPARATORS = ("_", "-", ".")


@dataclass(frozen=True)
class Finding:
    """One detector's judgment about one subject, with its evidence.

    Evidence travels with every judgment, not only with a fault: the numbers
    that did *not* trip a threshold are what make a demonstration legible, and
    they are the same numbers either way.

    ``confidence`` is a detector's own strength of belief, not a probability.
    D1's is high because a timeout is unambiguous; D2's reflects how far below
    the threshold the variance sits. It is reported rather than acted on --
    nothing gates a mode transition on it (FR-26).
    """

    detector: DetectorId
    subject: str
    judgment: Judgment
    confidence: float = 0.0
    evidence: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("a finding must name its subject")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        # Frozen protects the binding, not the dictionary behind it. A
        # finding's evidence is the record of why a mode transition happened,
        # so it must not be editable after the fact.
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    @property
    def faulted(self) -> bool:
        """Whether this finding asserts a fault. UNKNOWN does not."""
        return self.judgment is Judgment.FAULTED

    @property
    def fault_class(self) -> FaultClass:
        """What kind of thing this detector speaks about."""
        return _DETECTOR_IMPACT[self.detector][0]

    @property
    def mode_impact(self) -> Mode:
        """The mode this fault alone implies (section 5.6).

        Alone is the operative word: escalation when several faults are active
        at once is the mode manager's decision, not a detector's.
        """
        return _DETECTOR_IMPACT[self.detector][1]


def fault_id(detector: DetectorId, subject: str, detected_ts: float) -> str:
    """Build the id a fault is published under.

    Deterministic from the finding, so the same fault in a replayed recording
    gets the same id (FR-62). Collision would need a detector to clear and
    re-raise on the same subject within one second, which no detector's
    debounce window permits.
    """
    readable_subject = subject
    for separator in _SUBJECT_SEPARATORS:
        readable_subject = readable_subject.replace(separator, "")
    return f"f_{readable_subject}_{_DETECTOR_SLUG[detector]}_{int(detected_ts)}"


def build_fault_event(finding: Finding, detected_ts: float) -> FaultEvent:
    """Turn a faulted finding into the message that goes on the blackboard.

    :raises ValueError: if the finding does not assert a fault. Publishing a
        FaultEvent for a clear or unknown judgment would put a fault on the
        blackboard that no detector claimed, and every consumer downstream
        treats a FaultEvent as a fault.
    """
    if not finding.faulted:
        raise ValueError(
            f"{finding.detector.value} judged {finding.subject!r} "
            f"{finding.judgment.value}; only a fault can be published as one"
        )
    return FaultEvent(
        fault_id=fault_id(finding.detector, finding.subject, detected_ts),
        detector=finding.detector,
        subject=finding.subject,
        fault_class=finding.fault_class,
        confidence=finding.confidence,
        detected_ts=detected_ts,
        evidence=finding.evidence,
        mode_impact=finding.mode_impact,
    )
