"""Fault Diagnosis: one schema-constrained call per fault (FR-25, FR-43).

Single-shot and tool-less (section 5.7.1). It is given the fault the detector
bank raised -- which detector, on what, with what evidence -- and asked for
section 6.3's object: a hypothesis from a fixed set, a confidence, the
evidence it rests on, a recommended mode, and a sentence for the occupant.

**It never decides anything.** The mode has already changed by the time this
runs, in another process, within FR-26's two seconds (section 5.9.2's ``par``
block). What this adds is the explanation. ``recommended_mode`` is advisory
and is checked against section 5.6's legal transitions; an illegal one fails
post-decode validation (FR-44) along with everything else in the answer.

**It always answers.** A failed or unavailable call yields the generic
notification section 5.7.1 specifies, built from the detector's own finding
and published with ``generated: false``. A fault with a plain explanation is
better than one with none, and marking which is which keeps a model's words
from being mistaken for the system's.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from src.common.clock import Clock
from src.common.schemas import (
    CallSite,
    DetectorId,
    DiagnosisConfidence,
    FaultDiagnosis,
    FaultEvent,
    Hypothesis,
    Mode,
    ReasoningOutcome,
    is_legal_transition,
)
from src.reasoning.audit import ReasoningAudit, Trace
from src.reasoning.endpoint import ChatEndpoint, ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Most evidence items kept from a model's answer, and the longest each may be.
MAX_EVIDENCE_ITEMS = 8
MAX_EVIDENCE_CHARS = 64

#: What each detector's finding means, when no model is there to say so.
_HYPOTHESIS_BY_DETECTOR: dict[DetectorId, Hypothesis] = {
    DetectorId.D1_DROPOUT: Hypothesis.SENSOR_DROPOUT,
    DetectorId.D2_STUCK_AT: Hypothesis.SENSOR_STUCK,
    DetectorId.D3_OUT_OF_RANGE: Hypothesis.SENSOR_OUT_OF_RANGE,
    DetectorId.D4_DRIFT: Hypothesis.SENSOR_DRIFT,
    DetectorId.D5_ACTUATOR_NO_RESPONSE: Hypothesis.ACTUATOR_NO_RESPONSE,
    DetectorId.MODEL_DIVERGENCE: Hypothesis.MODEL_DIVERGENCE,
}

#: The plain sentence for each, completed with the subject.
_PLAIN_CAUSE: dict[Hypothesis, str] = {
    Hypothesis.SENSOR_DROPOUT: "has stopped reporting",
    Hypothesis.SENSOR_STUCK: "appears stuck at one value",
    Hypothesis.SENSOR_OUT_OF_RANGE: "is reporting values that are physically impossible",
    Hypothesis.SENSOR_DRIFT: "is drifting away from what the room model expects",
    Hypothesis.ACTUATOR_NO_RESPONSE: "is not cooling the room when told to",
    Hypothesis.MODEL_DIVERGENCE: "model has stopped fitting the room",
    Hypothesis.UNKNOWN: "has a fault the system could not classify",
}

#: A detector's confidence, coarsened. A number to two places would claim a
#: precision nothing here has.
_HIGH_CONFIDENCE = 0.9
_MEDIUM_CONFIDENCE = 0.6

DIAGNOSIS_PROMPT = (
    "You explain a fault in a smart room's climate system to its occupant. "
    "The system has already reacted; you do not control anything. You are "
    "given the fault a detector raised. Reply with a JSON object and nothing "
    "else, with keys: "
    '"primary_hypothesis" (one of: '
    + ", ".join(f'"{h.value}"' for h in Hypothesis)
    + '), "confidence" ("low", "medium" or "high"), '
    '"supporting_evidence" (a list of short snake_case names of the evidence '
    'you relied on), "recommended_mode" (one of: '
    + ", ".join(f'"{m.value}"' for m in Mode)
    + '), and "user_message" (one or two plain sentences for the occupant '
    "saying what is wrong and what the system is doing about it)."
)


class FaultDiagnoser:
    """Explains one fault, and always produces something to publish."""

    def __init__(
        self,
        clock: Clock,
        endpoint: ChatEndpoint,
        audit: ReasoningAudit,
    ) -> None:
        self._clock = clock
        self._endpoint = endpoint
        self._audit = audit

    def diagnose(self, event: FaultEvent, current_mode: Mode) -> FaultDiagnosis:
        """Ask the model, check what it says, and fall back if it fails."""
        features = self._features(event, current_mode)
        trace = Trace(CallSite.FAULT_DIAGNOSIS, f"fault {event.fault_id}", features)
        try:
            completion = self._endpoint.complete(
                [
                    {"role": "system", "content": DIAGNOSIS_PROMPT},
                    {"role": "user", "content": features},
                ],
                json_output=True,
            )
        except ReasoningUnavailableError as exc:
            self._audit.publish(trace, ReasoningOutcome.UNAVAILABLE, reason=str(exc))
            return self.generic(event, current_mode)

        trace.add(completion)
        diagnosis, problem = self._validate(completion.content, event, current_mode)
        if diagnosis is None:
            LOGGER.warning("discarding diagnosis of %s: %s", event.fault_id, problem)
            self._audit.publish(trace, ReasoningOutcome.DISCARDED, reason=problem)
            return self.generic(event, current_mode)

        self._audit.publish(
            trace,
            ReasoningOutcome.APPLIED,
            applied=f"{diagnosis.primary_hypothesis.value}: {diagnosis.user_message}",
        )
        return diagnosis

    @staticmethod
    def _features(event: FaultEvent, current_mode: Mode) -> str:
        """The structured feature vector FR-25 names, as the model sees it."""
        return json.dumps(
            {
                "detector": event.detector.value,
                "subject": event.subject,
                "class": event.fault_class.value,
                "detector_confidence": round(event.confidence, 2),
                "evidence": dict(event.evidence),
                "mode_impact": event.mode_impact.value,
                "current_mode": current_mode.value,
            }
        )

    def _validate(
        self, raw: str, event: FaultEvent, current_mode: Mode
    ) -> tuple[FaultDiagnosis | None, str]:
        """Post-decode semantic validation (FR-44). Returns the reason on failure."""
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"not JSON: {exc}"
        if not isinstance(decoded, dict):
            return None, "not a JSON object"

        evidence = decoded.get("supporting_evidence", [])
        if not isinstance(evidence, list) or not all(
            isinstance(item, str) for item in evidence
        ):
            return None, "supporting_evidence is not a list of strings"

        try:
            diagnosis = FaultDiagnosis(
                ts=self._clock.now(),
                fault_id=event.fault_id,
                primary_hypothesis=Hypothesis(decoded.get("primary_hypothesis")),
                confidence=DiagnosisConfidence(decoded.get("confidence")),
                supporting_evidence=tuple(
                    item[:MAX_EVIDENCE_CHARS] for item in evidence[:MAX_EVIDENCE_ITEMS]
                ),
                recommended_mode=Mode(decoded.get("recommended_mode")),
                user_message=str(decoded.get("user_message", "")).strip(),
                generated=True,
            )
        except (ValidationError, ValueError) as exc:
            return None, f"fails the schema: {exc}"

        if not is_legal_transition(current_mode, diagnosis.recommended_mode):
            return None, (
                f"recommends {diagnosis.recommended_mode.value}, which is not "
                f"reachable from {current_mode.value} (section 5.6)"
            )
        return diagnosis, ""

    def generic(self, event: FaultEvent, current_mode: Mode) -> FaultDiagnosis:
        """The notification section 5.7.1 specifies when there is no model.

        Built from the detector's own finding, so it is never wrong about
        *what* was found -- only less helpful about why.
        """
        hypothesis = _HYPOTHESIS_BY_DETECTOR.get(event.detector, Hypothesis.UNKNOWN)
        mode = (
            event.mode_impact
            if is_legal_transition(current_mode, event.mode_impact)
            else current_mode
        )
        if event.confidence >= _HIGH_CONFIDENCE:
            confidence = DiagnosisConfidence.HIGH
        elif event.confidence >= _MEDIUM_CONFIDENCE:
            confidence = DiagnosisConfidence.MEDIUM
        else:
            confidence = DiagnosisConfidence.LOW
        return FaultDiagnosis(
            ts=self._clock.now(),
            fault_id=event.fault_id,
            primary_hypothesis=hypothesis,
            confidence=confidence,
            supporting_evidence=tuple(
                name[:MAX_EVIDENCE_CHARS]
                for name in list(event.evidence)[:MAX_EVIDENCE_ITEMS]
            ),
            recommended_mode=mode,
            user_message=(
                f"The {event.subject} {_PLAIN_CAUSE[hypothesis]}. "
                f"The system is in {mode.value.replace('_', ' ').lower()} mode."
            ),
            generated=False,
        )
