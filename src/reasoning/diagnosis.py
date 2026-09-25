"""Fault Diagnosis: explaining a confirmed fault to a person (FR-25, FR-43).

A single-shot, schema-constrained call with no tools (section 5.7.1). It is
invoked once per newly confirmed fault, *after* the mode manager has already
changed the mode: the explanation is allowed to be slow, the response is not
(FR-26, section 5.9.2). Nothing here can change the mode.

The call is given the fault's feature vector -- detector, subject, evidence,
the mode it caused -- and must return the section 6.3 schema. Post-decode
validation (FR-44) discards anything that does not parse or names a value
outside the fixed sets, and ``recommended_mode`` is checked against the state
machine's legal transitions; an illegal recommendation is reported as such and
the detector-derived mode stands.

If the model gives nothing usable, or is unreachable, the fixed notification
for the detector is published instead (section 5.7.1): an occupant is told
something true either way.
"""

from __future__ import annotations

import json
import logging
from collections import deque

from pydantic import ValidationError

from src.common import topics
from src.common.clock import Clock
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    LEGAL_TRANSITIONS,
    DetectorId,
    DiagnosisConfidence,
    FaultDiagnosis,
    FaultEvent,
    Hypothesis,
    Mode,
    ModeState,
    ReasoningCaller,
    ReasoningRecord,
)
from src.reasoning.chat import ChatClient
from src.reasoning.single_shot import ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Faults waiting for an explanation. Bounded.
_QUEUE_LENGTH = 16
#: Fault ids already explained, so a retained fault redelivered on reconnect
#: is not explained twice. Bounded.
_REMEMBERED = 256

DIAGNOSIS_PROMPT = (
    "You explain faults in a smart room's temperature control system to the "
    "person who lives there. Reply with a JSON object and nothing else, with "
    "keys: \"primary_hypothesis\" (one of {hypotheses}), \"confidence\" (one "
    "of low, medium, high), \"supporting_evidence\" (a list of short "
    "snake_case phrases naming the evidence you relied on), "
    "\"recommended_mode\" (one of {modes}), and \"user_message\" (one or two "
    "plain sentences for the occupant: what is wrong and what the system is "
    "doing about it). For a sensor fault the system is running on its room "
    "model instead of the sensor. For an actuator fault the air conditioner "
    "has been put on hold until someone checks it, so the room may warm; "
    "never say the temperature will be kept or maintained. Do not invent "
    "evidence."
)

#: What a person is told when the model gives nothing usable (section 5.7.1).
_GENERIC: dict[DetectorId, tuple[Hypothesis, str]] = {
    DetectorId.D1_DROPOUT: (
        Hypothesis.SENSOR_DROPOUT,
        "The temperature sensor stopped reporting. The room is running on its model for now.",
    ),
    DetectorId.D2_STUCK_AT: (
        Hypothesis.SENSOR_STUCK,
        "The temperature sensor appears stuck. The room is running on its model for now.",
    ),
    DetectorId.D3_OUT_OF_RANGE: (
        Hypothesis.SENSOR_OUT_OF_RANGE,
        "The temperature sensor is reporting impossible values. The room is running on its model for now.",
    ),
    DetectorId.D4_DRIFT: (
        Hypothesis.SENSOR_DRIFT,
        "The temperature sensor seems to be drifting. The room is running on its model for now.",
    ),
    DetectorId.D5_ACTUATOR_NO_RESPONSE: (
        Hypothesis.ACTUATOR_NO_RESPONSE,
        "The air conditioner does not seem to be cooling the room. It is on hold until someone checks it.",
    ),
    DetectorId.MODEL_DIVERGENCE: (
        Hypothesis.MODEL_DIVERGENCE,
        "The room's model stopped making sense. The system is holding safe until reset.",
    ),
}


class FaultDiagnoser:
    """Explains each newly confirmed fault once."""

    def __init__(self, clock: Clock, blackboard: Blackboard, chat: ChatClient) -> None:
        self._clock = clock
        self._blackboard = blackboard
        self._chat = chat
        self._pending: deque[FaultEvent] = deque(maxlen=_QUEUE_LENGTH)
        self._explained: deque[str] = deque(maxlen=_REMEMBERED)
        self._mode = Mode.INIT

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.FAULT, FaultEvent, self._on_fault)
        self._blackboard.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)

    def _on_fault(self, _topic: str, event: FaultEvent) -> None:
        if event.fault_id in self._explained:
            return
        if any(pending.fault_id == event.fault_id for pending in self._pending):
            return
        self._pending.append(event)

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        self._mode = state.mode

    def process_pending(self) -> list[FaultDiagnosis]:
        diagnoses = []
        while self._pending:
            diagnoses.append(self.diagnose(self._pending.popleft()))
        return diagnoses

    def diagnose(self, event: FaultEvent) -> FaultDiagnosis:
        """Explain one fault and publish the explanation."""
        self._explained.append(event.fault_id)
        features = _feature_vector(event, self._mode)
        raw = ""
        latency_s = 0.0
        tokens = (0, 0)
        verdict = "accepted"
        try:
            turn = self._chat.complete(
                [
                    {"role": "system", "content": _prompt()},
                    {"role": "user", "content": features},
                ],
                json_only=True,
            )
            raw, latency_s = turn.content, turn.latency_s
            tokens = (turn.prompt_tokens, turn.completion_tokens)
            diagnosis = self._validate(raw, event)
        except ReasoningUnavailableError as exc:
            LOGGER.warning("diagnosis unavailable for %s: %s", event.fault_id, exc)
            diagnosis, verdict = None, "reasoning unavailable"
        if diagnosis is None:
            verdict = verdict if verdict != "accepted" else "discarded: failed validation"
            diagnosis = self._generic(event)
        elif not diagnosis.recommendation_legal:
            verdict = "accepted; recommended mode illegal, detector-derived mode stands"

        self._blackboard.publish(topics.DIAGNOSIS, diagnosis, fault_id=event.fault_id)
        self._blackboard.publish(
            topics.AUDIT_REASONING,
            ReasoningRecord(
                ts=self._clock.now(),
                caller=ReasoningCaller.DIAGNOSIS,
                trigger="fault confirmation",
                model=self._chat.model,
                inputs=features,
                raw_output=raw,
                verdict=verdict,
                applied=diagnosis.user_message,
                latency_s=latency_s,
                prompt_tokens=tokens[0],
                completion_tokens=tokens[1],
            ),
        )
        LOGGER.info("diagnosis of %s: %s", event.fault_id, diagnosis.user_message)
        return diagnosis

    def _validate(self, raw: str, event: FaultEvent) -> FaultDiagnosis | None:
        """Post-decode semantic validation (FR-44)."""
        try:
            decoded = json.loads(raw)
            if not isinstance(decoded, dict):
                return None
            recommended = Mode(decoded["recommended_mode"])
            evidence = decoded.get("supporting_evidence") or []
            if not isinstance(evidence, list):
                return None
            return FaultDiagnosis(
                ts=self._clock.now(),
                fault_id=event.fault_id,
                primary_hypothesis=Hypothesis(decoded["primary_hypothesis"]),
                confidence=DiagnosisConfidence(decoded["confidence"]),
                supporting_evidence=tuple(str(item) for item in evidence)[:8],
                recommended_mode=recommended,
                user_message=str(decoded["user_message"]).strip(),
                recommendation_legal=recommended in LEGAL_TRANSITIONS[self._mode],
            )
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
            LOGGER.warning("discarding diagnosis of %s: %s", event.fault_id, exc)
            return None

    def _generic(self, event: FaultEvent) -> FaultDiagnosis:
        hypothesis, message = _GENERIC.get(
            event.detector, (Hypothesis.UNKNOWN, "A fault was detected; the system is handling it.")
        )
        return FaultDiagnosis(
            ts=self._clock.now(),
            fault_id=event.fault_id,
            primary_hypothesis=hypothesis,
            confidence=DiagnosisConfidence.LOW,
            recommended_mode=event.mode_impact,
            user_message=message,
            recommendation_legal=event.mode_impact in LEGAL_TRANSITIONS[self._mode],
            generic=True,
        )


def _prompt() -> str:
    return DIAGNOSIS_PROMPT.format(
        hypotheses=", ".join(item.value for item in Hypothesis),
        modes=", ".join(mode.value for mode in Mode),
    )


def _feature_vector(event: FaultEvent, mode: Mode) -> str:
    """What the model is given: the fault as the detector reported it."""
    return json.dumps(
        {
            "detector": event.detector.value,
            "subject": event.subject,
            "fault_class": event.fault_class.value,
            "detector_confidence": event.confidence,
            "evidence": dict(event.evidence),
            "mode_now": mode.value,
            "mode_the_fault_caused": event.mode_impact.value,
        },
        sort_keys=True,
    )
