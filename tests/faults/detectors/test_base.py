"""Unit tests for the detector contract.

What matters here is that the three-valued judgment survives: UNKNOWN must
not be usable as CLEAR, and a finding that asserts nothing must not be
publishable as a fault.
"""

import pytest

from src.common.schemas import DetectorId, FaultClass, Mode
from src.faults.detectors.base import (
    Finding,
    Judgment,
    build_fault_event,
    fault_id,
)

SUBJECT = "temp_01"
TS = 1756032300.0


def _finding(**overrides) -> Finding:
    return Finding(
        **{
            "detector": DetectorId.D2_STUCK_AT,
            "subject": SUBJECT,
            "judgment": Judgment.FAULTED,
            "confidence": 0.94,
            "evidence": {"variance": 0.0002},
            **overrides,
        }
    )


class TestFinding:
    def test_a_faulted_finding_asserts_a_fault(self):
        assert _finding().faulted

    def test_a_clear_finding_does_not(self):
        assert not _finding(judgment=Judgment.CLEAR).faulted

    def test_an_unknown_finding_does_not_assert_health_either(self):
        """The absence of evidence is not evidence of absence: a detector that
        has heard nothing has not observed a working sensor."""
        unknown = _finding(judgment=Judgment.UNKNOWN)
        assert not unknown.faulted
        assert unknown.judgment is not Judgment.CLEAR

    def test_evidence_travels_with_a_clear_finding_too(self):
        """The numbers that did not trip a threshold are what make a
        demonstration legible."""
        clear = _finding(judgment=Judgment.CLEAR, evidence={"variance": 0.04})
        assert clear.evidence["variance"] == 0.04

    def test_evidence_cannot_be_edited_after_the_fact(self):
        finding = _finding()
        with pytest.raises(TypeError):
            finding.evidence["variance"] = 9.9

    def test_mutating_the_source_dictionary_does_not_reach_the_finding(self):
        evidence = {"variance": 0.0002}
        finding = _finding(evidence=evidence)
        evidence["variance"] = 9.9
        assert finding.evidence["variance"] == 0.0002

    def test_a_finding_is_frozen(self):
        with pytest.raises(Exception):
            _finding().judgment = Judgment.CLEAR

    def test_a_finding_must_name_its_subject(self):
        with pytest.raises(ValueError):
            _finding(subject="")

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_confidence_outside_the_unit_interval_is_refused(self, confidence):
        with pytest.raises(ValueError):
            _finding(confidence=confidence)

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_the_ends_of_the_unit_interval_are_allowed(self, confidence):
        assert _finding(confidence=confidence).confidence == confidence


class TestDetectorImpact:
    """Section 5.6: which mode a fault alone implies."""

    @pytest.mark.parametrize(
        "detector",
        [
            DetectorId.D1_DROPOUT,
            DetectorId.D2_STUCK_AT,
            DetectorId.D3_OUT_OF_RANGE,
            DetectorId.D4_DRIFT,
        ],
    )
    def test_a_sensor_detector_degrades_to_running_on_prediction(self, detector):
        finding = _finding(detector=detector)
        assert finding.fault_class is FaultClass.SENSOR
        assert finding.mode_impact is Mode.DEGRADED_SENSOR

    def test_the_actuator_detector_stops_actuation(self):
        finding = _finding(detector=DetectorId.D5_ACTUATOR_NO_RESPONSE)
        assert finding.fault_class is FaultClass.ACTUATOR
        assert finding.mode_impact is Mode.DEGRADED_ACTUATOR

    def test_a_diverged_model_holds_rather_than_degrades(self):
        """There is no prediction left to degrade onto."""
        finding = _finding(detector=DetectorId.MODEL_DIVERGENCE)
        assert finding.fault_class is FaultClass.MODEL
        assert finding.mode_impact is Mode.SAFE_HOLD

    def test_every_detector_has_an_impact(self):
        """A detector with no mapping would raise at publication time, which
        is the worst moment to discover it."""
        for detector in DetectorId:
            assert _finding(detector=detector).mode_impact in Mode


class TestFaultId:
    def test_the_id_names_subject_detector_and_time(self):
        assert fault_id(DetectorId.D2_STUCK_AT, SUBJECT, TS) == (
            "f_temp01_stuck_1756032300"
        )

    def test_the_id_is_deterministic_so_a_replay_reproduces_it(self):
        first = fault_id(DetectorId.D1_DROPOUT, SUBJECT, TS)
        assert first == fault_id(DetectorId.D1_DROPOUT, SUBJECT, TS)

    def test_different_detectors_on_one_subject_get_different_ids(self):
        assert fault_id(DetectorId.D1_DROPOUT, SUBJECT, TS) != fault_id(
            DetectorId.D2_STUCK_AT, SUBJECT, TS
        )

    def test_the_id_is_one_topic_level(self):
        """It is substituted into space/fault/{fault_id}, so a separator in it
        would silently reshape the topic tree."""
        for character in ("/", "+", "#"):
            assert character not in fault_id(DetectorId.D3_OUT_OF_RANGE, SUBJECT, TS)


class TestBuildFaultEvent:
    def test_the_event_carries_the_findings_evidence(self):
        event = build_fault_event(_finding(), TS)
        assert event.evidence["variance"] == 0.0002

    def test_the_event_carries_the_detectors_class_and_mode(self):
        event = build_fault_event(_finding(), TS)
        assert event.fault_class is FaultClass.SENSOR
        assert event.mode_impact is Mode.DEGRADED_SENSOR

    def test_the_event_is_timed_by_detection_not_by_the_finding(self):
        event = build_fault_event(_finding(), TS)
        assert event.detected_ts == TS

    def test_the_event_id_matches_the_generator(self):
        event = build_fault_event(_finding(), TS)
        assert event.fault_id == fault_id(DetectorId.D2_STUCK_AT, SUBJECT, TS)

    @pytest.mark.parametrize("judgment", [Judgment.CLEAR, Judgment.UNKNOWN])
    def test_a_finding_asserting_no_fault_cannot_be_published_as_one(self, judgment):
        """Every consumer downstream treats a FaultEvent as a fault."""
        with pytest.raises(ValueError):
            build_fault_event(_finding(judgment=judgment), TS)

    def test_the_documented_payload_shape_is_reproduced(self):
        """Section 6.2's example: same fields, same id shape."""
        event = build_fault_event(_finding(), TS)
        payload = event.model_dump(by_alias=True)
        assert set(payload) == {
            "fault_id",
            "detector",
            "subject",
            "class",
            "confidence",
            "detected_ts",
            "evidence",
            "mode_impact",
        }
