"""E3's harness finds a fault through the same channel tools.inject uses."""

import tempfile
from pathlib import Path

import pytest

from eval.experiments.e3_detection import FAULT_CLASSES, run_trial
from src.common.config import load_config


@pytest.mark.parametrize(
    "fault", [fault for fault in FAULT_CLASSES if fault.name in ("dropout", "out-of-range")],
    ids=lambda fault: fault.name,
)
def test_a_fast_fault_is_found_by_its_own_detector(fault):
    config = load_config(Path("config/default.yaml"))
    with tempfile.TemporaryDirectory() as scratch:
        trial = run_trial(config, fault, seed=1, directory=Path(scratch))
    assert trial.detected
    assert trial.latency_s <= fault.horizon_s


def test_every_detector_has_a_fault_class():
    detectors = {fault.expected.value for fault in FAULT_CLASSES}
    assert detectors == {
        "D1_DROPOUT", "D2_STUCK_AT", "D3_OUT_OF_RANGE", "D4_DRIFT", "D5_ACTUATOR_NO_RESPONSE",
    }


@pytest.mark.parametrize(
    "fault", [fault for fault in FAULT_CLASSES if fault.name in ("drift 0.6 C/min", "actuator")],
    ids=lambda fault: fault.name,
)
def test_drift_and_a_dead_unit_are_told_apart(fault):
    """S-02, as found live: a drifting thermometer beside a working unit was
    blamed on the unit, which was put on hold. The meter now tells them apart."""
    config = load_config(Path("config/default.yaml"))
    with tempfile.TemporaryDirectory() as scratch:
        trial = run_trial(config, fault, seed=1, directory=Path(scratch))
    assert trial.detected and trial.first_detector == fault.expected.value
