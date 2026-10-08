"""Unit tests for re-deriving detector thresholds from a recording (R-04).

A recording made the way ``tools.record`` makes one -- the real recorder on a
whole simulated system -- is calibrated from, and the overlay it writes must
load. The edge cases use recordings built by hand, where the measured
quantity is known exactly.
"""

import io
import json
from pathlib import Path

import pytest

from eval.system import System
from sim.replay import Recorded, load
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.schemas import FaultEvent, Mode, SensorReading, Unit
from tools.calibrate import CalibrationError, derive, main, render
from tools.record import Recorder

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
PERIOD_S = 5.0


@pytest.fixture(name="config")
def _config():
    """The default, accepting a short recording so the tests stay fast."""
    config = load_config(DEFAULT_CONFIG_PATH)
    calibration = config.calibration.model_copy(update={"min_hours": 0.1})
    return config.model_copy(update={"calibration": calibration})


@pytest.fixture(name="recording", scope="module")
def _recording() -> list[Recorded]:
    """Half an hour of a healthy simulated room, through the real recorder."""
    config = load_config(DEFAULT_CONFIG_PATH)
    clock = SimClock()
    sink = io.StringIO()
    system = System(config, clock)
    recorder = Recorder(clock, sink)
    recorder.attach(system.transport)
    system.transport.attach(recorder)
    system.run_for(1800.0)
    return [
        Recorded(float(entry["t"]), entry["topic"], entry["payload"])
        for entry in map(json.loads, sink.getvalue().splitlines())
    ]


def _setting(calibration, path):
    return next(setting for setting in calibration.settings if setting.path == path)


def _readings(sensor_id: str, arrivals: list[float], value: float = 25.0) -> list[Recorded]:
    unit = Unit.CELSIUS
    return [
        Recorded(
            t,
            topics.SENSOR_STATE.format(sensor_id=sensor_id),
            SensorReading(ts=1756032000.0 + t, sensor_id=sensor_id, value=value + 0.1 * (index % 3), unit=unit)
            .model_dump_json(),
        )
        for index, t in enumerate(arrivals)
    ]


class TestFromARealRecording:
    def test_every_noise_derived_threshold_is_derived(self, recording, config):
        paths = {setting.path for setting in derive(recording, config).settings}
        assert paths == {
            "detectors.dropout.timeout_periods",
            "detectors.stuck_at.variance_epsilon",
            "detectors.drift.threshold_sigma",
        }

    def test_each_setting_states_what_it_was_measured_from(self, recording, config):
        for setting in derive(recording, config).settings:
            assert "healthy" in setting.basis

    def test_the_overlay_loads_and_extends_the_base(self, recording, config, tmp_path):
        out = tmp_path / "site.yaml"
        out.write_text(render(derive(recording, config), DEFAULT_CONFIG_PATH, out), encoding="utf-8")
        calibrated = load_config(out)
        assert calibrated.controller == config.controller
        derived = _setting(derive(recording, config), "detectors.stuck_at.variance_epsilon")
        assert calibrated.detectors.stuck_at.variance_epsilon == derived.derived

    def test_d2_epsilon_sits_below_the_quietest_healthy_window(self, recording, config):
        calibration = derive(recording, config)
        epsilon = _setting(calibration, "detectors.stuck_at.variance_epsilon").derived
        quietest = min(
            series.quietest_variance(config.detectors.stuck_at.window_samples)
            for series in calibration.sensors
            if series.unit in (Unit.CELSIUS, Unit.PERCENT_RH)
            and series.quietest_variance(config.detectors.stuck_at.window_samples) is not None
        )
        assert 0.0 < epsilon < quietest

    def test_the_command_line_writes_a_loadable_overlay(self, recording, tmp_path):
        source = tmp_path / "healthy.jsonl"
        source.write_text(
            "\n".join(json.dumps({"t": e.t, "topic": e.topic, "payload": e.payload}) for e in recording),
            encoding="utf-8",
        )
        base = tmp_path / "base.yaml"
        base.write_text(
            DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
                "min_hours: 1.0", "min_hours: 0.1"
            ),
            encoding="utf-8",
        )
        out = tmp_path / "site.yaml"
        assert main([str(source), "--config", str(base), "--out", str(out)]) == 0
        assert load_config(out).calibration.min_hours == 0.1
        assert len(load(source)) == len(recording)


class TestDerivation:
    def test_the_timeout_clears_the_longest_healthy_gap(self, config):
        arrivals = [index * PERIOD_S for index in range(800)]
        del arrivals[400:403]  # one gap of four periods
        calibration = derive(_readings("temp_01", arrivals), config)
        timeout = _setting(calibration, "detectors.dropout.timeout_periods").derived
        assert timeout >= 4.0 * config.calibration.timeout_margin

    def test_a_link_dropping_past_nfr_02_is_reported(self, config):
        arrivals = [index * PERIOD_S for index in range(800)]
        del arrivals[400:410]  # eleven periods of silence
        calibration = derive(_readings("temp_01", arrivals), config)
        assert any("NFR-02" in warning for warning in calibration.warnings)

    def test_a_healthy_link_raises_no_warning(self, config):
        arrivals = [index * PERIOD_S for index in range(800)]
        assert derive(_readings("temp_01", arrivals), config).warnings == ()


class TestRefusals:
    def test_an_empty_recording_is_refused(self, config):
        with pytest.raises(CalibrationError):
            derive([], config)

    def test_a_short_recording_is_refused(self, config):
        with pytest.raises(CalibrationError):
            derive(_readings("temp_01", [0.0, 5.0, 10.0]), config)

    def test_a_recording_with_a_fault_is_refused(self, config):
        """Calibrating from it would set the thresholds to accept the fault."""
        fault = FaultEvent.model_validate(
            {
                "fault_id": "f_temp01_stuck_1",
                "detector": "D2_STUCK_AT",
                "subject": "temp_01",
                "class": "sensor",
                "confidence": 1.0,
                "detected_ts": 1756032300.0,
                "evidence": {},
                "mode_impact": Mode.DEGRADED_SENSOR,
            }
        )
        recording = _readings("temp_01", [index * PERIOD_S for index in range(800)])
        recording.insert(
            100,
            Recorded(500.0, topics.FAULT.format(fault_id=fault.fault_id), fault.model_dump_json(by_alias=True)),
        )
        with pytest.raises(CalibrationError, match="fault"):
            derive(recording, config)

    def test_a_withdrawn_fault_is_not_mistaken_for_one(self, config):
        """A retained fault is withdrawn with an empty payload."""
        recording = _readings("temp_01", [index * PERIOD_S for index in range(800)])
        recording.insert(100, Recorded(500.0, topics.FAULT.format(fault_id="f_x"), ""))
        assert derive(recording, config).settings
