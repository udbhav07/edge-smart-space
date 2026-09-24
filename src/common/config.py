"""Typed configuration for every tunable in the system.

No numeric policy value is written in code. Detector thresholds in
particular are re-derived from measured noise during hardware bring-up
(R-04), and that has to be a config edit rather than a code hunt.

Every field carries its unit in its name and a description of what it
governs. Values are validated on load, so a malformed config fails at
startup rather than at the first regulatory tick.

Defaults live in ``config/default.yaml`` and are the values in DESIGN.md
sections 5.2.2, 5.3, 5.4 and 5.5.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.schemas import Unit

#: Number of coefficients in the RC model (DESIGN.md section 5.2.1).
COEFFICIENT_COUNT = 4


class ConfigError(ValueError):
    """Configuration is missing, malformed, or internally inconsistent."""


class _Section(BaseModel):
    """Base for every config section: immutable and closed to typos."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Bounds(_Section):
    """An inclusive interval."""

    low: float
    high: float

    @model_validator(mode="after")
    def _low_below_high(self) -> Bounds:
        if self.low >= self.high:
            raise ValueError(f"low {self.low!r} must be below high {self.high!r}")
        return self

    def clamp(self, value: float) -> float:
        """Nearest value inside the interval."""
        return min(max(value, self.low), self.high)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


class MqttConfig(_Section):
    host: str = Field(min_length=1, description="Broker hostname")
    port: int = Field(gt=0, lt=65536, description="Broker TCP port")
    keepalive_s: float = Field(gt=0.0, description="MQTT keepalive interval")
    reconnect_min_s: float = Field(gt=0.0, description="First reconnect backoff")
    reconnect_max_s: float = Field(gt=0.0, description="Reconnect backoff ceiling")

    @model_validator(mode="after")
    def _backoff_range_is_ordered(self) -> MqttConfig:
        if self.reconnect_min_s > self.reconnect_max_s:
            raise ValueError("reconnect_min_s must not exceed reconnect_max_s")
        return self


class LoopConfig(_Section):
    regulatory_period_s: float = Field(
        gt=0.0, description="Regulatory tick period; NFR-01 requires 5 s"
    )
    sensor_period_s: float = Field(
        gt=0.0, description="Nominal indoor sampling interval; FR-01, A-02"
    )
    outdoor_period_s: float = Field(
        gt=0.0, description="Ambient sampling interval; FR-03, A-04"
    )


class EstimatorConfig(_Section):
    """RLS parameters from DESIGN.md section 5.2.2 and its safeguards table."""

    forgetting_factor: float = Field(
        gt=0.0, le=1.0, description="Lambda; effective memory is 1/(1-lambda) samples"
    )
    initial_covariance: float = Field(
        gt=0.0, description="P0 diagonal; large means a weak prior"
    )
    initial_theta: tuple[float, float, float] = Field(
        description=(
            "Coarse physical guess for the identified vector [a2, a3, a4], so "
            "early control is not wild. a1 follows as 1 - a2 (section 5.2.2)"
        )
    )
    max_covariance_trace: float = Field(
        gt=0.0, description="Trace bound against windup during low excitation"
    )
    min_excitation: float = Field(
        ge=0.0, description="Regressor variation below this skips the update"
    )
    divergence_window_samples: int = Field(
        gt=1,
        description=(
            "Updates the divergence judgement is made over. Must be long "
            "enough that ordinary noise cannot fill it (FR-06)"
        ),
    )
    divergence_rejection_fraction: float = Field(
        gt=0.0,
        le=1.0,
        description=(
            "Share of a full window that must be rejected before "
            "MODEL_DIVERGENCE is raised"
        ),
    )
    fault_history_samples: int = Field(
        default=130,
        gt=1,
        description="Samples of regressor history kept, so a free run can be "
        "seeded from before a late-detected fault began",
    )
    indoor_sensor_id: str = Field(
        min_length=1, description="Feeds T[k], the regressor's first entry"
    )
    outdoor_sensor_id: str = Field(min_length=1, description="Feeds T_out[k]")
    occupancy_sensor_id: str = Field(min_length=1, description="Feeds o[k]")
    excitation_window_samples: int = Field(
        gt=1, description="Samples over which regressor variation is judged"
    )
    residual_sigma_window_samples: int = Field(
        gt=1, description="Samples backing the published residual_sigma"
    )
    sample_interval_tolerance: float = Field(
        gt=0.0,
        le=1.0,
        description=(
            "Allowed deviation from the nominal period, as a fraction. A-02's "
            "uniform sampling will not survive WiFi, and the ARX form assumes "
            "a fixed step, so a pair spanning a wrong interval is skipped"
        ),
    )
    bounds_a1: Bounds
    bounds_a2: Bounds
    bounds_a3: Bounds
    bounds_a4: Bounds

    @property
    def coefficient_bounds(self) -> tuple[Bounds, Bounds, Bounds, Bounds]:
        """Plausibility box, in coefficient order."""
        return (self.bounds_a1, self.bounds_a2, self.bounds_a3, self.bounds_a4)

    @model_validator(mode="after")
    def _initial_theta_lies_inside_the_plausibility_box(self) -> EstimatorConfig:
        """Check the four coefficients the prior implies, a1 included.

        a1 is derived rather than configured, so a prior that looks fine as
        [a2, a3, a4] can still imply an a1 outside its range.
        """
        derived = (1.0 - self.initial_theta[0],) + self.initial_theta
        for name, value, bounds in zip(
            ("a1", "a2", "a3", "a4"), derived, self.coefficient_bounds
        ):
            if not bounds.contains(value):
                raise ValueError(
                    f"initial_theta implies {name} = {value!r}, outside its "
                    f"bounds [{bounds.low}, {bounds.high}]"
                )
        return self


class PersistenceConfig(_Section):
    """Coefficient persistence across restarts (FR-07, section 7.1)."""

    path: str = Field(min_length=1, description="Where theta and P are written")
    interval_s: float = Field(gt=0.0, description="How often state is written")
    max_age_s: float = Field(
        gt=0.0,
        description="Older than this on restart and the estimate is discarded",
    )


class ControllerConfig(_Section):
    """Deadband law parameters from DESIGN.md section 5.3."""

    deadband_c: float = Field(gt=0.0, description="Symmetric; prevents chatter")
    min_off_s: float = Field(
        ge=0.0, description="Compressor protection; also enforced by validator V-3"
    )
    default_setpoint_c: float = Field(
        description="Held before any goal arrives and when every goal is stale"
    )
    reassert_interval_s: float = Field(
        default=60.0,
        gt=0.0,
        description="How often the intended actuator state is re-sent, so a "
        "lost command on an open-loop path is not permanent (R-02)",
    )
    excitation_duration_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Identification phase after startup during which the "
        "compressor is driven on a schedule rather than by the error (R-01). "
        "Zero disables it.",
    )
    excitation_period_s: float = Field(
        default=900.0,
        gt=0.0,
        description="Full excitation cycle, half cooling and half off",
    )
    excitation_envelope_c: float = Field(
        default=2.0,
        gt=0.0,
        description="Excitation is abandoned if the room is further than this "
        "from setpoint, so identification never costs more comfort than stated",
    )

    @model_validator(mode="after")
    def _excitation_stays_outside_the_deadband(self) -> ControllerConfig:
        """An envelope inside the deadband would abandon excitation instantly.

        The deadband is the band in which the controller does nothing, so an
        envelope narrower than it means the room is outside the envelope
        exactly when the controller would have acted anyway -- and the
        excitation never runs, silently.
        """
        if self.excitation_duration_s > 0.0 and (
            self.excitation_envelope_c <= self.deadband_c
        ):
            raise ValueError(
                f"excitation_envelope_c ({self.excitation_envelope_c}) must "
                f"exceed deadband_c ({self.deadband_c}), or excitation never runs"
            )
        return self


class ValidatorConfig(_Section):
    """Hard constraints, rules V-1 to V-6 in DESIGN.md section 5.4."""

    setpoint_bounds_c: Bounds = Field(description="V-1 absolute bounds")
    max_step_c: float = Field(gt=0.0, description="V-2 rate limit per invocation")
    min_off_s: float = Field(ge=0.0, description="V-3 compressor dwell")
    min_command_interval_s: float = Field(
        gt=0.0, description="V-4 maximum command frequency"
    )
    goal_max_age_s: float = Field(gt=0.0, description="V-6 staleness horizon")


class EvaluationConfig(_Section):
    """Policy for the experiments (section 8.3), not for the running system.

    It lives in config rather than in ``eval`` because a success criterion
    somebody can change by editing a constant is not a criterion.
    """

    comfort_band_c: float = Field(
        default=1.0,
        gt=0.0,
        description="Distance from setpoint still counted as comfortable, in C",
    )


class DropoutDetectorConfig(_Section):
    """D1: no message within a timeout (FR-20)."""

    timeout_periods: float = Field(
        gt=0.0, description="Timeout as a multiple of the sensor period"
    )


class StuckAtDetectorConfig(_Section):
    """D2: variance collapse over a sliding window (FR-21)."""

    window_samples: int = Field(gt=1, description="Samples per variance window")
    variance_epsilon: float = Field(
        gt=0.0, description="Variance below this counts as stuck, in unit squared"
    )
    consecutive_windows: int = Field(
        gt=0, description="Windows in a row before the fault is raised"
    )


class OutOfRangeDetectorConfig(_Section):
    """D3: reading outside physical limits (FR-22).

    These are the limits, not the schema's. A reading outside them must be
    representable in order to arrive here at all.
    """

    temperature_c: Bounds
    humidity_pct: Bounds
    debounce_samples: int = Field(gt=0, description="Consecutive samples before raising")


class DriftDetectorConfig(_Section):
    """D4: two-sided CUSUM on the normalised model residual (FR-23).

    Both thresholds are in units of the residual's own standard deviation, so
    they carry over unchanged to a sensor with different noise. R-04 re-derives
    them from measured sigma during hardware bring-up.
    """

    slack_sigma: float = Field(
        gt=0.0, description="k: drift below this is absorbed rather than accumulated"
    )
    threshold_sigma: float = Field(
        gt=0.0, description="h: cumulative sum at which drift is declared"
    )
    min_residual_sigma_c: float = Field(
        gt=0.0,
        description="Noise floor below which the residual is not normalisable yet",
    )
    max_sample_sigma: float = Field(
        gt=0.0,
        description="Largest contribution one sample may make, in sigma",
    )
    warmup_samples: int = Field(
        ge=0,
        description="Estimates ignored before the test starts, while the "
        "model is still converging",
    )

    @model_validator(mode="after")
    def _one_sample_cannot_carry_the_test(self) -> DriftDetectorConfig:
        """A cap at or above the threshold is no cap at all.

        The point of a cumulative test is that persistence trips it, not
        magnitude. If a single sample can contribute the whole threshold, D4
        degenerates into the noisy threshold alarm it exists to replace --
        and a sensor being repaired produces exactly such a sample, because
        the reading steps back to the truth while the frozen model is still
        predicting from where it was.
        """
        if self.max_sample_sigma - self.slack_sigma >= self.threshold_sigma:
            raise ValueError(
                f"max_sample_sigma {self.max_sample_sigma!r} less slack "
                f"{self.slack_sigma!r} must stay below threshold_sigma "
                f"{self.threshold_sigma!r}, or one sample decides the test"
            )
        return self

    @model_validator(mode="after")
    def _slack_is_below_the_threshold(self) -> DriftDetectorConfig:
        """Slack at or above the threshold can never accumulate to it.

        The CUSUM adds ``z - k`` per sample, so a k no smaller than h makes a
        single sample the whole test and turns the detector into a noisy
        threshold alarm. Silently, which is the problem.
        """
        if self.slack_sigma >= self.threshold_sigma:
            raise ValueError(
                f"slack_sigma {self.slack_sigma!r} must be below threshold_sigma "
                f"{self.threshold_sigma!r}, or drift can never accumulate"
            )
        return self


class ActuatorDetectorConfig(_Section):
    """D5: no thermal response to sustained cooling (FR-24)."""

    evaluation_window_s: float = Field(
        gt=0.0, description="Sustained cooling required before the test applies"
    )
    warmup_samples: int = Field(
        ge=0,
        description="Estimates ignored before the test starts, while the "
        "model is still converging",
    )
    min_expected_cooling_c: float = Field(
        gt=0.0,
        description="Expected cooling below which no verdict is given, in C",
    )
    response_fraction: float = Field(
        gt=0.0,
        lt=1.0,
        description="Share of the model's expected cooling the room must deliver",
    )


class DetectorsConfig(_Section):
    dropout: DropoutDetectorConfig
    stuck_at: StuckAtDetectorConfig
    out_of_range: OutOfRangeDetectorConfig
    drift: DriftDetectorConfig
    actuator: ActuatorDetectorConfig


class ModeConfig(_Section):
    """Degradation state machine timings from DESIGN.md sections 5.6 and 7.2."""

    degraded_sensor_budget_s: float = Field(
        gt=0.0,
        description="Prediction-based control budget; E4 measures whether it is right",
    )
    fault_clear_confirm_s: float = Field(
        gt=0.0, description="Clear must hold this long before returning to NORMAL"
    )
    transition_deadline_s: float = Field(
        gt=0.0, description="FR-26 budget from confirmation to published mode"
    )


class SensorConfig(_Section):
    """One sensor's identity and physical limits (DESIGN.md section 5.1).

    ``limits`` are what the instrument can physically report, and the adapter
    uses them to flag a reading as suspect. It never suppresses one: D3's job
    is to detect an out-of-range reading (FR-22), and a reading filtered at
    Layer 1 could never reach the detector that exists to find it.
    """

    sensor_id: str = Field(min_length=1)
    unit: Unit
    limits: Bounds
    description: str = Field(default="", description="What this instrument is")


class SensorsConfig(_Section):
    """Every sensor the system expects to hear from."""

    adapters: tuple[SensorConfig, ...] = Field(min_length=1)
    vacancy_hold_off_s: float = Field(
        default=600.0,
        ge=0.0,
        description="FR-02: how long the room stays occupied after the last "
        "motion or door transition, in seconds",
    )

    @model_validator(mode="after")
    def _sensor_ids_are_unique(self) -> SensorsConfig:
        seen = [adapter.sensor_id for adapter in self.adapters]
        duplicates = {name for name in seen if seen.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate sensor ids: {sorted(duplicates)}")
        return self

    def by_id(self, sensor_id: str) -> SensorConfig:
        """Look one up.

        :raises KeyError: if no adapter is configured under that id.
        """
        for adapter in self.adapters:
            if adapter.sensor_id == sensor_id:
                return adapter
        raise KeyError(f"no sensor configured with id {sensor_id!r}")


class Layer1Source(str, Enum):
    """Which Layer 1 implementation runs (section 9.1).

    The phase transition is a configuration change rather than a code change:
    both publish the same topics, so nothing above Layer 1 can tell which is
    running and the whole test suite applies to either.
    """

    SIMULATED = "simulated"
    ESPHOME = "esphome"


class DeviceBinding(_Section):
    """One configured sensor and the device topic that feeds it.

    The topic belongs in configuration because it is decided when a node is
    flashed and named, not when this code is written. Guessing a naming
    convention here would produce code that looks finished and silently
    matches nothing (R-03).
    """

    sensor_id: str = Field(min_length=1)
    topic: str = Field(min_length=1, description="Device topic the node publishes on")


class IoActuatorConfig(_Section):
    """Where actuator commands are sent on hardware."""

    command_topic: str = Field(min_length=1)
    acknowledges: bool = Field(
        default=False,
        description="R-02: an IR path has no readback, so this is normally "
        "false and every ack is UNKNOWN",
    )


class IoConfig(_Section):
    """Layer 1 selection and the device details that belong to it."""

    source: Layer1Source = Layer1Source.SIMULATED
    stale_after_s: float = Field(
        gt=0.0,
        description="Age at which a held device value counts as silence",
    )
    devices: tuple[DeviceBinding, ...] = ()
    actuator: IoActuatorConfig

    @model_validator(mode="after")
    def _device_ids_are_unique(self) -> IoConfig:
        seen = [binding.sensor_id for binding in self.devices]
        duplicates = {name for name in seen if seen.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate device bindings: {sorted(duplicates)}")
        return self

    def topic_for(self, sensor_id: str) -> str | None:
        """The device topic feeding one sensor, if it has one."""
        for binding in self.devices:
            if binding.sensor_id == sensor_id:
                return binding.topic
        return None


class SpeechConfig(_Section):
    """Speech pipeline settings (DESIGN.md section 5.8).

    Capture opens only after wake-word detection and closes at end of
    utterance or a hard timeout (FR-50), so both timeouts are policy and
    live here rather than in code.
    """

    sample_rate_hz: int = Field(gt=0, description="Capture rate; Whisper expects 16 kHz")
    channels: int = Field(gt=0, description="Capture channels; mono for ASR")
    chunk_samples: int = Field(gt=0, description="Frame size; Silero VAD requires 512")
    device_index: int | None = Field(
        default=None, description="Input device; None selects the system default"
    )
    queue_seconds: float = Field(
        gt=0.0, description="Audio buffered before the oldest frame is dropped"
    )
    wake_word: str = Field(min_length=1, description="Model name to listen for")
    wake_word_threshold: float = Field(
        gt=0.0,
        le=1.0,
        description="Detection score gate; low values make capture near-continuous",
    )
    vad_silence_ms: int = Field(gt=0, description="Silence marking end of utterance")
    no_speech_timeout_s: float = Field(
        gt=0.0, description="Give up if nothing is said after the wake word"
    )
    command_timeout_s: float = Field(
        gt=0.0, description="Hard cap on one utterance (FR-50)"
    )
    asr_model: str = Field(min_length=1, description="Whisper model size")
    asr_device: str = Field(
        min_length=1,
        description="auto (prefer CUDA, fall back to CPU), cuda, or cpu",
    )
    asr_compute_type: str = Field(
        min_length=1, description="auto (float16 on GPU, int8 on CPU), or explicit"
    )

    @model_validator(mode="after")
    def _an_utterance_may_run_longer_than_the_silence_timeout(self) -> SpeechConfig:
        if self.command_timeout_s <= self.no_speech_timeout_s:
            raise ValueError(
                "command_timeout_s must exceed no_speech_timeout_s, or a "
                "spoken request is cut off before it can finish"
            )
        return self


class ReasoningConfig(_Section):
    """Inference endpoint shared by every call site (DESIGN.md section 5.7.5).

    One server serves all call sites, differentiated by system prompt and
    constraint mechanism. The endpoint is OpenAI-compatible, which both
    llama.cpp's server and Ollama expose.
    """

    base_url: str = Field(min_length=1, description="Local endpoint; no traffic leaves")
    model: str = Field(min_length=1)
    timeout_s: float = Field(gt=0.0, description="Abandon a stalled call")
    max_history_turns: int = Field(
        gt=0, description="Retained turns; unbounded history grows every prompt"
    )


class AssistanceConfig(_Section):
    """Assistance tool invocation (DESIGN.md section 5.7.6).

    The tool surface is declared in ``src/common/tools.py`` rather than here:
    a parameter list is an interface, not a tunable, and a tool that could be
    redefined in a YAML file could not be validated against.
    """

    confirmation_window_s: float = Field(
        gt=0.0,
        description="Seconds a COMMIT invocation stays confirmable (FR-74)",
    )
    max_tool_rounds: int = Field(
        gt=0,
        description="Tool-call rounds one reasoning call may make before answering",
    )
    result_timeout_s: float = Field(
        gt=0.0, description="Seconds to wait for a tool result before answering without"
    )


class SensorNoiseConfig(_Section):
    """Adversarial-by-default sensor imperfection.

    A kind simulator makes simulated success predict nothing about hardware,
    so every field here defaults to a value that hurts. A-02's uniform
    sampling will not survive WiFi.
    """

    sigma_c: float = Field(ge=0.0, description="Gaussian noise standard deviation")
    quantisation_c: float = Field(
        ge=0.0, description="Reading resolution; 0 disables quantisation"
    )
    jitter_s: float = Field(ge=0.0, description="Uniform sampling jitter, plus or minus")
    dropout_probability: float = Field(
        ge=0.0, le=1.0, description="Chance a sample never arrives"
    )
    bias_c: float = Field(description="Constant offset; a stand-in for calibration error")


class SimActuatorConfig(_Section):
    """Actuator imperfection. R-02: an IR path cannot confirm anything."""

    dead_time_s: float = Field(
        ge=0.0, description="Delay between command and any thermal effect"
    )
    command_loss_probability: float = Field(
        ge=0.0, le=1.0, description="Chance a command is never received"
    )
    acknowledges: bool = Field(
        description="False models open-loop IR, where ack is always UNKNOWN"
    )


class RoomConfig(_Section):
    """Ground-truth plant parameters.

    Deliberately expressed as physical R, C and gains rather than as a1 to a4.
    The estimator identifies a different parameterisation of the same system,
    which is what keeps DESIGN.md section 5.10's independence rule meaningful.
    """

    thermal_capacitance_j_per_k: float = Field(
        gt=0.0, description="C; thermal mass of the room"
    )
    thermal_resistance_k_per_w: float = Field(
        gt=0.0, description="R; coupling to ambient"
    )
    cooling_power_w: float = Field(
        gt=0.0, description="Air conditioner cooling capacity at full command"
    )
    occupant_gain_w: float = Field(ge=0.0, description="Internal gain per occupancy")
    initial_temperature_c: float
    solar_gain_amplitude_w: float = Field(
        ge=0.0,
        description="Unmodelled disturbance the estimator has no regressor for (R-04)",
    )
    solar_gain_period_s: float = Field(gt=0.0, description="Disturbance cycle length")
    baseline_humidity_pct: float = Field(
        default=55.0,
        ge=0.0,
        le=100.0,
        description="Relative humidity at the reference temperature, per cent",
    )


class OccupancyScheduleConfig(_Section):
    """How the room fills and empties (A-03).

    Not a detail of the scenario but a precondition for identification:
    occupancy is a regressor, and a constant regressor is an intercept the fit
    will use to absorb everything it cannot otherwise explain.
    """

    period_s: float = Field(gt=0.0, description="One cycle of coming and going")
    occupied_fraction: float = Field(
        gt=0.0,
        lt=1.0,
        description="Share of each cycle somebody is present. Strictly between "
        "zero and one: either end is a constant regressor again.",
    )


class SimConfig(_Section):
    room: RoomConfig
    sensor_noise: SensorNoiseConfig
    actuator: SimActuatorConfig
    occupancy: OccupancyScheduleConfig
    outdoor_mean_c: float
    outdoor_amplitude_c: float = Field(ge=0.0)
    outdoor_period_s: float = Field(gt=0.0)
    random_seed: int = Field(
        ge=0, description="Fixed so a scenario replays identically (FR-62)"
    )


class Config(_Section):
    """The whole configuration tree."""

    mqtt: MqttConfig
    loop: LoopConfig
    estimator: EstimatorConfig
    persistence: PersistenceConfig
    controller: ControllerConfig
    validator: ValidatorConfig
    detectors: DetectorsConfig
    evaluation: EvaluationConfig = EvaluationConfig()
    mode: ModeConfig
    sensors: SensorsConfig
    io: IoConfig
    speech: SpeechConfig
    reasoning: ReasoningConfig
    assistance: AssistanceConfig
    sim: SimConfig

    @model_validator(mode="after")
    def _compressor_dwell_agrees_across_components(self) -> Config:
        """The controller and validator both enforce dwell (DESIGN.md 5.3).

        Independent enforcement is deliberate. Disagreeing on the value is
        not: the controller would emit commands the validator always blocks.
        """
        if self.controller.min_off_s != self.validator.min_off_s:
            raise ValueError(
                f"controller.min_off_s ({self.controller.min_off_s}) must equal "
                f"validator.min_off_s ({self.validator.min_off_s})"
            )
        return self

    @model_validator(mode="after")
    def _default_setpoint_is_itself_admissible(self) -> Config:
        """The fallback must survive the gate it falls back through.

        A default outside V-1's bounds would be clamped on every startup, so
        the system would never actually hold the value it is configured to.
        """
        bounds = self.validator.setpoint_bounds_c
        if not bounds.contains(self.controller.default_setpoint_c):
            raise ValueError(
                f"controller.default_setpoint_c "
                f"({self.controller.default_setpoint_c}) is outside "
                f"validator.setpoint_bounds_c [{bounds.low}, {bounds.high}]"
            )
        return self


def load_config(path: Path) -> Config:
    """Read and validate a configuration file.

    :raises ConfigError: if the file is missing, is not a YAML mapping, or
        fails validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config at {path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config at {path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"config at {path} must be a mapping, got {type(raw).__name__}")

    try:
        return Config.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"config at {path} is invalid: {exc}") from exc
