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


class ValidatorConfig(_Section):
    """Hard constraints, rules V-1 to V-6 in DESIGN.md section 5.4."""

    setpoint_bounds_c: Bounds = Field(description="V-1 absolute bounds")
    max_step_c: float = Field(gt=0.0, description="V-2 rate limit per invocation")
    min_off_s: float = Field(ge=0.0, description="V-3 compressor dwell")
    min_command_interval_s: float = Field(
        gt=0.0, description="V-4 maximum command frequency"
    )
    goal_max_age_s: float = Field(gt=0.0, description="V-6 staleness horizon")


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


class DetectorsConfig(_Section):
    dropout: DropoutDetectorConfig
    stuck_at: StuckAtDetectorConfig
    out_of_range: OutOfRangeDetectorConfig


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


class SimConfig(_Section):
    room: RoomConfig
    sensor_noise: SensorNoiseConfig
    actuator: SimActuatorConfig
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
    mode: ModeConfig
    sensors: SensorsConfig
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
