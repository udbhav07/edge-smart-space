"""Live terminal view of the blackboard.

    python -m tools.blackboard_view              # repainting summary
    python -m tools.blackboard_view --stream     # one line per message
    python -m tools.blackboard_view --host x --port 1883

A read-only subscriber with no authority over anything (section 4.4). Killing
it changes nothing, and it can be attached to a running system at any time.

It subscribes to ``space/#`` rather than a fixed list, so a topic added next
week shows up here without anyone remembering to update this file. Topics it
recognises are decoded with the real schemas, which means a publisher that
drifts from the contract is visible as a decode failure rather than as
plausible-looking nonsense.

The most useful line is usually the last one. **Silent** names the topics
this build declares but nothing has published to, which is precisely the
question "is the estimator actually running?" answered without guesswork --
the shape of the bug where a service connects, looks healthy, and publishes
nothing at all.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import PAYLOAD_ENCODING, topic_matches
from src.common.schemas import (
    ActuatorState,
    AssistantReply,
    Coefficients,
    CommandKind,
    FaultDiagnosis,
    FaultEvent,
    Goal,
    ModeState,
    PreferenceHint,
    Quality,
    SensorHealth,
    SensorReading,
    ThermalEstimate,
    TrackingState,
    ValidationVerdict,
)
from pathlib import Path

LOGGER = logging.getLogger("blackboard_view")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

#: How often the repainting view redraws. Fast enough to feel live at the 5 s
#: regulatory cadence, slow enough not to fight the terminal.
REFRESH_INTERVAL_S = 1.0

#: Topics whose payload we can decode, in match order.
_SCHEMAS = (
    (topics.SENSOR_STATE, SensorReading),
    (topics.SENSOR_HEALTH, SensorHealth),
    (topics.ESTIMATE_THERMAL, ThermalEstimate),
    (topics.ESTIMATE_COEFFICIENTS, Coefficients),
    (topics.FAULT, FaultEvent),
    (topics.SYSTEM_MODE, ModeState),
    (topics.GOAL_PROPOSED, Goal),
    (topics.GOAL_ACTIVE, Goal),
    (topics.ACTUATOR_STATE, ActuatorState),
    (topics.CONTEXT_PREFERENCE, PreferenceHint),
    (topics.CONTEXT_REPLY, AssistantReply),
    (topics.AUDIT_VALIDATION, ValidationVerdict),
    (topics.DIAGNOSIS, FaultDiagnosis),
    (topics.CONTROL_TRACKING, TrackingState),
)

#: Every topic this build declares, for the silence report.
_DECLARED = tuple(
    value.wildcard()
    for value in vars(topics).values()
    if isinstance(value, topics.TopicSpec)
)

_CLEAR_SCREEN = "\033[H\033[J"


@dataclass
class TopicActivity:
    """What has arrived on one concrete topic."""

    latest: Any = None
    raw: dict | None = None
    count: int = 0
    last_seen_s: float = 0.0
    decode_failures: int = 0

    def age_s(self, now: float) -> float:
        return now - self.last_seen_s


@dataclass
class BlackboardView:
    """Accumulates what the bus has said. Holds no connection of its own."""

    clock: Clock
    started_s: float = 0.0
    activity: dict[str, TopicActivity] = field(default_factory=dict)
    #: The sensors this configuration expects, listed even before -- or
    #: after -- they speak: a sensor that went quiet is the one to see (U-14).
    expected_sensors: tuple[str, ...] = ()
    #: The meter on the unit's supply, and the draw that means it is running,
    #: so COOL beside a standby draw is pointed out (U-21).
    power_sensor_id: str | None = None
    running_power_w: float | None = None

    def __post_init__(self) -> None:
        self.started_s = self.clock.monotonic()

    @property
    def uptime_s(self) -> float:
        return self.clock.monotonic() - self.started_s

    @property
    def message_count(self) -> int:
        return sum(entry.count for entry in self.activity.values())

    def accept(self, topic: str, payload: bytes) -> TopicActivity:
        """Record one message, decoding it if we know its shape.

        An empty payload is a retained message being withdrawn -- a fault
        resolved, a diagnosis retired -- so the topic is forgotten rather
        than counted as a publisher that has drifted from the contract.
        """
        if not payload:
            self.activity.pop(topic, None)
            return TopicActivity()
        entry = self.activity.setdefault(topic, TopicActivity())
        entry.count += 1
        entry.last_seen_s = self.clock.monotonic()

        try:
            entry.raw = json.loads(payload.decode(PAYLOAD_ENCODING))
        except (UnicodeDecodeError, json.JSONDecodeError):
            entry.decode_failures += 1
            entry.raw = None
            return entry

        schema = self._schema_for(topic)
        if schema is None:
            entry.latest = None
            return entry
        try:
            entry.latest = schema.model_validate(entry.raw)
        except ValueError:
            # A publisher has drifted from the contract. Worth seeing as a
            # failure rather than as plausible-looking nonsense.
            entry.decode_failures += 1
            entry.latest = None
        return entry

    @staticmethod
    def _schema_for(topic: str):
        for spec, schema in _SCHEMAS:
            if topic_matches(spec.wildcard(), topic):
                return schema
        return None

    def matching(self, spec: topics.TopicSpec) -> dict[str, TopicActivity]:
        """Every concrete topic seen under one pattern."""
        pattern = spec.wildcard()
        return {
            topic: entry
            for topic, entry in sorted(self.activity.items())
            if topic_matches(pattern, topic)
        }

    def silent_patterns(self) -> tuple[str, ...]:
        """Declared topics nothing has published to.

        The first thing to look at when a service seems to be running but its
        output never appears.
        """
        return tuple(
            pattern
            for pattern in sorted(_DECLARED)
            if not any(topic_matches(pattern, topic) for topic in self.activity)
        )


def _clock_face(seconds: float) -> str:
    minutes, second = divmod(int(seconds), 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _age(entry: TopicActivity, now: float) -> str:
    age = entry.age_s(now)
    return f"{age:5.1f}s ago" if age < 600 else "   stale  "


def summarise(view: BlackboardView) -> str:
    """The repainting view."""
    now = view.clock.monotonic()
    lines = [
        f"=== edge smart space | blackboard ===   up {_clock_face(view.uptime_s)}"
        f"   {view.message_count} msgs",
        "",
    ]

    lines += _sensor_lines(view, now)

    actuators = view.matching(topics.ACTUATOR_STATE)
    if actuators:
        lines += ["", "ACTUATOR"]
        for topic, entry in actuators.items():
            state = entry.latest
            if state is None:
                continue
            simulated = " simulated" if state.simulated else ""
            setpoint = "--" if state.setpoint_c is None else f"{state.setpoint_c:.1f}"
            lines.append(
                f"  {state.actuator_id:<12} {state.kind.value:<9} setpoint {setpoint:<6} "
                f"ack {state.ack.value:<13}{simulated} {_age(entry, now)}"
            )
            standby = _standby_draw(view, state)
            if standby is not None:
                lines.append(
                    f"  {'':<12} told to cool, but the meter reads {standby:.0f} W: "
                    f"the unit is not running"
                )

    thermal = view.matching(topics.ESTIMATE_THERMAL)
    coefficients = view.matching(topics.ESTIMATE_COEFFICIENTS)
    if thermal or coefficients:
        lines += ["", "ESTIMATE"]
    for entry in thermal.values():
        estimate = entry.latest
        if estimate is None:
            continue
        lines.append(
            f"  t_in {estimate.t_in:7.3f}  t_pred {estimate.t_pred:7.3f}  "
            f"residual {estimate.residual:+7.4f}  sigma {estimate.residual_sigma:6.4f}  "
            f"confidence {estimate.model_confidence:5.3f}  {estimate.adaptation.value}"
        )
    for entry in coefficients.values():
        model = entry.latest
        if model is None:
            continue
        lines.append(
            f"  a1 {model.a1:9.6f}  a2 {model.a2:9.6f}  a3 {model.a3:9.6f}  "
            f"a4 {model.a4:9.6f}  trace {model.trace_p:8.4f}  "
            f"{model.samples_since_reset} samples"
        )

    mode = view.matching(topics.SYSTEM_MODE)
    goals = view.matching(topics.GOAL_ACTIVE)
    faults = view.matching(topics.FAULT)
    lines.append("")
    for entry in mode.values():
        if entry.latest is not None:
            lines.append(f"MODE      {entry.latest.mode.value}")
    for entry in goals.values():
        if entry.latest is not None:
            lines.append(f"GOAL      {entry.latest.setpoint_c:.1f} C "
                         f"from {entry.latest.source.value}")
    for entry in view.matching(topics.CONTROL_TRACKING).values():
        tracking = entry.latest
        if tracking is not None:
            # What the loop holds, beside what was asked, with the reason:
            # "set to 22" while the room is held at 23 is otherwise a lie (U-04).
            held = f"HOLDING   {tracking.tracked_c:.1f} C  ({tracking.band.value} tariff)"
            if tracking.reason:
                held += f"; asked {tracking.setpoint_c:.1f} C: {tracking.reason}"
            lines.append(held)
    if faults:
        lines.append(f"FAULTS    {len(faults)} raised")
        for topic, entry in list(faults.items())[-3:]:
            event = entry.latest
            if event is not None:
                lines.append(
                    f"          {event.detector.value} on {event.subject} "
                    f"-> {event.mode_impact.value}"
                )
    else:
        lines.append("FAULTS    none")
    diagnoses = view.matching(topics.DIAGNOSIS)
    if diagnoses:
        # Retained, so someone opening the view after the fault still reads
        # the explanation rather than finding it only in a log (U-14).
        lines.append("DIAGNOSIS")
        for entry in diagnoses.values():
            if entry.latest is not None:
                lines.append(f"          {entry.latest.fault_id}: {entry.latest.user_message}")

    broken = {t: e for t, e in view.activity.items() if e.decode_failures}
    if broken:
        lines += ["", "DECODE FAILURES  (a publisher has drifted from the contract)"]
        for topic, entry in sorted(broken.items()):
            lines.append(f"  {topic}  x{entry.decode_failures}")

    silent = view.silent_patterns()
    if silent:
        lines += ["", "silent: " + ", ".join(silent)]
    return "\n".join(lines)


def _sensor_lines(view: BlackboardView, now: float) -> list[str]:
    """Every sensor heard or expected, the troubled ones first (U-14).

    A sensor that has gone quiet publishes nothing, so a list built only from
    what arrived dropped exactly the one the viewer was opened to look at.
    """
    readings = {topic.split("/")[2]: entry for topic, entry in view.matching(topics.SENSOR_STATE).items()}
    health = {
        entry.latest.sensor_id: entry.latest
        for entry in view.matching(topics.SENSOR_HEALTH).values()
        if entry.latest is not None
    }
    names = set(readings) | set(view.expected_sensors) | set(health)
    if not names:
        return ["SENSORS   (nothing yet)"]

    def troubled(name: str) -> bool:
        status = health.get(name)
        if status is not None:
            return status.quality is not Quality.OK
        return name not in readings

    # The outdoor sensor reports once a minute, so for up to a minute after the
    # view opens it has said nothing. With a retained "ok" from the detectors
    # it is waiting, not silent, and does not belong at the top.

    lines = ["SENSORS"]
    for name in sorted(names, key=lambda name: (not troubled(name), name)):
        status = health.get(name)
        note = ""
        if status is not None and status.active_fault_id:
            note = f"  fault {status.active_fault_id}"
        entry = readings.get(name)
        if entry is None and status is not None and status.quality is Quality.OK:
            lines.append(f"  {name:<12} {'--':>15} {'ok':<8} awaiting its next reading{note}")
            continue
        if entry is None:
            lines.append(f"  {name:<12} {'-- silent':>15} {'':<8} nothing heard since this view started{note}")
            continue
        reading = entry.latest
        if reading is None:
            lines.append(f"  {name:<12} <undecodable>            {_age(entry, now)}{note}")
            continue
        quality = status.quality.value if status is not None else reading.quality.value
        lines.append(
            f"  {name:<12} {reading.value:>9.3f} {reading.unit.value:<5} "
            f"{quality:<8} {_age(entry, now)} {entry.count:>6} msgs{note}"
        )
    return lines


def _standby_draw(view: BlackboardView, state: ActuatorState) -> float | None:
    """The metered draw, when the unit is told to cool but is not running."""
    if state.kind is not CommandKind.COOL or view.power_sensor_id is None or view.running_power_w is None:
        return None
    meter = view.activity.get(topics.SENSOR_STATE.format(sensor_id=view.power_sensor_id))
    if meter is None or meter.latest is None:
        return None
    return meter.latest.value if meter.latest.value < view.running_power_w else None


def describe(topic: str, entry: TopicActivity) -> str:
    """One line for the streaming view."""
    message = entry.latest
    if message is None:
        if entry.raw is None:
            return f"[{_label(topic)}] {topic} <undecodable>"
        return f"[{_label(topic)}] {topic} {_compact(entry.raw)}"

    if isinstance(message, SensorReading):
        return (
            f"[sensor   ] {message.sensor_id:<12} {message.value:>9.3f} "
            f"{message.unit.value:<5} {message.quality.value}"
        )
    if isinstance(message, ActuatorState):
        return (
            f"[actuator ] {message.actuator_id:<12} {message.kind.value:<9} "
            f"ack {message.ack.value}"
        )
    if isinstance(message, ThermalEstimate):
        return (
            f"[thermal  ] t_in {message.t_in:7.3f} t_pred {message.t_pred:7.3f} "
            f"residual {message.residual:+7.4f} {message.adaptation.value}"
        )
    if isinstance(message, Coefficients):
        return (
            f"[coeff    ] a1 {message.a1:.6f} a2 {message.a2:.6f} "
            f"a3 {message.a3:.6f} a4 {message.a4:.6f}"
        )
    if isinstance(message, FaultEvent):
        return (
            f"[fault    ] {message.detector.value} on {message.subject} "
            f"-> {message.mode_impact.value}"
        )
    if isinstance(message, ModeState):
        return f"[mode     ] {message.mode.value}: {message.reason}"
    if isinstance(message, ValidationVerdict):
        return (
            f"[verdict  ] {message.verdict.value} {message.reason.value} "
            f"{_compact(dict(message.proposed))} -> {_compact(dict(message.applied))}"
        )
    if isinstance(message, Goal):
        return (
            f"[goal     ] {message.setpoint_c:.1f} C from {message.source.value} "
            f"in {message.mode.value}"
        )
    if isinstance(message, SensorHealth):
        fault = f" fault {message.active_fault_id}" if message.active_fault_id else ""
        return f"[health   ] {message.sensor_id:<12} {message.quality.value}{fault}"
    if isinstance(message, FaultDiagnosis):
        return f"[diagnosis] {message.fault_id}: {message.user_message}"
    if isinstance(message, PreferenceHint):
        target = f" {message.target_c:g} C" if message.target_c is not None else ""
        return f"[heard    ] {message.transcript or message.rationale!r} -> {message.intent.value}{target}"
    if isinstance(message, AssistantReply):
        # In voice mode nothing speaks the reply; this line is where the
        # occupant's answer appears.
        return f"[reply    ] {message.reply}"
    if isinstance(message, TrackingState):
        reason = f": {message.reason}" if message.reason else ""
        return (
            f"[tracking ] holding {message.tracked_c:.1f} C, asked "
            f"{message.setpoint_c:.1f} C ({message.band.value}){reason}"
        )
    # Anything else as its JSON, never a Python repr: one format a person
    # and a grep can both read (U-22).
    return f"[{_label(topic)}] {topic} {_compact(message.model_dump(mode='json'))}"


def _label(topic: str) -> str:
    """The bracketed kind, from the topic's second level, padded to line up."""
    parts = topic.split("/")
    return f"{parts[1] if len(parts) > 1 else topic:<9}"


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(", ", ": "), ensure_ascii=False)


def _build_client(host: str, port: int, view: BlackboardView, stream: bool):
    import paho.mqtt.client as mqtt

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"blackboard-view-{os.getpid()}",
    )

    def _on_connect(_client, _userdata, _flags, reason_code, _properties=None) -> None:
        # Subscribe here rather than before connecting: MQTT refuses a
        # SUBSCRIBE while the link is down, and drops them on reconnect.
        client.subscribe(topics.ALL_TOPICS, 0)
        print(f"connected to {host}:{port}, watching {topics.ALL_TOPICS}", flush=True)

    def _on_message(_client, _userdata, message) -> None:
        entry = view.accept(message.topic, message.payload)
        if stream:
            print(describe(message.topic, entry), flush=True)

    client.on_connect = _on_connect
    client.on_message = _on_message
    return client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live view of the blackboard.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", default=None, help="Overrides the configured broker.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print one line per message instead of repainting a summary.",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    host = arguments.host or config.mqtt.host
    port = arguments.port or config.mqtt.port

    # Line by line even into a pipe: "--stream | grep fault" printed nothing
    # for 40 s while a block buffer filled (U-22).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, errors="replace")

    clock = RealClock()
    view = BlackboardView(
        clock=clock,
        expected_sensors=tuple(sensor.sensor_id for sensor in config.sensors.adapters),
        power_sensor_id=config.devices.ac_power_sensor_id or None,
        running_power_w=config.devices.ac_running_power_w,
    )
    client = _build_client(host, port, view, arguments.stream)

    try:
        client.connect(host, port, int(config.mqtt.keepalive_s))
    except OSError as exc:
        LOGGER.error("cannot reach the broker at %s:%s: %s", host, port, exc)
        return 1

    client.loop_start()
    try:
        while True:
            clock.sleep(REFRESH_INTERVAL_S)
            if not arguments.stream:
                sys.stdout.write(_CLEAR_SCREEN + summarise(view) + "\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        client.loop_stop()
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
