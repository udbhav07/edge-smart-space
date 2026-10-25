"""What the console knows, folded from the blackboard (FR-56, FR-60).

Everything here is derived from messages any subscriber could read: the
console holds no state the rest of the system lacks, and loses nothing it
cannot rebuild by reconnecting, because the topics that matter are retained.

:class:`ConsoleState` takes ``(topic, payload)`` and returns the events a
browser should be pushed. It never touches the network, so it is tested
without a broker or a web server.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common import topics
from src.common.clock import Clock
from src.common.config import ConsoleConfig
from src.common.mqtt_client import PAYLOAD_ENCODING, topic_matches
from src.common.schemas import (
    ActuatorState,
    AssistantReply,
    Coefficients,
    CommandKind,
    ConsoleUtterance,
    FaultDiagnosis,
    FaultEvent,
    Goal,
    InjectionCommand,
    ModeReset,
    ModeState,
    PreferenceHint,
    ProfileContext,
    ReasoningRecord,
    SensorHealth,
    SensorReading,
    SpeechStatus,
    TariffState,
    ThermalEstimate,
    TrackingState,
    ValidationVerdict,
    Verdict,
)
from src.common.tools import ToolCatalogue, ToolInvocation, ToolResult, ToolStatus

LOGGER = logging.getLogger(__name__)

#: Command kinds that only restate the state in force. Their verdicts arrive
#: every five seconds and say nothing a person needs to see.
_RESTATEMENTS = frozenset({CommandKind.MAINTAIN.value, CommandKind.HOLD.value})

#: A typed utterance and the hint the reasoning layer makes of it arrive
#: separately; the hint's transcript is matched against what was typed this
#: recently to know it was not spoken.
_TYPED_KEPT = 32

Event = dict[str, Any]

#: Topics the board remembers. Fault ids are unique per occurrence, so the
#: table would otherwise grow by one entry for every fault ever raised.
_TOPICS_KEPT = 400
#: Arrivals kept per topic to measure its rate.
_ARRIVALS_KEPT = 60
#: Longest payload text kept per topic; a board shows it, it does not archive it.
_PAYLOAD_KEPT = 32000
#: Bookings, mails and inbox messages kept.
_RECORDS_KEPT = 100
#: Arrivals, and the time they must span, before a rate is worth stating (S-08).
_RATE_MIN_ARRIVALS = 5
_RATE_MIN_SPAN_S = 30.0

#: What the occupant does next, for the modes that wait on a person (S-14).
_NEXT_STEP = {
    "DEGRADED_ACTUATOR": " When the air conditioner has been checked, press Reset on Faults & control.",
    "SAFE_HOLD": " Nothing will be actuated until someone checks the room and presses Reset on Faults & control.",
}

#: A point of a periodic series older than this on arrival is history from
#: before this console was listening, not a reading (S-07).
_STALE_POINT_S = 60.0

#: Every declared topic, so the board can say whether one is retained.
_SPECS = tuple(value for value in vars(topics).values() if isinstance(value, topics.TopicSpec))

#: Series charted from the estimate and the coefficients (section 5.2).
_THERMAL_SERIES = ("t_in", "t_pred", "residual", "residual_sigma", "model_confidence")
_COEFFICIENT_SERIES = ("a1", "a2", "a3", "a4", "trace_p")


def _dump(message) -> dict[str, Any]:
    return message.model_dump(mode="json", by_alias=True)


@dataclass
class _Room:
    sensors: dict[str, dict] = field(default_factory=dict)
    health: dict[str, dict] = field(default_factory=dict)
    thermal: dict | None = None
    coefficients: dict | None = None
    actuator: dict | None = None
    mode: dict | None = None
    goal: dict | None = None
    tracking: dict | None = None
    tariff: dict | None = None
    speech: dict | None = None


class ConsoleState:
    """The console's view of the room, rebuilt from the blackboard."""

    def __init__(self, config: ConsoleConfig, clock: Clock, inbox_path: Path | None = None) -> None:
        self._config = config
        self._clock = clock
        #: Where the inbox is kept across restarts; None keeps it in memory
        #: only, which is what every test wants (S-10).
        self._inbox_path = inbox_path
        self._room = _Room()
        self._faults: dict[str, dict] = {}
        self._diagnoses: dict[str, dict] = {}
        self._profiles: dict[str, dict] = {}
        self._injections: dict[str, dict] = {}
        self._conversation: deque[dict] = deque(maxlen=config.conversation_kept)
        self._activity: deque[dict] = deque(maxlen=config.activity_kept)
        #: Named series for the charts: one per sensor, the estimate and
        #: the coefficients, each a bounded run of [ts, value] pairs.
        self._series: dict[str, deque] = {}
        self._topics: OrderedDict[str, dict] = OrderedDict()
        self._bookings: deque[dict] = deque(maxlen=_RECORDS_KEPT)
        self._sent: deque[dict] = deque(maxlen=_RECORDS_KEPT)
        self._inbox: deque[dict] = deque(self._load_inbox(), maxlen=_RECORDS_KEPT)
        self._waiting: OrderedDict[str, dict] = OrderedDict()
        self._invocations: OrderedDict[str, ToolInvocation] = OrderedDict()
        self._calendar: dict = {"entries": [], "as_of": None, "note": ""}
        self._calendar_requests: set[str] = set()
        #: Bookings confirmed here, whose outcome the occupant is told.
        self._confirmed: set[str] = set()
        self._catalogue: list[dict] = []
        self._typed: deque[str] = deque(maxlen=_TYPED_KEPT)
        self._set_here: deque[str] = deque(maxlen=_TYPED_KEPT)
        self._sequence = 0
        #: The last command kind shown, so a re-assertion of the state in
        #: force -- every 60 s, by design (R-02) -- is not shown as a decision.
        self._commanded: str | None = None
        self._handlers: tuple[tuple[str, type, Callable], ...] = (
            (topics.SENSOR_STATE.wildcard(), SensorReading, self._on_reading),
            (topics.SENSOR_HEALTH.wildcard(), SensorHealth, self._on_health),
            (topics.ESTIMATE_THERMAL.wildcard(), ThermalEstimate, self._on_thermal),
            (topics.ESTIMATE_COEFFICIENTS.wildcard(), Coefficients, self._on_coefficients),
            (topics.ACTUATOR_STATE.wildcard(), ActuatorState, self._on_actuator),
            (topics.SYSTEM_MODE.wildcard(), ModeState, self._on_mode),
            (topics.GOAL_ACTIVE.wildcard(), Goal, self._on_goal),
            (topics.GOAL_PROPOSED.wildcard(), Goal, self._on_proposed_goal),
            (topics.CONTROL_TRACKING.wildcard(), TrackingState, self._on_tracking),
            (topics.TARIFF_STATE.wildcard(), TariffState, self._on_tariff),
            (topics.SPEECH_STATUS.wildcard(), SpeechStatus, self._on_speech),
            (topics.FAULT.wildcard(), FaultEvent, self._on_fault),
            (topics.DIAGNOSIS.wildcard(), FaultDiagnosis, self._on_diagnosis),
            (topics.CONTEXT_UTTERANCE.wildcard(), ConsoleUtterance, self._on_typed),
            (topics.CONTEXT_PREFERENCE.wildcard(), PreferenceHint, self._on_hint),
            (topics.CONTEXT_REPLY.wildcard(), AssistantReply, self._on_reply),
            (topics.CONTEXT_PROFILE.wildcard(), ProfileContext, self._on_profile),
            (topics.ASSIST_PROPOSED.wildcard(), ToolInvocation, self._on_invocation),
            (topics.ASSIST_RESULT.wildcard(), ToolResult, self._on_result),
            (topics.ASSIST_CATALOGUE.wildcard(), ToolCatalogue, self._on_catalogue),
            (topics.AUDIT_VALIDATION.wildcard(), ValidationVerdict, self._on_verdict),
            (topics.AUDIT_REASONING.wildcard(), ReasoningRecord, self._on_reasoning),
            (topics.SYSTEM_RESET.wildcard(), ModeReset, self._on_reset),
            (topics.INJECT.wildcard(), InjectionCommand, self._on_injection),
        )

    # --- what the browser is given -------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything, for a browser that has just connected (or polls)."""
        room = self._room
        return {
            "server_ts": self._clock.now(),
            "room": {
                "sensors": room.sensors,
                "health": room.health,
                "thermal": room.thermal,
                "coefficients": room.coefficients,
                "actuator": room.actuator,
                "mode": room.mode,
                "goal": room.goal,
                "tracking": room.tracking,
                "tariff": room.tariff,
            },
            "speech": room.speech,
            "series": {key: list(points) for key, points in self._series.items()},
            "topics": [_board_row(entry) for entry in self._topics.values()],
            "bookings": list(self._bookings),
            "sent": list(self._sent),
            "inbox": list(self._inbox),
            "faults": list(self._faults.values()),
            "diagnoses": self._diagnoses,
            "profiles": list(self._profiles.values()),
            "injections": self._injections,
            "conversation": list(self._conversation),
            "activity": list(self._activity),
            "waiting": list(self._waiting.values()),
            "calendar": self._calendar,
            "tools": self._catalogue,
            "operator_controls": self._config.operator_controls,
        }

    def accept(self, topic: str, payload: bytes) -> list[Event]:
        """Fold one message in; returns the events to push."""
        board = self._on_board(topic, payload)
        return board + self._handle(topic, payload)

    def _handle(self, topic: str, payload: bytes) -> list[Event]:
        for pattern, schema, handler in self._handlers:
            if not topic_matches(pattern, topic):
                continue
            if not payload:
                return self._withdrawn(topic)
            try:
                message = schema.model_validate_json(payload.decode(PAYLOAD_ENCODING))
            except (ValueError, UnicodeDecodeError) as exc:
                LOGGER.debug("ignoring %s: %s", topic, exc)
                return []
            return handler(topic, message) or []
        return []

    # --- the console's own requests ------------------------------------

    def expect_calendar(self, invocation_id: str) -> None:
        """A calendar read the console asked for; its answer fills the view."""
        self._calendar_requests.add(invocation_id)

    def set_from_console(self, text: str) -> None:
        """A hint the console published itself: shown as the occupant's, typed."""
        self._set_here.append(text)

    def invocation(self, invocation_id: str) -> ToolInvocation | None:
        return self._invocations.get(invocation_id)

    def settle(self, invocation_id: str, outcome: str) -> list[Event]:
        """A booking answered from the console: take it off the waiting list."""
        entry = self._waiting.pop(invocation_id, None)
        if entry is None:
            return []
        if outcome == "Confirmed":
            self._confirmed.add(invocation_id)
        return [self._event("waiting", list(self._waiting.values())), self._step("booking", f"{outcome}: {entry['summary']}", tone="info")]

    # --- room ----------------------------------------------------------

    def _on_reading(self, _topic, reading: SensorReading):
        self._room.sensors[reading.sensor_id] = _dump(reading)
        return [self._event("sensor", _dump(reading)), *self._live_point(reading.sensor_id, reading.ts, reading.value)]

    def _on_health(self, _topic, health: SensorHealth):
        self._room.health[health.sensor_id] = _dump(health)
        return [self._event("health", _dump(health))]

    def _on_thermal(self, _topic, estimate: ThermalEstimate):
        self._room.thermal = _dump(estimate)
        return [self._event("thermal", self._room.thermal)] + [
            point for name in _THERMAL_SERIES for point in self._live_point(name, estimate.ts, getattr(estimate, name))
        ]

    def _on_coefficients(self, _topic, coefficients: Coefficients):
        self._room.coefficients = _dump(coefficients)
        return [self._event("coefficients", self._room.coefficients)] + [
            point for name in _COEFFICIENT_SERIES for point in self._live_point(name, coefficients.ts, getattr(coefficients, name))
        ]

    def _on_actuator(self, _topic, state: ActuatorState):
        self._room.actuator = _dump(state)
        return [self._event("actuator", self._room.actuator)]

    def _on_mode(self, _topic, state: ModeState):
        before = (self._room.mode or {}).get("mode")
        self._room.mode = _dump(state)
        events = [self._event("mode", self._room.mode)]
        if before is not None and before != state.mode.value:
            tone = "good" if state.mode.value == "NORMAL" else "alert"
            events.append(self._step("mode", f"Mode {before} → {state.mode.value}: {state.reason}", tone=tone))
            next_step = _NEXT_STEP.get(state.mode.value, "")
            events.append(self._say("system", f"The room is now in {state.mode.value.replace('_', ' ').lower()} mode: {state.reason}.{next_step}", tone=tone))
            events.append(self._mail_in("Room", f"Now {state.mode.value.replace('_', ' ').lower()}", f"{state.reason or 'No reason given.'}{next_step}", "mode", tone))
        return events

    def _on_goal(self, _topic, goal: Goal):
        self._room.goal = _dump(goal)
        return [self._event("goal", self._room.goal)]

    def _on_proposed_goal(self, _topic, goal: Goal):
        who = goal.source.value
        return [self._step("goal", f"{who.capitalize()} proposed {goal.setpoint_c:g} °C", detail=goal.rationale)]

    def _on_tracking(self, _topic, tracking: TrackingState):
        self._room.tracking = _dump(tracking)
        return [
            self._event("tracking", self._room.tracking),
            self._point("held_c", tracking.ts, tracking.tracked_c),
            self._point("asked_c", tracking.ts, tracking.setpoint_c),
        ]

    def _on_tariff(self, _topic, tariff: TariffState):
        self._room.tariff = _dump(tariff)
        return [self._event("tariff", self._room.tariff)]

    def _on_speech(self, _topic, status: SpeechStatus):
        self._room.speech = _dump(status)
        return [self._event("speech", self._room.speech)]

    # --- faults ----------------------------------------------------------

    def _on_fault(self, _topic, event: FaultEvent):
        new = event.fault_id not in self._faults
        self._faults[event.fault_id] = _dump(event)
        events = [self._event("faults", list(self._faults.values()))]
        if new:
            events.append(self._step("fault", f"{event.detector.value} raised on {event.subject}", tone="alert", detail=", ".join(f"{k} {v:.3g}" for k, v in sorted(event.evidence.items())[:4])))
        return events

    def _on_diagnosis(self, _topic, diagnosis: FaultDiagnosis):
        new = diagnosis.fault_id not in self._diagnoses
        self._diagnoses[diagnosis.fault_id] = _dump(diagnosis)
        events = [self._event("diagnoses", self._diagnoses)]
        if new:
            events.append(self._say("system", diagnosis.user_message, tone="alert", fault_id=diagnosis.fault_id))
            events.append(self._mail_in("Fault diagnosis", diagnosis.primary_hypothesis.value.replace("_", " ").capitalize(), diagnosis.user_message, "fault", "alert"))
        return events

    def _withdrawn(self, topic: str) -> list[Event]:
        """An empty retained payload: the thing it described is over."""
        leaf = topic.rsplit("/", 1)[-1]
        if topic_matches(topics.FAULT.wildcard(), topic) and self._faults.pop(leaf, None):
            return [self._event("faults", list(self._faults.values())), self._step("fault", f"Fault {leaf} retired", tone="good")]
        if topic_matches(topics.DIAGNOSIS.wildcard(), topic) and self._diagnoses.pop(leaf, None):
            return [self._event("diagnoses", self._diagnoses)]
        if topic_matches(topics.INJECT.wildcard(), topic) and self._injections.pop(leaf, None):
            return [self._event("injections", self._injections)]
        return []

    def _on_reset(self, _topic, reset: ModeReset):
        return [self._step("operator", f"Reset by {reset.requester}", detail=reset.reason, tone="info")]

    def _on_injection(self, _topic, command: InjectionCommand):
        if command.kind.value == "NONE":
            self._injections.pop(command.subject, None)
        else:
            self._injections[command.subject] = _dump(command)
        return [self._event("injections", self._injections)]

    # --- conversation ----------------------------------------------------

    def _on_typed(self, _topic, utterance: ConsoleUtterance):
        self._typed.append(utterance.text)
        return [self._say("occupant", utterance.text, source="console", profile_id=utterance.profile_id)]

    def _on_hint(self, _topic, hint: PreferenceHint):
        said = hint.transcript or hint.rationale
        events = []
        if said in self._set_here:
            self._set_here.remove(said)
            events.append(self._say("occupant", said, source="console", profile_id=hint.profile_id))
        elif said and said not in self._typed:
            events.append(self._say("occupant", said, source="voice", profile_id=hint.profile_id))
        target = f", {hint.target_c:g} °C" if hint.target_c is not None else ""
        comfort = f", {hint.comfort.value}" if hint.comfort.value != "unchanged" else ""
        events.append(self._step("understood", f"Understood as {hint.intent.value}{target}{comfort}", detail=said))
        return events

    def _on_reply(self, _topic, reply: AssistantReply):
        return [self._say("assistant", reply.reply, intent=reply.intent.value, profile_id=reply.profile_id, waiting=reply.awaiting_confirmation)]

    def _on_profile(self, _topic, context: ProfileContext):
        self._profiles[context.profile_id] = {
            "profile_id": context.profile_id,
            "display_name": context.display_name,
            "facts": list(context.facts),
            "preferences": dict(context.preferences),
            "suggestions": list(context.suggestions),
        }
        return [self._event("profiles", list(self._profiles.values()))]

    # --- tools -------------------------------------------------------------

    def _on_invocation(self, _topic, invocation: ToolInvocation):
        self._invocations[invocation.invocation_id] = invocation
        while len(self._invocations) > self._config.activity_kept:
            self._invocations.popitem(last=False)
        if invocation.invocation_id in self._calendar_requests:
            return []
        arguments = ", ".join(f"{k}={v}" for k, v in invocation.arguments.items() if k != "profile_id")
        return [self._step("tool", f"Called {invocation.tool}", detail=arguments, ref=invocation.invocation_id)]

    def _on_result(self, _topic, result: ToolResult):
        if result.invocation_id in self._calendar_requests:
            self._calendar_requests.discard(result.invocation_id)
            self._calendar = _calendar_from(result, self._clock.now())
            return [self._event("calendar", self._calendar)]
        events = []
        if result.status is ToolStatus.CONFIRMATION_REQUIRED:
            invocation = self._invocations.get(result.invocation_id)
            if invocation is not None:
                self._waiting[result.invocation_id] = {
                    "invocation_id": result.invocation_id,
                    "tool": result.tool,
                    "arguments": {k: v for k, v in invocation.arguments.items() if k != "profile_id"},
                    "expires_ts": invocation.expires_ts,
                    "summary": _summary(invocation),
                }
                events.append(self._event("waiting", list(self._waiting.values())))
        elif result.invocation_id in self._waiting:
            self._waiting.pop(result.invocation_id)
            events.append(self._event("waiting", list(self._waiting.values())))
        if result.invocation_id in self._confirmed and result.status is not ToolStatus.CONFIRMATION_REQUIRED:
            # The answer to a yes given here: said in the conversation, not
            # only in the trace, or the occupant never learns it was booked.
            self._confirmed.discard(result.invocation_id)
            events.append(self._say("assistant", result.message or f"{result.tool}: {result.status.value}", intent="service"))
        tone = "good" if result.status is ToolStatus.OK else "info" if result.status is ToolStatus.CONFIRMATION_REQUIRED else "alert"
        events.append(self._step("result", f"{result.tool}: {result.status.value.replace('_', ' ').lower()}", detail=result.message, tone=tone, ref=result.invocation_id))
        if result.tool in ("schedule_event", "cancel_event", "move_event", "keep_event") and result.status is ToolStatus.OK:
            events.append(self._event("calendar_stale", True))
            events.append(self._mail_in("Calendar", result.tool.replace("_", " ").capitalize(), result.message, "calendar", "good"))
        if result.tool == "pencil_event" and result.status is ToolStatus.OK and result.detail.get("added"):
            # Overheard, not asked for (FR-77): said so in the inbox, so the
            # occupant learns it is there and can keep or discard it.
            events.append(self._event("calendar_stale", True))
            events.append(self._mail_in("Calendar", "Pencilled in from conversation", result.message, "calendar", "info"))
        if result.status is ToolStatus.OK and result.tool == "book_travel":
            events.extend(self._booked(result))
        if result.status is ToolStatus.OK and result.tool == "send_mail":
            events.extend(self._mailed(result))
        return events

    def _on_catalogue(self, _topic, catalogue: ToolCatalogue):
        self._catalogue = [{"name": t.name, "purpose": t.purpose, "effect": t.effect.value} for t in catalogue.tools]
        return [self._event("tools", self._catalogue)]

    # --- audit ---------------------------------------------------------------

    def _on_verdict(self, _topic, verdict: ValidationVerdict):
        kind = str(verdict.proposed.get("kind", ""))
        if kind in _RESTATEMENTS and verdict.verdict is Verdict.ACCEPTED:
            return []
        if kind and verdict.verdict is Verdict.ACCEPTED:
            if kind == self._commanded:
                return []
            before, self._commanded = self._commanded, kind
            change = f"{before} → {kind}" if before else kind
            return [self._step("command", f"Air conditioner: {change}", tone="info")]
        proposed = verdict.proposed.get("setpoint_c", kind)
        applied = verdict.applied.get("setpoint_c", verdict.applied.get("kind", ""))
        tone = "good" if verdict.verdict is Verdict.ACCEPTED else "warn"
        text = f"Safety gate: {verdict.verdict.value.lower()} {proposed}"
        if verdict.verdict is not Verdict.ACCEPTED:
            text += f" → {applied}"
        return [self._step("safety", text, detail=verdict.reason.value, tone=tone)]

    def _on_reasoning(self, _topic, record: ReasoningRecord):
        calls = f", tools: {', '.join(record.tool_calls)}" if record.tool_calls else ""
        return [self._step("model", f"{record.caller.value.capitalize()} thought for {record.latency_s:.1f} s", detail=f"{record.trigger}; {record.prompt_tokens + record.completion_tokens} tokens{calls}; {record.verdict}")]

    # --- bookings and mail ----------------------------------------------------

    def _booked(self, result: ToolResult) -> list[Event]:
        invocation = self._invocations.get(result.invocation_id)
        arguments = dict(invocation.arguments) if invocation is not None else {}
        record = {
            "reference": str(result.detail.get("reference", "")),
            "kind": str(result.detail.get("kind", arguments.get("kind", ""))),
            "destination": str(result.detail.get("destination", arguments.get("destination", ""))),
            "origin": str(arguments.get("origin") or ""),
            "depart_on": str(arguments.get("depart_on", "")),
            "nights": arguments.get("nights"),
            "ts": result.ts,
            "message": result.message,
        }
        self._bookings.append(record)
        return [
            self._event("bookings", list(self._bookings)),
            self._mail_in("Bookings", f"{record['reference']} confirmed", result.message, "booking", "good"),
        ]

    def _mailed(self, result: ToolResult) -> list[Event]:
        record = {
            "reference": str(result.detail.get("reference", "")),
            "to": str(result.detail.get("to", "")),
            "subject": str(result.detail.get("subject", "")),
            "body": str(result.detail.get("body", "")),
            "ts": result.ts,
        }
        self._sent.append(record)
        return [self._event("sent", list(self._sent))]

    def _mail_in(self, sender: str, subject: str, body: str, kind: str, tone: str = "neutral") -> Event:
        now = self._clock.now()
        # Unique across restarts, since the inbox now outlives one (S-10).
        item = {"id": f"m{int(now * 1000)}_{self._next_id()}", "ts": now, "sender": sender, "subject": subject, "body": body, "kind": kind, "tone": tone}
        self._inbox.append(item)
        self._save_inbox()
        return {"type": "inbox", "data": item}

    def _load_inbox(self) -> list[dict]:
        """The inbox as it was before a restart (S-10). Unreadable is empty."""
        if self._inbox_path is None:
            return []
        try:
            items = json.loads(self._inbox_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            LOGGER.warning("inbox at %s unreadable, starting empty: %s", self._inbox_path, exc)
            return []
        return [item for item in items if isinstance(item, dict) and "id" in item][-_RECORDS_KEPT:]

    def _save_inbox(self) -> None:
        if self._inbox_path is None:
            return
        try:
            self._inbox_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._inbox_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(list(self._inbox)), encoding="utf-8")
            os.replace(temporary, self._inbox_path)
        except OSError as exc:
            # The inbox is a courtesy; failing to keep it never stops the console.
            LOGGER.warning("could not keep the inbox at %s: %s", self._inbox_path, exc)

    # --- the MQTT board --------------------------------------------------------

    def _on_board(self, topic: str, payload: bytes) -> list[Event]:
        now = self._clock.now()
        if not payload:
            self._topics.pop(topic, None)
            return [{"type": "mqtt", "data": {"topic": topic, "ts": now, "withdrawn": True}}]
        # Cut only what is too long to keep; cutting before parsing would turn
        # a large message such as the tool catalogue into broken text.
        text = payload.decode(PAYLOAD_ENCODING, errors="replace")
        body: Any = text[:_PAYLOAD_KEPT]
        if len(text) <= _PAYLOAD_KEPT:
            try:
                body = json.loads(text)
            except ValueError:
                pass
        entry = self._topics.pop(topic, None) or {
            "topic": topic,
            "count": 0,
            "first_ts": now,
            "retained": _retained(topic),
            "arrivals": deque(maxlen=_ARRIVALS_KEPT),
        }
        entry["count"] += 1
        entry["last_ts"] = now
        entry["payload"] = body
        entry["arrivals"].append(now)
        self._topics[topic] = entry
        while len(self._topics) > _TOPICS_KEPT:
            self._topics.popitem(last=False)
        return [{"type": "mqtt", "data": {"topic": topic, "ts": now, "payload": body, "count": entry["count"], "retained": entry["retained"]}}]

    def _live_point(self, key: str, ts: float, value: float) -> list[Event]:
        """A point of a series published every few seconds, if it is current.

        A retained estimate from before a restart arrives first and stale:
        charted, it drew a smooth climb from the old room to the new one --
        24 to 29 C "in two minutes" -- that never happened (S-07). The latest
        value still shows in the tiles, with its age; only the chart skips it.
        """
        if self._clock.now() - ts > _STALE_POINT_S:
            return []
        return [self._point(key, ts, value)]

    def _point(self, key: str, ts: float, value: float) -> Event:
        series = self._series.setdefault(key, deque(maxlen=self._config.history_kept))
        point = [round(ts, 3), value]
        series.append(point)
        return {"type": "series", "data": {"key": key, "point": point}}

    # --- event helpers -----------------------------------------------------

    def _next_id(self) -> str:
        self._sequence += 1
        return f"e{self._sequence}"

    def _event(self, kind: str, data: Any) -> Event:
        return {"type": kind, "data": data}

    def _step(self, kind: str, text: str, detail: str = "", tone: str = "neutral", ref: str = "") -> Event:
        step = {"id": self._next_id(), "ts": self._clock.now(), "kind": kind, "text": text, "detail": detail, "tone": tone, "ref": ref}
        self._activity.append(step)
        return {"type": "step", "data": step}

    def _say(self, role: str, text: str, **extra: Any) -> Event:
        turn = {"id": self._next_id(), "ts": self._clock.now(), "role": role, "text": text, **extra}
        self._conversation.append(turn)
        return {"type": "turn", "data": turn}


def _summary(invocation: ToolInvocation) -> str:
    """One line a person can say yes or no to."""
    arguments = invocation.arguments
    if invocation.tool == "send_mail":
        # Live: "A mock send_mail" (S-09).
        to = str(arguments.get("to", "")).strip() or "someone"
        subject = str(arguments.get("subject", "")).strip()
        return f"A mock email to {to}" + (f": {subject}" if subject else "")
    kind = str(arguments.get("kind", invocation.tool))
    where = str(arguments.get("destination", "")).strip()
    origin = str(arguments.get("origin") or "").strip()
    when = str(arguments.get("depart_on", "")).strip()[:10]
    nights = arguments.get("nights")
    if kind == "hotel":
        text = "A mock hotel" + (f" in {where}" if where else "")
        if nights:
            text += f", {nights} night{'s' if str(nights) != '1' else ''}"
        return text + (f" from {when}" if when else "")
    text = f"A mock {kind}" + (f" from {origin}" if origin else "") + (f" to {where}" if where else "")
    return text + (f" on {when}" if when else "")


def _calendar_from(result: ToolResult, now: float) -> dict:
    """The entries a get_events result lists, from its flat detail text."""
    if result.status is not ToolStatus.OK:
        return {"entries": [], "as_of": now, "note": result.message}
    tentative = {event_id for event_id in str(result.detail.get("tentative_ids", "")).split(",") if event_id}
    entries = []
    for chunk in str(result.detail.get("entries", "")).split("; "):
        parts = chunk.split(" ", 2)
        if len(parts) == 3:
            entry = {"event_id": parts[0], "starts_at": parts[1], "subject": parts[2], "tentative": parts[0] in tentative}
            if entry["tentative"]:
                entry["heard"] = str(result.detail.get(f"heard_{parts[0]}", ""))
            entries.append(entry)
    return {"entries": entries, "as_of": now, "note": "" if entries else "Nothing in the calendar."}


def _board_row(entry: dict) -> dict:
    """A topic as the board shows it: arrivals reduced to a rate per minute.

    Only once there are enough to measure one: live, two messages a few
    seconds apart showed a topic published on change as "19.6/min" (S-08).
    """
    arrivals = entry["arrivals"]
    span = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0
    measurable = len(arrivals) >= _RATE_MIN_ARRIVALS and span >= _RATE_MIN_SPAN_S
    rate = (len(arrivals) - 1) * 60.0 / span if measurable else 0.0
    row = {key: value for key, value in entry.items() if key != "arrivals"}
    row["per_minute"] = round(rate, 1)
    return row


def _retained(topic: str) -> bool:
    """Whether the topic contract retains it; unknown topics are not."""
    return any(topic_matches(spec.wildcard(), topic) and spec.retain for spec in _SPECS)


def _jsonable(value: Any) -> Any:
    if isinstance(value, deque):
        return list(value)
    raise TypeError(f"not serialisable: {type(value).__name__}")


def encode(event: Event) -> str:
    return json.dumps(event, separators=(",", ":"), ensure_ascii=False, default=_jsonable)
