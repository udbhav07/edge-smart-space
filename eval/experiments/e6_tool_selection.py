"""E6: is supervisory tool selection reliable? (DESIGN.md section 8.3)

About fifty hand-built scenarios, run through the real call sites against the
configured model server, scored on three things *reported separately*:

* **Schema validity** -- every tool call's arguments parsed and fitted the
  declaration, and the answer fitted its schema. Section 5.7.4 is blunt about
  this one: under a constrained decode it is close to 100% by construction and
  is not a result. It is reported so that it can be seen not to be one.
* **Tool selection** -- the right tools were called. For the supervisor: it
  read what the decision depends on and ended with ``propose_setpoint``. For
  Personal Context: it called exactly the tool the request needed, or none,
  and classified the intent correctly.
* **Argument plausibility** -- the values made sense. A proposed setpoint
  inside the band the occupant's policy implies; a calendar entry on the day
  and hour that were asked for; a flight to the city that was named.

Everything but the model is real: the supervisor proposes to the actual gate
on the in-process bus, and its snapshot is fed the scenario's room over the
same topics a running system uses. Personal Context's tools are answered by a
recorder that behaves as the executor does -- OK for a read or a write,
CONFIRMATION_REQUIRED for a booking -- so a scenario cannot write to a real
calendar.

The simulated clock stands at Sunday 24 August 2025, 16:10 room time, so every
relative date in a scenario has one right answer.

Run it (needs the inference server): ``python -m eval.experiments.e6_tool_selection``
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import RealClock, SimClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AdaptationState,
    DetectorId,
    FaultClass,
    FaultEvent,
    Intent,
    Mode,
    ModeState,
    ReasoningOutcome,
    SensorReading,
    TariffBand,
    TariffState,
    ThermalEstimate,
    Unit,
)
from src.common.tools import (
    ASSISTANCE_TOOLS,
    ArgumentValue,
    ToolArgumentError,
    ToolResult,
    ToolStatus,
)
from src.control.service import build_service as build_control
from src.reasoning.audit import ReasoningAudit
from src.reasoning.endpoint import ChatEndpoint
from src.reasoning.single_shot import PersonalContext
from src.reasoning.supervisor_agent import EnvironmentalSupervisor
from src.reasoning.supervisor_tools import (
    PROPOSE_SETPOINT,
    RoomSnapshot,
    SupervisorTools,
    ToolAnswer,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
RESULTS_DIR = Path("runs/e6")

#: The reads every supervisor decision depends on: the mode it must report,
#: and the two facts the policy turns on.
_ESSENTIAL_READS = frozenset({"get_active_faults", "get_occupancy", "get_tariff_state"})

#: How close to the setpoint in force a proposal must stay while a fault is
#: active. The policy says hold; this is what "hold" is allowed to mean.
_HOLD_TOLERANCE_C = 0.25


# --- Scenarios --------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorCase:
    """A room state, and the band of setpoints the policy allows in it."""

    label: str
    occupied: bool
    tariff: TariffBand
    fault: bool
    t_in_c: float
    low_c: float
    high_c: float


@dataclass(frozen=True)
class AssistCase:
    """An utterance, the tool it needs, and what its arguments must say.

    ``checks`` maps an argument to what it must contain: a timestamp's
    ``YYYY-MM-DDTHH`` prefix, or a word a string must include.
    """

    label: str
    utterance: str
    tool: str | None
    intent: Intent
    checks: Mapping[str, str] = field(default_factory=dict)


def supervisor_cases(config: Config) -> list[SupervisorCase]:
    """Every combination the policy distinguishes, at two temperatures.

    The band is derived from the policy the supervisor is told, so changing
    the comfort target in config changes what counts as right.
    """
    comfort = config.controller.default_setpoint_c
    relax = config.reasoning.vacancy_relax_c
    peak = config.tariff.peak_offset_c
    cases = []
    for occupied in (True, False):
        for tariff in (TariffBand.NORMAL, TariffBand.PEAK):
            for fault in (False, True):
                for t_in in (23.0, 27.5):
                    shift = peak if tariff is TariffBand.PEAK else 0.0
                    if fault:
                        low = high = comfort
                        low, high = low - _HOLD_TOLERANCE_C, high + _HOLD_TOLERANCE_C
                    elif occupied:
                        low, high = comfort + shift - 0.5, comfort + shift + 0.5
                    else:
                        low, high = comfort + shift, comfort + shift + relax
                    label = (
                        f"{'occupied' if occupied else 'empty'}, {tariff.value}"
                        f"{', sensor fault' if fault else ''}, room {t_in:g} C"
                    )
                    cases.append(
                        SupervisorCase(label, occupied, tariff, fault, t_in, low, high)
                    )
    for t_in in (31.0, 20.0):
        cases.append(
            SupervisorCase(
                f"occupied, normal, room {t_in:g} C (extreme)",
                True, TariffBand.NORMAL, False, t_in, comfort - 0.5, comfort + 0.5,
            )
        )
    for t_in in (31.0, 20.0):
        cases.append(
            SupervisorCase(
                f"empty, peak, room {t_in:g} C (extreme)",
                False, TariffBand.PEAK, False, t_in,
                comfort + peak, comfort + peak + relax,
            )
        )
    return cases


#: Room time is Sunday 2025-08-24 16:10. Every relative date has one answer.
ASSIST_CASES: tuple[AssistCase, ...] = (
    # schedule_event: a write, run on the reasoning layer's own authority
    AssistCase("meeting on Thursday", "put the design review in my calendar on Thursday at 3pm",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-28T15", "subject": "review"}),
    AssistCase("call tomorrow morning", "schedule a call with Priya tomorrow at 10 in the morning",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-25T10", "subject": "Priya"}),
    AssistCase("reminder tonight", "remind me to take my medicine at 9 tonight",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-24T21", "subject": "medicine"}),
    AssistCase("dentist on Tuesday", "add a dentist appointment on Tuesday at 11:30",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-26T11", "subject": "dentist"}),
    AssistCase("gym on Friday", "book the gym session into my calendar for Friday at 7am",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-29T07", "subject": "gym"}),
    AssistCase("lunch on Wednesday", "lunch with Arjun on Wednesday at 1pm, put it in",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-27T13", "subject": "Arjun"}),
    AssistCase("standup tomorrow", "add the team standup tomorrow at 9:15",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-25T09", "subject": "standup"}),
    AssistCase("viva on Saturday", "put my viva rehearsal on Saturday at 4 in the afternoon",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-30T16", "subject": "viva"}),
    AssistCase("hour-long review", "schedule a one hour code review on Monday at 2pm",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-25T14", "duration_min": "60"}),
    AssistCase("call in an hour", "set up a call with the landlord at 5:30 today",
               "schedule_event", Intent.SERVICE, {"starts_at": "2025-08-24T17", "subject": "landlord"}),
    # get_events: a read
    AssistCase("what on Thursday", "what have I got on Thursday?",
               "get_events", Intent.SERVICE, {"from_time": "2025-08-28", "to_time": "2025-08-28"}),
    AssistCase("free tomorrow", "am I free tomorrow?",
               "get_events", Intent.SERVICE, {"from_time": "2025-08-25"}),
    AssistCase("rest of today", "what is left in my calendar today?",
               "get_events", Intent.SERVICE, {"from_time": "2025-08-24"}),
    AssistCase("Friday plans", "do I have anything on Friday?",
               "get_events", Intent.SERVICE, {"from_time": "2025-08-29"}),
    AssistCase("Monday morning", "what's on Monday morning?",
               "get_events", Intent.SERVICE, {"from_time": "2025-08-25"}),
    AssistCase("Wednesday", "read me Wednesday's schedule",
               "get_events", Intent.SERVICE, {"from_time": "2025-08-27"}),
    AssistCase("Tuesday meetings", "any meetings on Tuesday?",
               "get_events", Intent.SERVICE, {"from_time": "2025-08-26"}),
    AssistCase("Saturday", "is Saturday free?",
               "get_events", Intent.SERVICE, {"from_time": "2025-08-30"}),
    # book_travel: a commit, which must come back as a question
    AssistCase("flight to Delhi", "book me a flight to Delhi from Hyderabad on Friday",
               "book_travel", Intent.SERVICE,
               {"kind": "flight", "destination": "Delhi", "depart_on": "2025-08-29"}),
    AssistCase("hotel in Mumbai", "I need a hotel in Mumbai for two nights from Wednesday",
               "book_travel", Intent.SERVICE,
               {"kind": "hotel", "destination": "Mumbai", "depart_on": "2025-08-27", "nights": "2"}),
    AssistCase("flight to Bengaluru", "get me a flight from Hyderabad to Bengaluru tomorrow",
               "book_travel", Intent.SERVICE,
               {"kind": "flight", "destination": "Bengaluru", "depart_on": "2025-08-25"}),
    AssistCase("hotel in Goa", "book a hotel in Goa for three nights starting Saturday",
               "book_travel", Intent.SERVICE,
               {"kind": "hotel", "destination": "Goa", "depart_on": "2025-08-30", "nights": "3"}),
    AssistCase("flight to Chennai", "fly me to Chennai on Thursday from Hyderabad",
               "book_travel", Intent.SERVICE,
               {"kind": "flight", "destination": "Chennai", "depart_on": "2025-08-28"}),
    AssistCase("hotel in Pune", "a hotel room in Pune on Monday for one night please",
               "book_travel", Intent.SERVICE,
               {"kind": "hotel", "destination": "Pune", "depart_on": "2025-08-25", "nights": "1"}),
    # no tool: a preference about the room, or nothing at all
    AssistCase("too warm", "it's too warm in here", None, Intent.ENVIRONMENT),
    AssistCase("a named temperature", "make it 22 degrees", None, Intent.ENVIRONMENT),
    AssistCase("a bit cooler", "could you cool it down a little", None, Intent.ENVIRONMENT),
    AssistCase("too cold", "I'm freezing, warm it up", None, Intent.ENVIRONMENT),
    AssistCase("greeting", "hello there", None, Intent.NONE),
    AssistCase("thanks", "thank you, that's all", None, Intent.NONE),
)


# --- Scoring ----------------------------------------------------------------


@dataclass(frozen=True)
class CaseScore:
    """One scenario's three verdicts, and why."""

    site: str
    label: str
    schema_valid: bool
    selected_right: bool
    plausible: bool
    reached_model: bool
    latency_s: float
    detail: str


class _RecordingTools(SupervisorTools):
    """The supervisor's real tools, with every answer kept for scoring."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.answers: list[tuple[str, ToolAnswer]] = []

    def execute(self, name: str, raw_arguments: str) -> ToolAnswer:
        answer = super().execute(name, raw_arguments)
        self.answers.append((name, answer))
        return answer


def run_supervisor_case(config: Config, case: SupervisorCase, client=None) -> CaseScore:
    """Put the room in the scenario's state and let the supervisor decide."""
    clock = SimClock()
    transport = LoopbackTransport()
    board = Blackboard(config.mqtt, transport)
    snapshot = RoomSnapshot(config, clock, board)
    snapshot.subscribe()
    tools = _RecordingTools(config, clock, board, snapshot)
    supervisor = EnvironmentalSupervisor(
        config,
        clock,
        ChatEndpoint(config.reasoning, clock, client),
        tools,
        ReasoningAudit(clock, board, config.reasoning.max_audit_chars),
    )
    control_board = Blackboard(config.mqtt, transport)
    control = build_control(config, clock, control_board)
    control.subscribe()
    world = Blackboard(config.mqtt, transport)
    for each in (board, control_board, world):
        transport.attach(each)
    _stage(config, clock, world, case)

    run = supervisor.run("cadence")
    record = run.record
    reached = record.outcome is not ReasoningOutcome.UNAVAILABLE
    # An "error" answer is a call that did not parse or did not fit the
    # declaration. A proposal the post-decode check discarded is well-formed
    # and wrong, which is a plausibility failure, not a schema one.
    malformed = [name for name, answer in tools.answers if "error" in answer.content]
    schema_valid = reached and not malformed
    called = set(record.tool_calls)
    selected = PROPOSE_SETPOINT.name in called and _ESSENTIAL_READS <= called
    proposed = None if run.goal is None else run.goal.setpoint_c
    plausible = proposed is not None and case.low_c <= proposed <= case.high_c
    detail = (
        f"called {list(record.tool_calls)}; proposed "
        f"{'nothing' if proposed is None else f'{proposed:.1f} C'} "
        f"(allowed {case.low_c:.1f}-{case.high_c:.1f}); {record.outcome.value}"
        f"{': ' + record.reason if record.reason and run.goal is None else ''}"
    )
    return CaseScore(
        "supervisor", case.label, schema_valid, selected, plausible, reached,
        record.latency_s, detail,
    )


def _stage(config: Config, clock: SimClock, world: Blackboard, case: SupervisorCase) -> None:
    """Publish the scenario's room, as the rest of the system would."""
    now = clock.now()
    fault_ids = ("f_temp01_stuck_e6",) if case.fault else ()
    if case.fault:
        world.publish(
            topics.FAULT,
            FaultEvent(
                fault_id=fault_ids[0],
                detector=DetectorId.D2_STUCK_AT,
                subject=config.estimator.indoor_sensor_id,
                fault_class=FaultClass.SENSOR,
                confidence=0.94,
                detected_ts=now - 120.0,
                evidence={"variance": 0.0001},
                mode_impact=Mode.DEGRADED_SENSOR,
            ),
            fault_id=fault_ids[0],
        )
    world.publish(
        topics.SYSTEM_MODE,
        ModeState(
            ts=now,
            mode=Mode.DEGRADED_SENSOR if case.fault else Mode.NORMAL,
            since_ts=now - 120.0,
            active_fault_ids=fault_ids,
        ),
    )
    occupancy_id = config.estimator.occupancy_sensor_id
    # Present an hour ago; for an empty room, gone for the last half hour, so
    # the snapshot sees a transition and a vacancy duration to report.
    history = [(now - 3600.0, 1.0)]
    if not case.occupied:
        history.append((now - 1800.0, 0.0))
    for ts, value in history:
        world.publish(
            topics.SENSOR_STATE,
            SensorReading(ts=ts, sensor_id=occupancy_id, value=value, unit=Unit.BOOLEAN),
            sensor_id=occupancy_id,
        )
    world.publish(
        topics.SENSOR_STATE,
        SensorReading(
            ts=clock.now(), sensor_id=config.estimator.outdoor_sensor_id,
            value=33.0, unit=Unit.CELSIUS,
        ),
        sensor_id=config.estimator.outdoor_sensor_id,
    )
    world.publish(
        topics.CONTEXT_TARIFF,
        TariffState(
            ts=clock.now(), band=case.tariff, since_ts=clock.now() - 1800.0,
            next_transition_ts=clock.now() + 7200.0, offset_c=config.tariff.peak_offset_c,
        ),
    )
    world.publish(
        topics.ESTIMATE_THERMAL,
        ThermalEstimate(
            ts=clock.now(), t_in=case.t_in_c, t_pred=case.t_in_c - 0.05, residual=0.05,
            residual_sigma=0.15, model_confidence=0.8,
            adaptation=AdaptationState.FROZEN if case.fault else AdaptationState.ACTIVE,
        ),
    )


class _RecordingToolClient:
    """Answers like the executor would, and remembers what it was asked."""

    def __init__(self, clock: SimClock) -> None:
        self._clock = clock
        self.calls: list[tuple[str, dict[str, ArgumentValue]]] = []
        self.malformed: list[str] = []
        self._specs = {spec.name: spec for spec in ASSISTANCE_TOOLS}

    def call(self, tool: str, arguments: dict[str, ArgumentValue], rationale: str) -> ToolResult:
        self.calls.append((tool, dict(arguments)))
        spec = self._specs.get(tool)
        status, message, provider = ToolStatus.OK, "Done.", "e6_recorder"
        if spec is None:
            status, message, provider = ToolStatus.UNKNOWN_TOOL, f"no tool {tool}", ""
            self.malformed.append(tool)
        else:
            try:
                spec.validate_arguments(arguments)
            except ToolArgumentError as exc:
                status, message, provider = ToolStatus.BAD_ARGUMENTS, str(exc), ""
                self.malformed.append(tool)
            else:
                if spec.requires_confirmation:
                    status, message, provider = (
                        ToolStatus.CONFIRMATION_REQUIRED, "needs confirming", "",
                    )
        return ToolResult(
            ts=self._clock.now(), invocation_id=f"inv_e6_{len(self.calls)}", tool=tool,
            status=status, message=message, provider=provider,
        )


def run_assist_case(config: Config, case: AssistCase, client=None) -> CaseScore:
    """Say the scenario's words to Personal Context and score what it does."""
    clock = SimClock()
    transport = LoopbackTransport()
    board = Blackboard(config.mqtt, transport)
    transport.attach(board)
    tools = _RecordingToolClient(clock)
    context = PersonalContext(
        config, clock, ChatEndpoint(config.reasoning, clock, client),
        ReasoningAudit(clock, board, config.reasoning.max_audit_chars), tools=tools,
    )
    hint = context.extract(case.utterance, trigger="e6")
    records = [
        payload for name, payload, _, _ in transport.published
        if name == "space/audit/reasoning"
    ]
    record = json.loads(records[-1]) if records else {}
    outcome = record.get("outcome", ReasoningOutcome.UNAVAILABLE.value)
    reached = outcome != ReasoningOutcome.UNAVAILABLE.value

    schema_valid = reached and not tools.malformed and outcome != ReasoningOutcome.DISCARDED.value
    called = [name for name, _ in tools.calls]
    expected = [] if case.tool is None else [case.tool]
    intent = Intent.NONE if hint is None else hint.intent
    selected = sorted(set(called)) == expected and intent is case.intent
    plausible = selected and _arguments_plausible(tools.calls, case)
    detail = (
        f"called {called or 'nothing'}; intent {intent.value}; {outcome}"
        f"{'; args ' + json.dumps(tools.calls[0][1]) if tools.calls else ''}"
    )
    return CaseScore(
        "personal_context", case.label, schema_valid, selected, plausible, reached,
        float(record.get("latency_s", 0.0)), detail,
    )


def _arguments_plausible(calls, case: AssistCase) -> bool:
    if case.tool is None:
        return not calls
    arguments = next((args for name, args in calls if name == case.tool), None)
    if arguments is None:
        return False
    for name, wanted in case.checks.items():
        value = arguments.get(name)
        if value is None:
            if name == "duration_min" and wanted == "60":
                continue  # an omitted duration is the declared default of 60
            return False
        if wanted.lower() not in str(value).lower():
            return False
    return True


# --- Running it ---------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    cases: int
    reached_model: int
    schema_validity: float
    selection_accuracy: float
    argument_plausibility: float


def summarise(scores: list[CaseScore]) -> Summary:
    """The three rates, over the cases that reached the model at all.

    A server that never answered says nothing about selection, so those cases
    are counted and excluded rather than scored as wrong.
    """
    reached = [score for score in scores if score.reached_model]
    if not reached:
        return Summary(len(scores), 0, 0.0, 0.0, 0.0)

    def rate(predicate: Callable[[CaseScore], bool]) -> float:
        return sum(1 for score in reached if predicate(score)) / len(reached)

    return Summary(
        len(scores),
        len(reached),
        rate(lambda s: s.schema_valid),
        rate(lambda s: s.selected_right),
        rate(lambda s: s.plausible),
    )


def run(config: Config, client=None, limit: int | None = None) -> list[CaseScore]:
    scores: list[CaseScore] = []
    for case in supervisor_cases(config)[:limit]:
        scores.append(run_supervisor_case(config, case, client))
        LOGGER.info("%s", scores[-1].detail)
    for case in ASSIST_CASES[:limit]:
        scores.append(run_assist_case(config, case, client))
        LOGGER.info("%s", scores[-1].detail)
    return scores


def report(scores: list[CaseScore]) -> str:
    lines = [f"{'site':<17} {'schema':>6} {'select':>6} {'args':>6}  scenario"]
    for score in scores:
        def mark(ok: bool) -> str:
            return "ok" if ok else "--" if not score.reached_model else "MISS"

        lines.append(
            f"{score.site:<17} {mark(score.schema_valid):>6} "
            f"{mark(score.selected_right):>6} {mark(score.plausible):>6}  {score.label}"
        )
    for site in ("supervisor", "personal_context"):
        summary = summarise([s for s in scores if s.site == site])
        lines.append(
            f"{site}: {summary.reached_model}/{summary.cases} reached the model; "
            f"schema validity {summary.schema_validity:.0%} (not a result, section "
            f"5.7.4), tool selection {summary.selection_accuracy:.0%}, argument "
            f"plausibility {summary.argument_plausibility:.0%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run experiment E6.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Cases per call site")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    scores = run(config, limit=arguments.limit)
    print(report(scores))
    arguments.out.mkdir(parents=True, exist_ok=True)
    stamp = int(RealClock().now())
    path = arguments.out / f"e6_{config.reasoning.model.replace(':', '_')}_{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "model": config.reasoning.model,
                "base_url": config.reasoning.base_url,
                "scores": [asdict(score) for score in scores],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written to {path}")
    reached = sum(1 for score in scores if score.reached_model)
    if not reached:
        print("no case reached the model: is the inference server running?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
