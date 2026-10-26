"""What the occupant can do from the console, as blackboard messages (FR-56 to FR-58).

Each action publishes exactly what an existing path publishes, so nothing
here is a new authority:

* **say** -- the words, on ``space/context/utterance``; the reasoning layer
  understands them with the same call it uses for speech.
* **set_target** -- a preference hint, which the goal manager and the
  validator gate exactly as a spoken one (FR-57). Never an actuator command.
* **answer** -- a booking's invocation republished verbatim on
  ``space/assist/confirmed``, as ``tools.confirm`` does (FR-54); a no
  publishes nothing.
* **calendar** -- ``get_events`` and ``cancel_event`` through the executor,
  requested by the console (FR-58).
* **reset**, **inject** -- the operator tools' messages, offered only when
  ``console.operator_controls`` is on (FR-31).
"""

from __future__ import annotations

import itertools
import logging
from datetime import timedelta

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.injection import (
    ACTUATOR_SUPPORTED_FAULTS,
    BINARY_SUPPORTED_FAULTS,
    CONTINUOUS_SUPPORTED_FAULTS,
    FAULTS_REQUIRING_MAGNITUDE,
    InjectedFault,
)
from src.common.localtime import local_time
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Comfort,
    ConsoleUtterance,
    InjectionCommand,
    Intent,
    ModeReset,
    PreferenceHint,
    SpeechSettings,
    Unit,
)
from src.common.tools import ToolInvocation, ToolRequester
from src.console.state import ConsoleState

LOGGER = logging.getLogger(__name__)

#: Who the console's own requests are from, on resets and injections.
REQUESTER = "console"


class ActionRefused(ValueError):
    """A console request that cannot be honoured, and why, in plain words."""


class ConsoleActions:
    """Turns clicks into the messages the rest of the system already reads."""

    def __init__(self, config: Config, clock: Clock, blackboard: Blackboard, state: ConsoleState) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._state = state
        self._ids = itertools.count(1)

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{int(self._clock.now())}_{next(self._ids)}"

    # --- conversation ------------------------------------------------------

    def say(self, text: str, profile_id: str = "") -> ConsoleUtterance:
        words = " ".join(text.split())
        if not words:
            raise ActionRefused("there is nothing to say")
        utterance = ConsoleUtterance(
            ts=self._clock.now(), utterance_id=self._id("say"), text=words[:500], profile_id=profile_id
        )
        self._blackboard.publish(topics.CONTEXT_UTTERANCE, utterance)
        return utterance

    def set_target(self, target_c: float, profile_id: str = "") -> PreferenceHint:
        """A comfort setting, as supervisory input the validator gates (FR-57)."""
        said = f"Set the room to {target_c:g} °C"
        hint = PreferenceHint(
            ts=self._clock.now(),
            intent=Intent.ENVIRONMENT,
            comfort=Comfort.UNCHANGED,
            subject="temperature",
            target_c=float(target_c),
            rationale="set from the console",
            transcript=said,
            profile_id=profile_id,
            typed=True,
        )
        return self._prefer(hint, said)

    def comfort(self, direction: str, profile_id: str = "") -> PreferenceHint:
        """Warmer or cooler, as a person says it; control picks the step (FR-80)."""
        try:
            comfort = Comfort(direction)
        except ValueError as exc:
            raise ActionRefused("say warmer or cooler") from exc
        if comfort is Comfort.UNCHANGED:
            raise ActionRefused("say warmer or cooler")
        said = f"A little {comfort.value}"
        hint = PreferenceHint(
            ts=self._clock.now(),
            intent=Intent.ENVIRONMENT,
            comfort=comfort,
            subject="temperature",
            rationale="from Home",
            transcript=said,
            profile_id=profile_id,
            typed=True,
        )
        return self._prefer(hint, said)

    def automatic(self, profile_id: str = "") -> PreferenceHint:
        """Hand the temperature back to the room (FR-81)."""
        said = "Back to automatic"
        hint = PreferenceHint(
            ts=self._clock.now(),
            intent=Intent.ENVIRONMENT,
            comfort=Comfort.UNCHANGED,
            subject="temperature",
            rationale="from Home",
            transcript=said,
            profile_id=profile_id,
            typed=True,
            release=True,
        )
        return self._prefer(hint, said)

    def cooling_off(self, hours: float, profile_id: str = "") -> PreferenceHint:
        """No cooling for a while: the gate's upper bound, held for that long (FR-81).

        Not an off switch: the room still cools if it passes cooling_off_c,
        and the request goes through the same gate as any other.
        """
        if not 0.0 < hours <= 24.0:
            raise ActionRefused("cooling can be off for up to 24 hours")
        target = self._config.console.cooling_off_c
        said = f"Cooling off for {hours:g} h (unless the room passes {target:g} °C)"
        hint = PreferenceHint(
            ts=self._clock.now(),
            intent=Intent.ENVIRONMENT,
            comfort=Comfort.UNCHANGED,
            subject="temperature",
            target_c=target,
            rationale="cooling off, from Home",
            transcript=said,
            profile_id=profile_id,
            typed=True,
            hold_s=hours * 3600.0,
        )
        return self._prefer(hint, said)

    def _prefer(self, hint: PreferenceHint, said: str) -> PreferenceHint:
        self._state.set_from_console(said)
        self._blackboard.publish(topics.CONTEXT_PREFERENCE, hint)
        return hint

    # --- voice (FR-79, FR-82, FR-83) -------------------------------------------

    def speech_settings(self, changes: dict) -> SpeechSettings:
        """Change some of how the room listens and speaks; the rest stays."""
        current = dict(self._state.speech_settings or {})
        current.pop("ts", None)
        known = set(SpeechSettings.model_fields) - {"ts"}
        unknown = set(changes) - known
        if unknown:
            raise ActionRefused(f"unknown settings: {', '.join(sorted(unknown))}")
        settings = SpeechSettings(**{**current, **changes, "ts": self._clock.now()})
        self._blackboard.publish(topics.SPEECH_SETTINGS, settings)
        return settings

    # --- bookings (FR-54) ----------------------------------------------------

    def answer(self, invocation_id: str, yes: bool) -> str:
        invocation = self._state.invocation(invocation_id)
        if invocation is None:
            raise ActionRefused("that request is no longer known; it may have expired")
        if yes:
            self._blackboard.publish(topics.ASSIST_CONFIRMED, invocation)
            self._state.settle(invocation_id, "Confirmed")
            return "confirmed"
        # A no publishes nothing: the booking was never made, and it expires.
        self._state.settle(invocation_id, "Declined")
        return "declined"

    # --- calendar (FR-58) ------------------------------------------------------

    def refresh_calendar(self) -> str:
        now = local_time(self._clock.now(), self._config.site.utc_offset_h)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=self._config.console.calendar_days)
        invocation = self._invoke("get_events", {
            "from_time": start.isoformat(timespec="seconds"),
            "to_time": end.isoformat(timespec="seconds"),
        })
        self._state.expect_calendar(invocation.invocation_id)
        self._blackboard.publish(topics.ASSIST_PROPOSED, invocation)
        return invocation.invocation_id

    def add_entry(self, subject: str, starts_at: str, duration_min: int | None = None) -> str:
        """Put an entry in the occupant's own calendar: a write, so no confirmation (FR-58)."""
        if not subject.strip() or not starts_at.strip():
            raise ActionRefused("an entry needs a title and a start time")
        arguments: dict = {"subject": subject.strip(), "starts_at": starts_at.strip()}
        if duration_min:
            arguments["duration_min"] = int(duration_min)
        return self._send("schedule_event", arguments)

    def book(self, kind: str, destination: str, depart_on: str, origin: str = "", nights: int | None = None) -> str:
        """Ask for a booking. It is a commit: the executor holds it for a yes (FR-54)."""
        if kind not in ("flight", "hotel"):
            raise ActionRefused("a booking is a flight or a hotel")
        if not destination.strip() or not depart_on.strip():
            raise ActionRefused("a booking needs a destination and a date")
        arguments: dict = {"kind": kind, "destination": destination.strip(), "depart_on": depart_on.strip()}
        if kind == "flight" and origin.strip():
            arguments["origin"] = origin.strip()
        if kind == "hotel" and nights:
            arguments["nights"] = int(nights)
        return self._send("book_travel", arguments)

    def send_mail(self, to: str, subject: str, body: str) -> str:
        """Write a mail. A commit like a booking: held for a yes, and only ever a mock."""
        if not to.strip():
            raise ActionRefused("say who the mail is for")
        if not body.strip() and not subject.strip():
            raise ActionRefused("the mail is empty")
        return self._send("send_mail", {"to": to.strip(), "subject": subject.strip(), "body": body.strip()})

    def _send(self, tool: str, arguments: dict) -> str:
        invocation = self._invoke(tool, arguments)
        self._blackboard.publish(topics.ASSIST_PROPOSED, invocation)
        return invocation.invocation_id

    def remove_entry(self, subject: str, starts_at: str) -> str:
        """Delete one entry: the occupant's own calendar, so no confirmation."""
        if not subject.strip():
            raise ActionRefused("say which entry to remove")
        arguments = {"subject": subject.strip()}
        if starts_at:
            arguments["on"] = starts_at[:10]
        invocation = self._invoke("cancel_event", arguments)
        self._blackboard.publish(topics.ASSIST_PROPOSED, invocation)
        return invocation.invocation_id

    def keep_entry(self, event_id: str) -> str:
        """Make an entry pencilled in from conversation a firm one (FR-77)."""
        if not event_id.strip():
            raise ActionRefused("say which entry to keep")
        return self._send("keep_event", {"event_id": event_id.strip()})

    def settle_fact(self, profile_id: str, fact: str, keep: bool) -> str:
        """Keep or dismiss a fact overheard about someone (FR-78).

        The console is where the person says so; it decides nothing itself.
        """
        if not profile_id.strip() or not fact.strip():
            raise ActionRefused("say whose suggestion, and which")
        return self._send("settle_fact", {"profile_id": profile_id.strip(), "fact": fact.strip(), "keep": bool(keep)})

    def _invoke(self, tool: str, arguments: dict) -> ToolInvocation:
        now = self._clock.now()
        return ToolInvocation(
            ts=now,
            invocation_id=self._id("con"),
            tool=tool,
            arguments=arguments,
            requester=ToolRequester.CONSOLE,
            rationale="requested from the console",
            expires_ts=now + self._config.assistance.confirmation_window_s,
        )

    # --- operator (FR-31, section 5.6) ---------------------------------------

    def reset(self, reason: str) -> ModeReset:
        self._require_operator()
        reset = ModeReset(ts=self._clock.now(), requester=REQUESTER, reason=reason.strip() or "reset from the console")
        self._blackboard.publish(topics.SYSTEM_RESET, reset)
        return reset

    def inject(self, subject: str, fault: str, magnitude: float | None = None) -> InjectionCommand:
        self._require_operator()
        try:
            kind = InjectedFault[fault.upper()]
        except KeyError as exc:
            raise ActionRefused(f"{fault!r} is not a fault the system can inject") from exc
        supported = self._supported(subject)
        if kind not in supported:
            raise ActionRefused(f"{subject} cannot be given {kind.value.lower().replace('_', ' ')}")
        if kind in FAULTS_REQUIRING_MAGNITUDE and magnitude is None:
            raise ActionRefused(f"{kind.value.lower()} needs a value")
        command = InjectionCommand(
            ts=self._clock.now(),
            subject=subject,
            kind=kind,
            magnitude=float(magnitude) if kind in FAULTS_REQUIRING_MAGNITUDE else 0.0,
            requester=REQUESTER,
        )
        self._blackboard.publish(topics.INJECT, command, subject=subject)
        return command

    def _supported(self, subject: str) -> frozenset[InjectedFault]:
        if subject == topics.AIR_CONDITIONER_ID:
            return ACTUATOR_SUPPORTED_FAULTS
        for sensor in self._config.sensors.adapters:
            if sensor.sensor_id == subject:
                return BINARY_SUPPORTED_FAULTS if sensor.unit is Unit.BOOLEAN else CONTINUOUS_SUPPORTED_FAULTS
        raise ActionRefused(f"{subject!r} is not a sensor or actuator here")

    def _require_operator(self) -> None:
        if not self._config.console.operator_controls:
            raise ActionRefused("operator controls are turned off for this console")
