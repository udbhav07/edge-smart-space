"""The fault lifecycle: which faults are live, and which one matters most.

Five detectors answer independently and none of them knows what the others
said. This is where those answers become the system's position on what is
wrong: a set of active faults, each with the evidence that raised it, and one
of them nominated as dominant (DESIGN.md section 5.5).

**Clear confirmation lives here, and only here.** A fault stays active until
its detector has judged CLEAR continuously for the configured period (FR-30).
Two things follow. A detector that flaps -- D3 watching a value oscillating
across a bound, which is a genuine failure signature -- produces one fault
rather than a storm of raise/clear pairs, each of which would otherwise mint a
new fault id and write another retained topic. And the mode manager needs no
timer of its own: mode follows the active set, so there is one confirmation
period with one owner rather than the same number in two components, which is
how they drift apart.

**UNKNOWN never clears a fault.** If a temperature sensor is stuck and then
goes silent, D2's window empties and it stops being able to say anything --
and the sensor is now *more* broken, not less. A detector that cannot judge is
not a detector reporting health.

**Priority is severity, not recency.** Actuator faults outrank sensor faults
because an actuator fault stops actuation while a sensor fault degrades onto
prediction, and model divergence outranks both because there is no prediction
left to degrade onto (section 5.6). Ties go to the fault detected first, on the
reasoning that the earliest fault is the likelier root cause of the later ones.

What is *not* decided here: the mode. Section 5.6 escalates to SAFE_HOLD when
several faults are active at once, and that is a judgment about the system's
state rather than about any one fault. The aggregator reports the set; the mode
manager decides what to do about it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from src.common.clock import Clock
from src.common.schemas import DetectorId, FaultEvent, Mode
from src.faults.detectors.base import Finding, Judgment, build_fault_event

LOGGER = logging.getLogger(__name__)

#: Severity order for nominating a dominant fault, least severe first. Derived
#: from what each mode does to actuation in section 5.6, not from a preference:
#: degrading onto prediction keeps the loop closed, ceasing actuation does not,
#: and holding is the end of the line.
_SEVERITY_ORDER: Mapping[Mode, int] = MappingProxyType(
    {
        Mode.DEGRADED_SENSOR: 1,
        Mode.DEGRADED_ACTUATOR: 2,
        Mode.SAFE_HOLD: 3,
    }
)

#: Detector order for a tie that timestamps cannot break, so the nomination is
#: deterministic and a replayed recording nominates the same fault (FR-62).
_DETECTOR_ORDER: tuple[DetectorId, ...] = tuple(DetectorId)


@dataclass(frozen=True)
class AggregateOutcome:
    """What changed in one pass, and what stands afterwards.

    Transitions are reported rather than acted on: publishing belongs to the
    service, mode belongs to the mode manager, and an aggregator that did
    either could not be tested without both.
    """

    raised: tuple[FaultEvent, ...] = ()
    cleared: tuple[FaultEvent, ...] = ()
    active: tuple[FaultEvent, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether anything happened worth publishing."""
        return bool(self.raised or self.cleared)


@dataclass
class _ActiveFault:
    """A live fault and how long its detector has been saying otherwise."""

    event: FaultEvent
    clear_since_s: float | None = None


class FaultAggregator:
    """Holds the active fault set and nominates the dominant fault.

    Keyed by detector and subject, so the collection is bounded by the size of
    the detector bank times the number of sensors -- no window, no history, and
    nothing to cap (NFR-05).
    """

    def __init__(self, clock: Clock, clear_confirm_s: float) -> None:
        if clear_confirm_s < 0.0:
            raise ValueError(
                f"clear confirmation cannot be negative, got {clear_confirm_s!r}"
            )
        self._clock = clock
        self._clear_confirm_s = clear_confirm_s
        self._active: dict[tuple[DetectorId, str], _ActiveFault] = {}

    @property
    def clear_confirm_s(self) -> float:
        """How long a clear must hold before a fault is retired (FR-30)."""
        return self._clear_confirm_s

    @property
    def active(self) -> tuple[FaultEvent, ...]:
        """Every live fault, most severe first."""
        return self._ranked(entry.event for entry in self._active.values())

    @property
    def dominant(self) -> FaultEvent | None:
        """The fault that matters most, or None when nothing is wrong.

        Optional is the honest return type: "no faults" is a state the caller
        must handle, and a sentinel fault would have to lie about a detector.
        """
        ranked = self.active
        return ranked[0] if ranked else None

    def ingest(self, findings: Iterable[Finding]) -> AggregateOutcome:
        """Fold one round of judgments into the active set.

        Findings from any subset of detectors are accepted: a detector with
        nothing to say this tick simply does not appear, which is different
        from appearing with an UNKNOWN judgment. Neither clears a fault.
        """
        now = self._clock.now()
        raised: list[FaultEvent] = []
        cleared: list[FaultEvent] = []

        for finding in findings:
            key = (finding.detector, finding.subject)
            entry = self._active.get(key)

            if finding.judgment is Judgment.FAULTED:
                if entry is None:
                    raised.append(self._raise(key, finding, now))
                else:
                    # Still faulted: the clear countdown, if one had started,
                    # never happened.
                    entry.clear_since_s = None
            elif finding.judgment is Judgment.CLEAR and entry is not None:
                retired = self._maybe_retire(key, entry, now)
                if retired is not None:
                    cleared.append(retired)

        return AggregateOutcome(
            raised=tuple(raised),
            cleared=tuple(cleared),
            active=self.active,
        )

    def retire_all(self) -> tuple[FaultEvent, ...]:
        """Discard every active fault, as an operator reset does (section 5.6).

        This is what a manual reset actually means. The operator is not
        asserting that the room is fine -- they are asserting that they have
        looked at it and dealt with it, so the accumulated evidence is stale
        and the question should be asked again from scratch.

        It cannot hide a real fault. Every detector re-gathers from live
        inputs, so a fault that is still present is raised again within its own
        window: a reset re-tests rather than overrides. That is also why it is
        needed at all. D5 reads UNKNOWN whenever cooling is not being
        commanded, and UNKNOWN never retires a fault, so a DEGRADED_ACTUATOR
        that blocks actuation could never observe the evidence that would
        clear it. Without this the system would hold forever.

        :returns: the retired faults, so the caller can withdraw each from the
            blackboard and reset the detector that raised it.
        """
        retired = self.active
        self._active.clear()
        if retired:
            LOGGER.warning(
                "retired %d fault(s) on reset: %s",
                len(retired),
                ", ".join(event.fault_id for event in retired),
            )
        return retired

    def _raise(
        self, key: tuple[DetectorId, str], finding: Finding, now: float
    ) -> FaultEvent:
        event = build_fault_event(finding, now)
        self._active[key] = _ActiveFault(event=event)
        LOGGER.warning(
            "%s raised on %s: %s (confidence %.2f)",
            finding.detector.value,
            finding.subject,
            event.fault_id,
            finding.confidence,
        )
        return event

    def _maybe_retire(
        self, key: tuple[DetectorId, str], entry: _ActiveFault, now: float
    ) -> FaultEvent | None:
        """Retire the fault once its clear has held long enough (FR-30)."""
        if entry.clear_since_s is None:
            entry.clear_since_s = now
            LOGGER.info(
                "%s cleared on %s; holding %.0f s before retiring %s",
                key[0].value,
                key[1],
                self._clear_confirm_s,
                entry.event.fault_id,
            )
            if self._clear_confirm_s > 0.0:
                return None

        if now - entry.clear_since_s < self._clear_confirm_s:
            return None

        del self._active[key]
        LOGGER.info("retired %s", entry.event.fault_id)
        return entry.event

    def _ranked(self, events: Iterable[FaultEvent]) -> tuple[FaultEvent, ...]:
        """Most severe first, then earliest detected, then detector order."""
        return tuple(
            sorted(
                events,
                key=lambda event: (
                    -_SEVERITY_ORDER[event.mode_impact],
                    event.detected_ts,
                    _DETECTOR_ORDER.index(event.detector),
                ),
            )
        )
