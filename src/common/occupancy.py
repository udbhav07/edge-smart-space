"""Deriving binary occupancy from the two devices that report it (FR-02).

A PIR reports *motion*, not presence, and the difference is the whole problem.
A person reading quietly sets off nothing, so raw PIR says an occupied room is
empty within a minute or two. The fix is a hold-off: the room stays occupied
until nothing has been seen for long enough that absence is the better
explanation.

A door reed switch fills the other gap. Motion tells you somebody is there;
a door tells you the population may have changed. Treating an opening as
evidence of presence is the conservative reading and it is the one that
matches how the space is used -- somebody has either just come in, or just
left and may come back.

**Conservative on purpose.** Every ambiguous case resolves to *occupied*,
because the cost is asymmetric: believing an empty room is occupied wastes
some cooling, while believing an occupied room is empty makes the system stop
cooling a room with a person in it. Section 7.1 makes the same choice for a
failed PIR.

This lives in ``common`` for the reason ``injection`` does: the simulator and
the ESP32 adapter must derive occupancy the same way, and neither can import
the other. The rule is Layer 1's, but the statement of it is shared.
"""

from __future__ import annotations

from src.common.clock import Clock


class OccupancyTracker:
    """Binary occupancy with a vacancy hold-off.

    Holds only the moment of the last evidence, so there is nothing to bound
    and nothing to grow (NFR-05).
    """

    def __init__(self, hold_off_s: float, clock: Clock) -> None:
        if hold_off_s < 0.0:
            raise ValueError(
                f"hold-off cannot be negative, got {hold_off_s!r}"
            )
        self._hold_off_s = hold_off_s
        self._clock = clock
        self._last_evidence_s: float | None = None

    @property
    def hold_off_s(self) -> float:
        """How long the room stays occupied after the last sign of life."""
        return self._hold_off_s

    @property
    def occupied(self) -> bool:
        """Whether somebody is believed to be present.

        False before any evidence has ever arrived. A room nothing has been
        seen in is empty, which is the correct starting assumption -- the
        conservative-to-occupied rule applies to *losing* evidence, not to
        never having had any, or the system would cool an empty building from
        the moment it booted.
        """
        if self._last_evidence_s is None:
            return False
        return self._clock.monotonic() - self._last_evidence_s < self._hold_off_s

    @property
    def quiet_for_s(self) -> float:
        """How long since the last evidence. Zero if there has never been any."""
        if self._last_evidence_s is None:
            return 0.0
        return self._clock.monotonic() - self._last_evidence_s

    def motion(self) -> None:
        """A PIR fired: somebody moved."""
        self._observe()

    def door_transition(self) -> None:
        """The reed switch changed: somebody came in, or left.

        Both are treated as presence. An opening cannot distinguish arrival
        from departure, and the conservative reading of an ambiguous signal is
        that the room is occupied.
        """
        self._observe()

    def _observe(self) -> None:
        self._last_evidence_s = self._clock.monotonic()
