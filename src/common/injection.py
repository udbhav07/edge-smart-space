"""Fault-injection vocabulary shared by every Layer 1 implementation.

FR-31 requires every fault class to be injectable without modifying
production code paths. Section 5.9.2 injects into the *sensor adapter*, not
over the blackboard, so injection is a Layer 1 concern and the simulator and
the hardware adapters have to speak the same language about it.

That is why this lives in ``common`` rather than in ``sim``: if the
simulator owned the vocabulary, the ESP32 adapter could not use it without
importing the simulator, and Layer 1 would have two different notions of
what a stuck sensor is. Nothing above Layer 1 imports this at all -- a fault
reaches the detectors as absent, frozen or implausible readings, exactly as
a real fault would, and no consumer can tell the difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InjectedFault(str, Enum):
    """Fault classes injectable at Layer 1, matching detectors D1 to D3.

    D4 (drift) is included because a drifting sensor is injectable at the
    source even though the detector that catches it works on the model
    residual rather than on the signal.
    """

    NONE = "NONE"
    STUCK_AT = "STUCK_AT"
    DROPOUT = "DROPOUT"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    DRIFT = "DRIFT"


@dataclass(frozen=True)
class FaultInjection:
    """What to inject and how hard.

    ``magnitude`` is interpreted per fault: the frozen reading for STUCK_AT,
    the reported value for OUT_OF_RANGE, and degrees per second for DRIFT.
    It is ignored for NONE and DROPOUT.
    """

    kind: InjectedFault
    magnitude: float = 0.0


#: The absence of an injected fault. A sensor in this state still exhibits
#: whatever nominal imperfection its implementation models.
NO_FAULT = FaultInjection(kind=InjectedFault.NONE)

#: Faults meaningful for a two-valued signal. A binary sensor cannot drift
#: and has no range to leave, so injecting either is refused rather than
#: silently ignored: an injection that appears to work and does nothing
#: turns a detection trial into a phantom missed detection.
BINARY_SUPPORTED_FAULTS = frozenset(
    {InjectedFault.NONE, InjectedFault.DROPOUT, InjectedFault.STUCK_AT}
)
