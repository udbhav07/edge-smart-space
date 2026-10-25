"""What each operating mode means, said to a person (S-14, FR-79).

One wording, shared: the console's inbox writes it and the speech service
says it aloud, so the occupant never reads one thing and hears another.
"""

from __future__ import annotations

from src.common.schemas import Mode

#: A mode in a sentence a customer understands, not a detector code.
PLAIN_MODE: dict[Mode, str] = {
    Mode.INIT: "The room is starting up.",
    Mode.NORMAL: "Everything is working normally again.",
    Mode.DEGRADED_SENSOR: (
        "The indoor temperature sensor isn't trustworthy right now, so the room "
        "is being kept on the model's estimate."
    ),
    Mode.DEGRADED_ACTUATOR: "The air conditioner isn't cooling the room, so it is on hold.",
    Mode.SAFE_HOLD: (
        "The room has been running without a working sensor or air conditioner "
        "for too long, so it is holding safe."
    ),
}

#: What the occupant does next, for the modes that wait on a person.
NEXT_STEP: dict[Mode, str] = {
    Mode.DEGRADED_ACTUATOR: "When the air conditioner has been checked, press Reset on Faults & control.",
    Mode.SAFE_HOLD: "Nothing will be actuated until someone checks the room and presses Reset on Faults & control.",
}


def mode_sentence(mode: Mode) -> str:
    """The mode, and what to do about it if anything, in one or two sentences."""
    step = NEXT_STEP.get(mode, "")
    return f"{PLAIN_MODE[mode]} {step}".strip()
