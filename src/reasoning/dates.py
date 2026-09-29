"""The day an occupant named, worked out from their words rather than by the model.

A 7B model told the coming days in a table still booked "Tuesday at 11" on a
Monday. Which day "Tuesday", "tomorrow" or "next Friday" means is not a
judgement; it is arithmetic on the date, and arithmetic is done here. The
result is used twice (FR-44):

* **Before the call**, to annotate the request -- "Tuesday = 2026-09-29" -- so
  the model copies a date instead of computing one.
* **After the call**, as a post-decode semantic check. When the utterance names
  exactly one day and a tool argument carries a different date, the date is
  replaced by the named one and the time the model chose is kept. A request
  naming no day, or more than one, is left entirely to the model.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_RELATIVE = {
    "day after tomorrow": 2,
    "tomorrow": 1,
    "tonight": 0,
    "today": 0,
    "this evening": 0,
    "this afternoon": 0,
    "this morning": 0,
}

_WEEKDAY_PATTERN = re.compile(r"\b(?:next\s+|this\s+|on\s+)?(" + "|".join(_WEEKDAYS) + r")s?\b")


def named_days(utterance: str, today: date) -> dict[str, date]:
    """Every day the utterance names, as the phrase used and the date it means.

    A weekday is the next one to come, today included; "next" and "this"
    before it mean the same, which is how people use them for the coming week.
    """
    text = utterance.lower()
    found: dict[str, date] = {}
    for phrase in sorted(_RELATIVE, key=len, reverse=True):
        if re.search(rf"\b{phrase}\b", text):
            found[phrase] = today + timedelta(days=_RELATIVE[phrase])
            text = re.sub(rf"\b{phrase}\b", " ", text)
    for match in _WEEKDAY_PATTERN.finditer(text):
        weekday = _WEEKDAYS.index(match.group(1))
        found[match.group(1).capitalize()] = today + timedelta(
            days=(weekday - today.weekday()) % 7
        )
    return found


def the_one_day(utterance: str, today: date) -> date | None:
    """The single day an utterance names, or None when it names none or several."""
    days = set(named_days(utterance, today).values())
    return days.pop() if len(days) == 1 else None


def annotation(utterance: str, today: date) -> str:
    """The resolved days, as a line appended to what the model is asked."""
    days = named_days(utterance, today)
    if not days:
        return ""
    listed = "; ".join(f"{phrase} = {day.isoformat()}" for phrase, day in days.items())
    return f"(Dates named: {listed}.)"


def on_day(value: str, day: date) -> str | None:
    """An ISO-8601 local date-time moved to ``day``, its time kept.

    :returns: the corrected value, or None when it already falls on that day
        or is not a date-time at all (which the argument check will refuse).
    """
    try:
        moment = datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        return None
    if moment.date() == day:
        return None
    return moment.replace(year=day.year, month=day.month, day=day.day).isoformat(
        timespec="seconds"
    )
