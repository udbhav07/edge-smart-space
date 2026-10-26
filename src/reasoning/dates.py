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
from dataclasses import dataclass
from datetime import date, datetime, timedelta

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_RELATIVE = {
    "day after tomorrow": 2,
    "tomorrow": 1,
    "yesterday": -1,
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


#: "in 10 minutes", "in an hour", "in half an hour", "in 2 hrs".
_FROM_NOW = re.compile(
    r"\bin\s+(?P<count>an?|one|two|three|half\s+an|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>minutes?|mins?|hours?|hrs?)\b"
)
_COUNT_WORDS = {"a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0}


def from_now(utterance: str) -> timedelta | None:
    """How far ahead "in N minutes" puts something, or None if it says no such thing.

    The model read "remind me to call mom in 10 minutes" as a ten-minute
    entry starting now, which had already passed by the time it was checked.
    """
    match = _FROM_NOW.search(utterance.lower())
    if match is None:
        return None
    count, unit = match.group("count"), match.group("unit")
    if count.startswith("half"):
        return timedelta(minutes=30)
    amount = _COUNT_WORDS.get(count) or float(count)
    return timedelta(hours=amount) if unit.startswith(("hour", "hr")) else timedelta(minutes=amount)


_HOUR_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "noon": 12, "midday": 12,
}
_HOUR = r"(?P<hour>\d{1,2}|" + "|".join(_HOUR_WORDS) + r")"
#: The ways a time of day is said, most specific first. Each names ``hour``
#: and may name ``minute`` (or be half / quarter) and ``period``.
_TIMES = (
    (re.compile(r"\b(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\s*(?P<period>am|pm|a\.m\.|p\.m\.)?"), None),
    (re.compile(r"\bhalf\s+past\s+" + _HOUR + r"\b"), 30),
    (re.compile(r"\bquarter\s+past\s+" + _HOUR + r"\b"), 15),
    (re.compile(r"\bquarter\s+to\s+" + _HOUR + r"\b"), -15),
    (re.compile(r"\b" + _HOUR + r"\s+thirty\b"), 30),
    (re.compile(r"\b" + _HOUR + r"\s*(?P<period>am|pm|a\.m\.|p\.m\.|o'?clock)\b"), 0),
    (re.compile(r"\b(?:at|to|for|by|until)\s+" + _HOUR + r"\b"), 0),
)


@dataclass(frozen=True)
class TimeSaid:
    """A time of day as said: an hour on the 12-hour clock unless a period or
    a 24-hour figure says which half of the day."""

    hour: int
    minute: int
    period: str | None  # "am", "pm", or None when the words do not say

    def nearest_to(self, hour: int) -> tuple[int, int]:
        """On the 24-hour clock: as said, or the half of the day nearer ``hour``."""
        if self.period == "pm":
            return (self.hour % 12) + 12, self.minute
        if self.period == "am" or self.hour > 12:
            return self.hour % 12 if self.period == "am" else self.hour, self.minute
        candidates = (self.hour % 12, (self.hour % 12) + 12)
        return min(candidates, key=lambda h: abs(h - hour)), self.minute


def time_said(utterance: str) -> TimeSaid | None:
    """The time of day the words name -- "half past two", "at four", "10:30",
    "6 pm" -- or None when they name none (S-03).

    Live, "actually make it half past two" was read as wanting the room
    warmer. A time said is arithmetic on the words, so it is read here.
    """
    text = utterance.lower()
    for pattern, minute in _TIMES:
        match = pattern.search(text)
        if match is None:
            continue
        raw = match.group("hour")
        hour = _HOUR_WORDS.get(raw) or int(raw)
        if hour > 23:
            continue
        found = minute
        if minute is None:
            found = int(match.group("minute"))
        period_word = match.groupdict().get("period") or ""
        period = "pm" if period_word.startswith("p") else "am" if period_word.startswith("a") else None
        if found == -15:
            hour, found = (hour - 1) % 12 or 12, 45
        return TimeSaid(hour=hour, minute=found, period=period)
    return None


def annotation(utterance: str, today: date, now: datetime | None = None) -> str:
    """The resolved days and times, as a line appended to what the model is asked."""
    parts = [
        f"{phrase} = {day.isoformat()}" for phrase, day in named_days(utterance, today).items()
    ]
    ahead = from_now(utterance)
    if ahead is not None and now is not None:
        parts.append(f"{_FROM_NOW.search(utterance.lower()).group(0)} = {(now + ahead).isoformat(timespec='minutes')}")
    if not parts:
        return ""
    return f"(Dates named: {'; '.join(parts)}.)"


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
