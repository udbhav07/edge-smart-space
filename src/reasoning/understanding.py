"""What an utterance asked for, checked against its own words (FR-44, FR-53).

The model classifies an utterance; this module checks that classification
against the words, deterministically, the way ``dates.py`` checks the day a
model wrote. A user test with the served model (docs/user-testing/2026-10-08)
found each of these cases wrong in practice:

* **"remember that..." and "forget..." were dropped as asking for nothing**,
  so memory worked for two phrasings in six. A memory request is a service
  request; "what do you know about me" is a question.
* **Questions were acted on.** "Why is it so warm?" lowered the setpoint, and
  "what's the temperature?" was filed as a preference with nothing in it. A
  question, or a greeting, is a QUESTION, and a question never carries a
  temperature.
* **"It's a bit stuffy" warmed the room.** Where the words name a comfort
  plainly, the words win.
* **"Make it 40" was labelled cooler.** A named temperature carries no
  direction unless one was said.
* **"Move it to 4", said straight after booking a meeting, set the room to
  4 C.** A bare number with no temperature in the utterance, following an
  exchange that was not about temperature, is a follow-up to that exchange.

Every rule here can only move a request *away* from the plant: to a question,
a service request, or no direction. None of them invents a setpoint, so a rule
that misfires costs a reply, never a temperature (FR-45).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.common.schemas import Comfort, ContextTurn, Intent, PreferenceHint

#: Asking the system to keep or drop something about the speaker.
_MEMORY_REQUEST = re.compile(
    r"\b(please\s+)?(remember|memorise|memorize|note)\s+(that|this|my|i|me|i'm|im)\b"
    r"|\b(don'?t\s+)?forget\s+(that|this|what|about|my|i)\b"
    r"|^\s*(please\s+)?(remember|forget)\b",
)

#: Asking what is remembered: a question, answered from the profile.
_MEMORY_QUESTION = re.compile(
    r"\bwhat\s+(do|did|have)\s+you\s+(know|remember|remembered|kept|stored)\b"
    r"|\bwhat\s+do\s+you\s+know\s+about\s+me\b"
    r"|\bdo\s+you\s+remember\s+(anything|what)\b"
)

#: How a question opens. "How about..." is a suggestion, not a question.
_QUESTION_OPENING = re.compile(
    r"^\s*(why|what|what's|whats|how(?!\s+about)|when|where|who|which|"
    r"is|are|was|were|do|does|did|have|has|can\s+you\s+tell|tell\s+me)\b"
)

#: A greeting or small talk, which deserves an answer and asks for nothing.
_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|night)|thanks|thank\s+you|"
    r"how\s+are\s+you|bye|goodbye)\b[\s!.,?]*(there|jarvis|room)?[\s!.,?]*$"
)

#: Words that ask for a change, so an utterance opening like a question
#: ("can you...", "could you...") is still a request when one appears.
_REQUEST_VERB = re.compile(
    r"\b(make|set|turn|change|put|lower|raise|increase|decrease|reduce|bump|"
    r"cool\s+(it|the|this)?\s*down|warm\s+(it|the|this)?\s*up|heat\s+(it|the)?\s*up|"
    r"switch|adjust)\b"
)

#: Plain comfort words. Each is unambiguous about direction.
_WANTS_COOLER = re.compile(
    r"\b(stuffy|muggy|humid|sticky|sweaty|sweating|boiling|roasting|baking|"
    r"too\s+(hot|warm)|so\s+(hot|warm)|(it'?s|its|it\s+is)\s+(hot|warm)|i'?m\s+(hot|warm))\b"
)
_WANTS_WARMER = re.compile(
    r"\b(freezing|chilly|too\s+(cold|cool)|so\s+(cold|cool)|"
    r"(it'?s|its|it\s+is)\s+(cold|cool)|i'?m\s+(cold|cool))\b"
)

#: Anything that makes a number in the utterance a temperature.
_TEMPERATURE_CUE = re.compile(
    r"(°|\bdegrees?\b|\bdeg\b|\bcelsius\b|\btemp|\bwarm|\bcool|\bcold|\bhot\b|"
    r"\bheat|\bfreez|\bchill|\bthermostat|\bair\s*con|\bac\b|\ba/c\b|\broom\b)"
)

#: What a recent exchange about temperature looks like, in either voice.
_TEMPERATURE_EXCHANGE = re.compile(r"\bdegrees?\b|\btemperature\b|\bsafe\s+range\b")


#: "forget (that|about) X". "Don't forget" is the opposite and is not this.
_FORGET = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?<!don't\s)forget\s+"
    r"(?:that\s+|about\s+|what\s+i\s+said\s+about\s+)?(?P<what>.+)$",
    re.IGNORECASE,
)
#: "(please) remember (that) X", "don't forget (that) X".
_REMEMBER = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?:remember|memori[sz]e|note|don'?t\s+forget)\s+"
    r"(?:that\s+)?(?P<what>.+)$",
    re.IGNORECASE,
)
#: "X, remember that" -- the request after the fact.
_REMEMBER_AFTER = re.compile(
    r"^(?P<what>.+?)[,.;]?\s+(?:please\s+)?(?:remember|note)\s+(?:that|this|it)\s*$",
    re.IGNORECASE,
)
#: First-person openings, in the third person the profile is written in.
_THIRD_PERSON = (
    (re.compile(r"^(?:i\s+am|i'?m)\s+", re.IGNORECASE), "Is "),
    (re.compile(r"^i\s+have\s+", re.IGNORECASE), "Has "),
    (re.compile(r"^my\s+", re.IGNORECASE), "Their "),
    (re.compile(r"^i\s+(like|love|prefer|hate|want|need|eat|drink|work|live|enjoy|dislike|take)\s+", re.IGNORECASE), None),
)
_STATED = frozenset({"Is", "Has", "Likes", "Loves", "Prefers", "Hates", "Wants", "Needs", "Eats",
                     "Drinks", "Works", "Lives", "Enjoys", "Dislikes", "Takes"})


def memory_request(text: str) -> tuple[str, str] | None:
    """Which memory tool the words ask for, and with what (U-07).

    The served model, asked to remember, sometimes called ``forget``, and
    sometimes called nothing while replying "I'll remember that". The words
    are unambiguous where the model is not, so they decide.

    :returns: ``("remember", fact)`` or ``("forget", about)``, or None when
        the words do not plainly ask either.
    """
    said = " ".join(text.split()).rstrip(" .!?")
    forget = _FORGET.match(said)
    if forget and not said.lower().startswith(("don't", "dont")):
        stated = _in_third_person(forget.group("what"))
        first, _, rest = stated.partition(" ")
        # "Is vegetarian" is kept; "vegetarian" is what to look for.
        if first in _STATED and rest:
            return "forget", rest.rstrip(".")
        return "forget", forget.group("what").strip(" ,.;")
    for pattern in (_REMEMBER, _REMEMBER_AFTER):
        remember = pattern.match(said)
        if remember:
            return "remember", _in_third_person(remember.group("what"))
    return None


def _in_third_person(fact: str) -> str:
    fact = fact.strip(" ,.;")
    for pattern, replacement in _THIRD_PERSON:
        match = pattern.match(fact)
        if match:
            verb = replacement or match.group(1).lower().capitalize() + "s "
            fact = verb + fact[match.end():]
            break
    return (fact[:1].upper() + fact[1:] + ".") if fact else fact


def understood(
    hint: PreferenceHint, transcript: str, recent: Sequence[ContextTurn] = ()
) -> PreferenceHint:
    """The hint, with its classification checked against the words.

    :param recent: the conversation so far, newest last, used only to tell a
        follow-up from a request about the room.
    """
    text = transcript.strip().lower()
    if not text:
        return hint

    if _MEMORY_QUESTION.search(text):
        return _as(hint, Intent.QUESTION, "memory")
    if _MEMORY_REQUEST.search(text):
        return _as(hint, Intent.SERVICE, "memory")
    if _SMALL_TALK.match(text):
        return _as(hint, Intent.QUESTION, "conversation")
    if _QUESTION_OPENING.match(text) and not _REQUEST_VERB.search(text):
        return _as(hint, Intent.QUESTION, hint.subject or "room")

    if hint.intent is not Intent.ENVIRONMENT:
        return hint

    if hint.target_c is not None and not _TEMPERATURE_CUE.search(text):
        if recent and not _was_about_temperature(recent[-1]):
            # "Move it to 4" after a calendar entry: the number belongs to
            # the entry, and the assistant has the conversation to resolve it.
            return _as(hint, Intent.SERVICE, "follow-up")

    if hint.target_c is not None:
        # A temperature was named; a direction only if one was also said.
        return hint.model_copy(update={"comfort": _said_comfort(text) or Comfort.UNCHANGED})

    said = _said_comfort(text)
    if said is not None and said is not hint.comfort:
        return hint.model_copy(update={"comfort": said})
    if hint.comfort is Comfort.UNCHANGED:
        # A preference with nothing in it is a remark or a question.
        return _as(hint, Intent.QUESTION, hint.subject or "room")
    return hint


def _said_comfort(text: str) -> Comfort | None:
    cooler = bool(_WANTS_COOLER.search(text))
    warmer = bool(_WANTS_WARMER.search(text))
    if cooler and not warmer:
        return Comfort.COOLER
    if warmer and not cooler:
        return Comfort.WARMER
    return None


def _was_about_temperature(turn: ContextTurn) -> bool:
    return bool(
        _TEMPERATURE_EXCHANGE.search(turn.said.lower())
        or _TEMPERATURE_EXCHANGE.search(turn.replied.lower())
    )


def _as(hint: PreferenceHint, intent: Intent, subject: str) -> PreferenceHint:
    """Re-file a hint as something that carries no temperature."""
    return hint.model_copy(
        update={
            "intent": intent,
            "subject": subject,
            "comfort": Comfort.UNCHANGED,
            "target_c": None,
        }
    )
