"""What the room remembers about each person (FR-52), kept on this node.

Two kinds of memory, kept differently because they are used differently:

* **Long-term memory is a Markdown file per person**, ``<profile_id>.md``:
  a name, named preferences, and facts the person asked to be remembered. It
  is a document a person can read, correct and delete without any tool --
  which is the point. Memory about someone should be legible to them, and a
  binary store or a database row is not. The file is parsed tolerantly, so a
  hand edit that adds a section of its own keeps it on the next write.
* **Short-term context is a JSON file per person**, ``<profile_id>.json``:
  the last few exchanges, so "move it to four" means something after "put
  the review in at three". It expires (``profiles.short_term_ttl_s``) and is
  bounded (``short_term_max_turns``), because a conversation is not a record.
  A guest -- a voice nobody enrolled -- has short-term context and never
  long-term memory.

Knowing who is speaking personalises replies and grants nothing: no profile
unlocks a tool, a setpoint or a booking (FR-52).

The store holds no state between calls: every read goes to disk. A hand edit
is therefore live at once, and two processes reading the same directory --
the executor and a management tool -- cannot disagree about what it holds.
Writes are atomic, as for the calendar: a crash mid-write leaves the previous
file, not half of a new one.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from src.common.clock import Clock
from src.common.config import ProfilesConfig
from src.common.localtime import local_time
from src.common.schemas import (
    GUEST_PROFILE_ID,
    PROFILE_ID_PATTERN,
    ContextTurn,
    ProfileContext,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_ENCODING = "utf-8"
_TEMPORARY_SUFFIX = ".tmp"
_MARKDOWN_SUFFIX = ".md"
_JSON_SUFFIX = ".json"

_PREFERENCES = "Preferences"
_FACTS = "Facts"
_ID_KEY = "profile_id"
_CREATED_KEY = "created"

_PROFILE_ID = re.compile(PROFILE_ID_PATTERN)
_BULLET = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
_KEY_VALUE = re.compile(r"^([A-Za-z_][\w ]*?)\s*:\s*(.*)$")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

#: Written into every profile so a person opening the file knows what it is.
_EXPLANATION = (
    "<!-- What the smart space remembers about this person (FR-52). Edit it\n"
    "     freely: the facts below are what the assistant is told. Recognising\n"
    "     this voice personalises replies only; it unlocks nothing. Delete the\n"
    "     file, or run `python -m tools.profiles delete`, to forget them. -->"
)


class ProfileError(ValueError):
    """A profile request that cannot be honoured, and why."""


class ProfileNotFoundError(ProfileError):
    """No profile by that id."""


class ProfileLimitError(ProfileError):
    """A configured bound was reached: profiles, facts, or a fact's length."""


@dataclass(frozen=True)
class Profile:
    """One person's long-term memory, as the file holds it."""

    profile_id: str
    display_name: str
    created: str = ""
    preferences: Mapping[str, str] = field(default_factory=dict)
    facts: tuple[str, ...] = ()
    #: Sections a person added by hand, kept verbatim so a write never
    #: deletes what somebody else wrote.
    other_sections: str = ""
    #: Lines the parser could not place, shown by ``tools.profiles show``
    #: rather than dropped without a word (U-15). Not kept on the next write.
    unread: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "preferences", MappingProxyType(dict(self.preferences)))


def valid_profile_id(profile_id: str) -> bool:
    """Whether an id is safe as a file name and a topic level."""
    return bool(_PROFILE_ID.match(profile_id))


def clean_text(text: str, limit: int) -> str:
    """One line of plain text, at most ``limit`` characters.

    Newlines and Markdown structure are removed: a fact that carried
    ``\\n## Facts`` would otherwise rewrite the file it is stored in.

    :raises ProfileLimitError: if nothing is left, or it is too long.
    """
    line = " ".join(text.split())
    line = line.lstrip("#*-> ").strip()
    if not line:
        raise ProfileLimitError("there is nothing to keep")
    if len(line) > limit:
        raise ProfileLimitError(f"that is {len(line)} characters; at most {limit} are kept")
    return line


class ProfileStore:
    """Long-term Markdown and short-term JSON, one file of each per person."""

    def __init__(self, config: ProfilesConfig, clock: Clock, utc_offset_h: float) -> None:
        self._config = config
        self._clock = clock
        self._utc_offset_h = utc_offset_h
        self._directory = Path(config.directory)
        self._context_directory = Path(config.context_directory)

    # --- long-term ----------------------------------------------------

    def profile_ids(self) -> tuple[str, ...]:
        """Every person with a profile, by id. Files with unsafe names are skipped."""
        if not self._directory.is_dir():
            return ()
        ids = []
        for path in sorted(self._directory.glob(f"*{_MARKDOWN_SUFFIX}")):
            if valid_profile_id(path.stem) and path.stem != GUEST_PROFILE_ID:
                ids.append(path.stem)
            else:
                LOGGER.warning("ignoring %s: not a valid profile id", path.name)
        return tuple(ids)

    def exists(self, profile_id: str) -> bool:
        return valid_profile_id(profile_id) and self._long_term_path(profile_id).is_file()

    def load(self, profile_id: str) -> Profile:
        """One person's long-term memory.

        :raises ProfileNotFoundError: if there is no such profile.
        """
        self._require_id(profile_id)
        path = self._long_term_path(profile_id)
        try:
            text = path.read_text(encoding=_ENCODING)
        except FileNotFoundError as exc:
            raise ProfileNotFoundError(f"nobody is enrolled as {profile_id!r}") from exc
        return parse_profile(profile_id, text)

    def create(self, profile_id: str, display_name: str) -> Profile:
        """Start long-term memory for a new person.

        :raises ProfileError: for an unsafe id, an existing profile, the
            guest id, or a full room.
        """
        self._require_id(profile_id)
        if profile_id == GUEST_PROFILE_ID:
            raise ProfileError(f"{GUEST_PROFILE_ID!r} is whoever was not recognised")
        if self.exists(profile_id):
            raise ProfileError(f"{profile_id!r} is already enrolled")
        if len(self.profile_ids()) >= self._config.max_profiles:
            raise ProfileLimitError(
                f"the room keeps memory for {self._config.max_profiles} people at most"
            )
        name = clean_text(display_name or profile_id, self._config.max_fact_chars)
        created = local_time(self._clock.now(), self._utc_offset_h).date().isoformat()
        profile = Profile(profile_id=profile_id, display_name=name, created=created)
        self._save(profile)
        LOGGER.info("created the profile for %s", profile_id)
        return profile

    def remember(self, profile_id: str, fact: str) -> tuple[Profile, bool]:
        """Keep a fact. Returns the profile and whether it was new.

        :raises ProfileError: no such profile, an empty or too-long fact, or
            memory full. Full is refused rather than making room by dropping
            the oldest: forgetting what someone asked to be remembered,
            silently, is worse than saying no.
        """
        profile = self.load(profile_id)
        line = clean_text(fact, self._config.max_fact_chars)
        if line.casefold() in (existing.casefold() for existing in profile.facts):
            return profile, False
        if len(profile.facts) >= self._config.max_facts:
            raise ProfileLimitError(
                f"{self._config.max_facts} facts are already kept; forget one first"
            )
        updated = _replaced(profile, facts=(*profile.facts, line))
        self._save(updated)
        return updated, True

    def forget(self, profile_id: str, about: str) -> tuple[Profile, tuple[str, ...]]:
        """Remove every fact mentioning ``about``. Returns what was removed."""
        profile = self.load(profile_id)
        needle = " ".join(about.split()).casefold()
        if not needle:
            raise ProfileError("say what to forget")
        removed = tuple(fact for fact in profile.facts if needle in fact.casefold())
        if removed:
            kept = tuple(fact for fact in profile.facts if fact not in removed)
            profile = _replaced(profile, facts=kept)
            self._save(profile)
        return profile, removed

    def set_preference(self, profile_id: str, key: str, value: str) -> Profile:
        """Name a preference, such as ``comfort_c: 23``."""
        profile = self.load(profile_id)
        name = clean_text(key, self._config.max_fact_chars).replace(" ", "_").lower()
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
            raise ProfileError(f"{key!r} is not a preference name")
        preferences = dict(profile.preferences)
        preferences[name] = clean_text(value, self._config.max_fact_chars)
        updated = _replaced(profile, preferences=preferences)
        self._save(updated)
        return updated

    def delete(self, profile_id: str) -> bool:
        """Forget a person entirely: long-term memory and short-term context.

        :returns: whether there was anything to delete.
        """
        self._require_id(profile_id)
        removed = False
        for path in (self._long_term_path(profile_id), self._short_term_path(profile_id)):
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                continue
        if removed:
            LOGGER.info("deleted everything kept about %s", profile_id)
        return removed

    # --- short-term ---------------------------------------------------

    def record_turn(self, profile_id: str, said: str, replied: str) -> tuple[ContextTurn, ...]:
        """Add one exchange to a person's short-term context, or the guest's."""
        self._require_short_term(profile_id)
        limit = self._config.max_turn_chars
        turn = ContextTurn(
            ts=self._clock.now(),
            said=" ".join(said.split())[:limit],
            replied=" ".join(replied.split())[:limit],
        )
        turns = (*self.recent(profile_id), turn)[-self._config.short_term_max_turns:]
        self._write_json(
            self._short_term_path(profile_id),
            {
                "schema_version": SCHEMA_VERSION,
                _ID_KEY: profile_id,
                "turns": [item.model_dump() for item in turns],
            },
        )
        return turns

    def recent(self, profile_id: str) -> tuple[ContextTurn, ...]:
        """The exchanges still within the short-term window, oldest first.

        A short-term file that cannot be read is logged and treated as empty:
        losing the last few minutes of conversation costs a follow-up
        question, and refusing to answer over it would cost more.
        """
        self._require_short_term(profile_id)
        path = self._short_term_path(profile_id)
        try:
            raw = json.loads(path.read_text(encoding=_ENCODING))
            turns = [ContextTurn.model_validate(item) for item in raw["turns"]]
        except FileNotFoundError:
            return ()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOGGER.warning("short-term context for %s unreadable, starting afresh: %s", profile_id, exc)
            return ()
        # Every unrecognised voice shares the guest's context, so it is kept
        # only as long as one conversation lasts (U-20).
        ttl_s = (
            self._config.guest_short_term_ttl_s
            if profile_id == GUEST_PROFILE_ID
            else self._config.short_term_ttl_s
        )
        horizon = self._clock.now() - ttl_s
        fresh = [turn for turn in turns if turn.ts >= horizon]
        return tuple(fresh[-self._config.short_term_max_turns:])

    def clear_recent(self, profile_id: str) -> bool:
        """End a conversation now rather than when it expires."""
        self._require_short_term(profile_id)
        try:
            self._short_term_path(profile_id).unlink()
            return True
        except FileNotFoundError:
            return False

    # --- what the assistant is told ------------------------------------

    def context(self, profile_id: str) -> ProfileContext:
        """Everything known about one person, as the blackboard carries it."""
        recent = self.recent(profile_id)
        if profile_id == GUEST_PROFILE_ID:
            return ProfileContext(ts=self._clock.now(), profile_id=profile_id, recent=recent)
        profile = self.load(profile_id)
        return ProfileContext(
            ts=self._clock.now(),
            profile_id=profile_id,
            display_name=profile.display_name,
            preferences=profile.preferences,
            facts=profile.facts,
            recent=recent,
        )

    # --- files ----------------------------------------------------------

    def _long_term_path(self, profile_id: str) -> Path:
        return self._directory / f"{profile_id}{_MARKDOWN_SUFFIX}"

    def _short_term_path(self, profile_id: str) -> Path:
        return self._context_directory / f"{profile_id}{_JSON_SUFFIX}"

    @staticmethod
    def _require_id(profile_id: str) -> None:
        if not valid_profile_id(profile_id):
            raise ProfileError(f"{profile_id!r} is not a valid profile id")

    def _require_short_term(self, profile_id: str) -> None:
        self._require_id(profile_id)
        if profile_id != GUEST_PROFILE_ID and not self.exists(profile_id):
            raise ProfileNotFoundError(f"nobody is enrolled as {profile_id!r}")

    def _save(self, profile: Profile) -> None:
        self._write_text(self._long_term_path(profile.profile_id), render_profile(profile))

    def _write_json(self, path: Path, payload: dict) -> None:
        self._write_text(path, json.dumps(payload, indent=2))

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + _TEMPORARY_SUFFIX)
        temporary.write_text(text, encoding=_ENCODING)
        os.replace(temporary, path)


def _replaced(profile: Profile, **changes) -> Profile:
    fields = {
        "profile_id": profile.profile_id,
        "display_name": profile.display_name,
        "created": profile.created,
        "preferences": profile.preferences,
        "facts": profile.facts,
        "other_sections": profile.other_sections,
    }
    fields.update(changes)
    return Profile(**fields)


# --- the Markdown form -------------------------------------------------------


def render_profile(profile: Profile) -> str:
    """A profile as the Markdown file a person can read and edit."""
    lines = [
        f"# {profile.display_name}",
        "",
        _EXPLANATION,
        "",
        f"- {_ID_KEY}: {profile.profile_id}",
    ]
    if profile.created:
        lines.append(f"- {_CREATED_KEY}: {profile.created}")
    lines += ["", f"## {_PREFERENCES}", ""]
    lines += [f"- {key}: {value}" for key, value in profile.preferences.items()] or [
        "_None yet. Add lines such as `- comfort_c: 23`._"
    ]
    lines += ["", f"## {_FACTS}", ""]
    lines += [f"- {fact}" for fact in profile.facts] or ["_Nothing remembered yet._"]
    text = "\n".join(lines) + "\n"
    if profile.other_sections.strip():
        text += "\n" + profile.other_sections.strip() + "\n"
    return text


def parse_profile(profile_id: str, text: str) -> Profile:
    """Read a profile back, tolerantly.

    The file name is the id; an id written inside that disagrees is reported
    and the name wins, so renaming a file cannot leave two people with one
    identity. Unknown sections are kept verbatim.

    "Edit it freely" has to survive a person's own way of writing (U-15):
    section headings in any case and at any level, a bare ``facts:`` line,
    ``*`` bullets, a fact with no bullet at all, and a first line that names
    the person without a ``#``. Whatever still cannot be read is not dropped
    in silence: it is listed in :attr:`Profile.unread` for the person to see.
    """
    text = _COMMENT.sub("", text)
    display_name = profile_id
    titled = False
    created = ""
    preferences: dict[str, str] = {}
    facts: list[str] = []
    other: list[str] = []
    unread: list[str] = []
    section: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            if level == 1 and section is None and not titled:
                display_name, titled = title or profile_id, True
                continue
            section = _section_named(title) or title
            if section not in _KNOWN_SECTIONS:
                other.append(line)
            continue
        if section not in (None, *_KNOWN_SECTIONS):
            other.append(line)
            continue
        if not stripped or _PLACEHOLDER.match(stripped):
            continue
        if stripped.endswith(":") and _section_named(stripped) is not None:
            section = _section_named(stripped)
            continue
        bullet = _BULLET.match(line)
        item = bullet.group(1) if bullet else stripped
        if section == _FACTS:
            facts.append(item)
            continue
        pair = _KEY_VALUE.match(item)
        if section == _PREFERENCES:
            if pair is None:
                unread.append(stripped)
            else:
                preferences[_key(pair)] = pair.group(2).strip()
            continue
        key = _key(pair) if bullet and pair else None
        if key == _ID_KEY:
            if pair.group(2).strip() != profile_id:
                LOGGER.warning("%s.md names itself %r; the file name wins", profile_id, pair.group(2).strip())
        elif key == _CREATED_KEY:
            created = pair.group(2).strip()
        elif not bullet and not titled and not facts and not preferences:
            display_name, titled = stripped, True
        else:
            unread.append(stripped)

    return Profile(
        profile_id=profile_id,
        display_name=display_name,
        created=created,
        preferences=preferences,
        facts=tuple(facts),
        other_sections="\n".join(other),
        unread=tuple(unread),
    )


_KNOWN_SECTIONS = (_PREFERENCES, _FACTS)
_SECTION_BY_NAME = {name.casefold(): name for name in _KNOWN_SECTIONS}
_PLACEHOLDER = re.compile(r"^_.*_$")


def _section_named(title: str) -> str | None:
    """The known section a heading names, in whatever case it was written."""
    return _SECTION_BY_NAME.get(title.rstrip(":").strip().casefold())


def _key(pair: re.Match) -> str:
    return pair.group(1).strip().lower().replace(" ", "_")
