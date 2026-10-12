"""See and manage what the room keeps about each person (FR-52, NFR-05).

    python -m tools.profiles list
    python -m tools.profiles create ravi --name Ravi   # no voice needed yet
    python -m tools.profiles show ravi
    python -m tools.profiles remember ravi "prefers meetings after ten"
    python -m tools.profiles forget ravi meetings
    python -m tools.profiles prefer ravi comfort_c 23
    python -m tools.profiles clear-recent ravi       # end the conversation now
    python -m tools.profiles delete ravi --yes       # forget them entirely
    python -m tools.profiles prune                   # expired context, orphaned files

Three kinds of thing are kept, each in a file a person can open:

* long-term memory, ``profiles.directory/<id>.md`` -- edit it by hand freely;
* short-term context, ``profiles.context_directory/<id>.json`` -- expires;
* the voiceprint, in ``speech.voiceprints_path`` -- feature statistics only.

This works on the files directly, so it needs no broker and no running
service. The assistance service republishes every profile on its interval,
so a change made here reaches the blackboard within
``profiles.publish_interval_s`` -- and a deleted person is withdrawn from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.assistance.__main__ import build_store
from src.assistance.memory import ProfileError, ProfileStore
from src.common.clock import RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.schemas import GUEST_PROFILE_ID
from src.speech.speaker_profile import SpeakerIdentifier

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class Profiles:
    """The management operations, over the store and the voiceprints."""

    def __init__(self, config: Config, store: ProfileStore, voices: SpeakerIdentifier) -> None:
        self._config = config
        self._store = store
        self._voices = voices

    def list(self) -> list[str]:
        enrolled = self._voices.enrolled
        ids = self._store.profile_ids()
        if not ids:
            return ["nobody is enrolled"]
        lines = []
        for profile_id in ids:
            profile = self._store.load(profile_id)
            utterances = enrolled.get(profile_id, 0)
            voice = (
                f"voice: {utterances} utterance(s)"
                if self._voices.matchable(profile_id)
                else f"voice: {utterances} utterance(s), not yet recognisable"
            )
            lines.append(
                f"{profile_id:12s} {profile.display_name:16s} {len(profile.facts):3d} fact(s)  "
                f"{len(self._store.recent(profile_id)):2d} recent  {voice}"
            )
        guest = len(self._store.recent(GUEST_PROFILE_ID))
        lines.append(f"{GUEST_PROFILE_ID:12s} {'(unrecognised)':16s}   -        {guest:2d} recent")
        return lines

    def show(self, profile_id: str) -> list[str]:
        context = self._store.context(profile_id)
        lines = [f"{context.display_name or '(guest)'} [{profile_id}]"]
        if profile_id != GUEST_PROFILE_ID:
            lines.append(f"  long-term:  {Path(self._config.profiles.directory) / (profile_id + '.md')}")
            lines.append(f"  voiceprint: {self._voices.enrolled.get(profile_id, 0)} utterance(s)")
        lines.append("  preferences:" if context.preferences else "  preferences: none")
        lines.extend(f"    {key}: {value}" for key, value in context.preferences.items())
        lines.append("  facts:" if context.facts else "  facts: none")
        lines.extend(f"    - {fact}" for fact in context.facts)
        if profile_id != GUEST_PROFILE_ID:
            unread = self._store.load(profile_id).unread
            if unread:
                lines.append("  not understood (dropped at the next change; fix the file):")
                lines.extend(f"    ? {line}" for line in unread)
        lines.append("  recent:" if context.recent else "  recent: none")
        lines.extend(f"    > {turn.said}\n      < {turn.replied}" for turn in context.recent)
        return lines

    def create(self, profile_id: str, display_name: str) -> list[str]:
        """A profile with no voice yet (U-24): memory first, enrolment later."""
        self._store.create(profile_id, display_name or profile_id)
        return [
            f"created {profile_id}; enrol their voice with "
            f"python -m tools.enroll {profile_id} --record 3"
        ]

    def remember(self, profile_id: str, fact: str) -> list[str]:
        _, added = self._store.remember(profile_id, fact)
        return ["kept" if added else "already kept"]

    def forget(self, profile_id: str, about: str) -> list[str]:
        _, removed = self._store.forget(profile_id, about)
        return [f"forgot: {fact}" for fact in removed] or [f"nothing mentioned {about!r}"]

    def prefer(self, profile_id: str, key: str, value: str) -> list[str]:
        self._store.set_preference(profile_id, key, value)
        return [f"{key}: {value}"]

    def clear_recent(self, profile_id: str) -> list[str]:
        return ["conversation ended" if self._store.clear_recent(profile_id) else "no conversation to end"]

    def delete(self, profile_id: str) -> list[str]:
        """Everything kept about a person: memory, context and voiceprint."""
        memory = self._store.delete(profile_id)
        voice = self._voices.remove(profile_id)
        if not (memory or voice):
            return [f"nothing was kept about {profile_id}"]
        return [f"deleted everything kept about {profile_id}"]

    def prune(self) -> list[str]:
        """Expired short-term files, and anything kept for nobody."""
        lines = []
        known = {*self._store.profile_ids(), GUEST_PROFILE_ID}
        context_directory = Path(self._config.profiles.context_directory)
        for path in sorted(context_directory.glob("*.json")) if context_directory.is_dir() else ():
            if path.stem not in known:
                path.unlink()
                lines.append(f"removed context kept for nobody: {path.name}")
            elif not self._store.recent(path.stem):
                path.unlink()
                lines.append(f"removed expired context: {path.name}")
        for profile_id in self._voices.enrolled:
            if profile_id not in known:
                self._voices.remove(profile_id)
                lines.append(f"removed a voiceprint with no profile: {profile_id}")
        return lines or ["nothing to prune"]


def build(config: Config) -> Profiles:
    return Profiles(config, build_store(config, RealClock()), SpeakerIdentifier(config.speech))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.profiles", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("prune")
    for name in ("show", "clear-recent"):
        commands.add_parser(name).add_argument("profile_id")
    create = commands.add_parser("create")
    create.add_argument("profile_id")
    create.add_argument("--name", default="")
    remember = commands.add_parser("remember")
    remember.add_argument("profile_id")
    remember.add_argument("fact")
    forget = commands.add_parser("forget")
    forget.add_argument("profile_id")
    forget.add_argument("about")
    prefer = commands.add_parser("prefer")
    prefer.add_argument("profile_id")
    prefer.add_argument("key")
    prefer.add_argument("value")
    delete = commands.add_parser("delete")
    delete.add_argument("profile_id")
    delete.add_argument("--yes", action="store_true", help="Delete without asking")
    arguments = parser.parse_args(argv)

    try:
        profiles = build(load_config(arguments.config))
        command = arguments.command
        if command == "list":
            lines = profiles.list()
        elif command == "prune":
            lines = profiles.prune()
        elif command == "create":
            lines = profiles.create(arguments.profile_id, arguments.name)
        elif command == "show":
            lines = profiles.show(arguments.profile_id)
        elif command == "clear-recent":
            lines = profiles.clear_recent(arguments.profile_id)
        elif command == "remember":
            lines = profiles.remember(arguments.profile_id, arguments.fact)
        elif command == "forget":
            lines = profiles.forget(arguments.profile_id, arguments.about)
        elif command == "prefer":
            lines = profiles.prefer(arguments.profile_id, arguments.key, arguments.value)
        else:
            if not arguments.yes and input(f"delete everything kept about {arguments.profile_id}? [y/N] ").strip().lower() != "y":
                print("kept")
                return 1
            lines = profiles.delete(arguments.profile_id)
    except (ConfigError, ProfileError) as exc:
        print(f"error: {exc}")
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
