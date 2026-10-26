"""What is waiting to reach Google, kept across restarts (FR-86).

A queue of operations in a JSON file, written atomically after every change,
so an entry made or a mail confirmed while the node is offline is sent once
Google can be reached -- after a restart too -- and never twice:

* a calendar entry is inserted under an id derived from the room's own entry
  id, so a retried insert that had in fact landed is recognised (409) rather
  than duplicated;
* operations on one event that cancel out while still waiting are folded
  together: an entry made and deleted offline sends nothing, and a move made
  before its insert went out is folded into the insert.

Each failure waits longer before the next attempt, from ``retry_min_s`` up to
``retry_max_s``; a rejection (Google said no) is dropped with its reason.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from functools import wraps
from pathlib import Path

LOGGER = logging.getLogger(__name__)

INSERT = "calendar.insert"
PATCH = "calendar.patch"
DELETE = "calendar.delete"
SEND = "gmail.send"
KINDS = frozenset({INSERT, PATCH, DELETE, SEND})


def _locked(method):
    """Run under the outbox's lock: the tools add while the sync drains."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return guarded


class OutboxFull(RuntimeError):
    """The bound on waiting operations has been reached."""


class Outbox:
    """Operations waiting for Google, oldest first."""

    def __init__(self, path: Path, max_items: int, retry_min_s: float, retry_max_s: float) -> None:
        self._path = path
        self._max = max_items
        self._retry_min_s = retry_min_s
        self._retry_max_s = retry_max_s
        self._lock = threading.RLock()
        self._items, self._next_id = self._load()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[dict, ...]:
        return tuple(dict(item) for item in self._items)

    @_locked
    def add(self, kind: str, args: dict, now: float) -> str:
        """Queue one operation. Returns its id.

        :raises OutboxFull: when the bound is reached; nothing is dropped
            silently to make room.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown operation {kind!r}")
        folded = self._fold(kind, args)
        if folded is not None:
            self._save()
            return folded
        if len(self._items) >= self._max:
            raise OutboxFull(f"{self._max} operations are already waiting for Google")
        op_id = f"op{self._next_id:06d}"
        self._next_id += 1
        self._items.append({"op_id": op_id, "kind": kind, "args": dict(args), "created_ts": now, "attempts": 0, "next_try_ts": now})
        self._save()
        return op_id

    @_locked
    def due(self, now: float) -> list[dict]:
        return [dict(item) for item in self._items if item["next_try_ts"] <= now]

    @_locked
    def contains(self, op_id: str) -> bool:
        return any(item["op_id"] == op_id for item in self._items)

    @_locked
    def done(self, op_id: str) -> None:
        self._items = [item for item in self._items if item["op_id"] != op_id]
        self._save()

    @_locked
    def retry_later(self, op_id: str, now: float) -> None:
        for item in self._items:
            if item["op_id"] == op_id:
                item["attempts"] += 1
                wait = min(self._retry_max_s, self._retry_min_s * 2 ** (item["attempts"] - 1))
                item["next_try_ts"] = now + wait
        self._save()

    @_locked
    def retry_all_now(self, now: float) -> None:
        """After signing in again, try everything at once."""
        for item in self._items:
            item["next_try_ts"] = now
        self._save()

    def _fold(self, kind: str, args: dict) -> str | None:
        """Combine with an operation on the same event still waiting, if any."""
        if kind not in (PATCH, DELETE):
            return None
        google_id = args.get("google_id")
        pending = next((item for item in self._items if item["kind"] == INSERT and item["args"].get("google_id") == google_id), None)
        if pending is None:
            return None
        if kind == DELETE:
            # Made and removed before it ever left: nothing to send at all.
            self._items.remove(pending)
            return pending["op_id"]
        pending["args"]["body"] = {**pending["args"]["body"], **args.get("body", {})}
        return pending["op_id"]

    def _load(self) -> tuple[list[dict], int]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return [], 1
        except (OSError, ValueError) as exc:
            # Refused rather than replaced: overwriting it would lose what the
            # occupant was promised would be sent.
            raise ValueError(f"the Google outbox at {self._path} is unreadable: {exc}") from exc
        items = [item for item in raw.get("items", []) if item.get("kind") in KINDS]
        return items, int(raw.get("next_id", len(items) + 1))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"next_id": self._next_id, "items": self._items}, indent=1), encoding="utf-8")
        os.replace(temporary, self._path)
