"""Coefficient persistence across restarts (FR-07).

An estimate takes hours of operating data to earn. Losing it to a power cut
would mean the first hours after every restart are run on the prior, which
is exactly the period the self-calibration claim is about. So theta and P
are written periodically and read back on startup.

What is stored is the *identified* vector [a2, a3, a4] and its 3x3
covariance, not the four coefficients the model is described by. a1 is
derived on load like everywhere else, so a stored file can never disagree
with the steady-state identity.

Two rules make that safe rather than merely convenient:

* **A stale estimate is discarded.** Section 7.1 puts the horizon at a day.
  A room's thermal behaviour after a week of disuse is not what was
  identified before it, and adopting a week-old covariance would assert
  confidence the data no longer supports.
* **A write never half-happens.** The file is written beside its destination
  and renamed into place, so a crash mid-write leaves the previous estimate
  intact rather than a truncated one. Restoring garbage is worse than
  restoring nothing, because nothing falls back to a prior that is at least
  known to be plausible.

Anything unreadable, malformed or stale returns None and the caller starts
from the configured prior. That is a degradation, not an error: a missing
estimate must never stop the system coming up (FR-11).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.clock import Clock
from src.common.config import PersistenceConfig
from src.estimation.rc_model import IDENTIFIED_COUNT

LOGGER = logging.getLogger(__name__)

#: Bumped when the stored shape changes, so an old file is discarded rather
#: than misread into the current structure. Version 2 stores the three
#: identified parameters; version 1 stored four, and reading one of those as
#: the current vector would silently mean something else entirely.
SCHEMA_VERSION = 2

_ENCODING = "utf-8"
_TEMPORARY_SUFFIX = ".tmp"


@dataclass(frozen=True)
class PersistedEstimate:
    """What was on disk, once it has been checked."""

    ts: float
    theta: np.ndarray
    covariance: np.ndarray
    samples_since_reset: int


class CoefficientStore:
    """Reads and writes the estimate, and decides when a write is due."""

    def __init__(self, config: PersistenceConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._path = Path(config.path)
        self._last_written_ts: float | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def last_written_ts(self) -> float | None:
        return self._last_written_ts

    def is_due(self) -> bool:
        """Whether the configured interval has elapsed since the last write."""
        if self._last_written_ts is None:
            return True
        return self._clock.now() - self._last_written_ts >= self._config.interval_s

    def save(
        self, theta: np.ndarray, covariance: np.ndarray, samples_since_reset: int
    ) -> bool:
        """Write the estimate, atomically.

        :returns: whether the write succeeded. A failure is logged and
            reported rather than raised: losing a snapshot must not stop the
            estimator, which still holds the live estimate in memory.
        """
        payload = {
            "version": SCHEMA_VERSION,
            "ts": self._clock.now(),
            "theta": [float(value) for value in theta],
            "covariance": [[float(value) for value in row] for row in covariance],
            "samples_since_reset": int(samples_since_reset),
        }
        temporary = self._path.with_suffix(self._path.suffix + _TEMPORARY_SUFFIX)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, indent=2), encoding=_ENCODING)
            os.replace(temporary, self._path)
        except OSError as exc:
            LOGGER.warning("could not persist coefficients to %s: %s", self._path, exc)
            return False

        self._last_written_ts = payload["ts"]
        return True

    def load(self) -> PersistedEstimate | None:
        """Read the estimate back.

        :returns: the estimate, or None when there is nothing usable. Missing,
            unreadable, malformed, wrong-version and stale all return None:
            each means "start from the prior", and distinguishing them would
            only tempt a caller into treating one of them as recoverable.
        """
        raw = self._read()
        if raw is None:
            return None

        try:
            stored_ts = float(raw["ts"])
            theta = np.array(raw["theta"], dtype=float)
            covariance = np.array(raw["covariance"], dtype=float)
            samples = int(raw["samples_since_reset"])
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("persisted estimate at %s is malformed: %s", self._path, exc)
            return None

        if theta.shape != (IDENTIFIED_COUNT,) or covariance.shape != (
            IDENTIFIED_COUNT,
            IDENTIFIED_COUNT,
        ):
            LOGGER.warning("persisted estimate at %s has the wrong shape", self._path)
            return None

        age_s = self._clock.now() - stored_ts
        if age_s > self._config.max_age_s:
            LOGGER.info(
                "discarding persisted estimate: %.0f s old, horizon is %.0f s",
                age_s,
                self._config.max_age_s,
            )
            return None

        return PersistedEstimate(
            ts=stored_ts,
            theta=theta,
            covariance=covariance,
            samples_since_reset=samples,
        )

    def _read(self) -> dict | None:
        try:
            raw = json.loads(self._path.read_text(encoding=_ENCODING))
        except FileNotFoundError:
            LOGGER.info("no persisted estimate at %s; starting from the prior", self._path)
            return None
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("persisted estimate at %s is unreadable: %s", self._path, exc)
            return None

        if not isinstance(raw, dict):
            LOGGER.warning("persisted estimate at %s is not an object", self._path)
            return None
        if raw.get("version") != SCHEMA_VERSION:
            LOGGER.warning(
                "persisted estimate at %s is version %r, expected %d",
                self._path,
                raw.get("version"),
                SCHEMA_VERSION,
            )
            return None
        return raw
