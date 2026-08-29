"""Unit tests for coefficient persistence.

Everything unusable must come back as None, because the caller's response to
all of it is the same: start from the prior. What is worth testing hardest
is that a corrupt or stale file is *not* adopted -- restoring garbage is
worse than restoring nothing.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.common.clock import SimClock
from src.common.config import PersistenceConfig
from src.estimation.persistence import SCHEMA_VERSION, CoefficientStore

THETA = np.array([0.97, 0.03, -0.08, 0.005])
COVARIANCE = np.eye(4) * 0.25
SAMPLES = 14203
INTERVAL_S = 300.0
MAX_AGE_S = 86400.0


@pytest.fixture(name="config")
def _config(tmp_path: Path) -> PersistenceConfig:
    return PersistenceConfig(
        path=str(tmp_path / "state" / "coefficients.json"),
        interval_s=INTERVAL_S,
        max_age_s=MAX_AGE_S,
    )


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="store")
def _store(config, clock) -> CoefficientStore:
    return CoefficientStore(config, clock)


class TestRoundTrip:
    def test_an_estimate_survives_a_write_and_read(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        restored = store.load()
        assert np.allclose(restored.theta, THETA)

    def test_the_covariance_survives(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        assert np.allclose(store.load().covariance, COVARIANCE)

    def test_the_sample_count_survives(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        assert store.load().samples_since_reset == SAMPLES

    def test_the_write_time_is_recorded_from_the_injected_clock(self, store, clock):
        clock.advance(500.0)
        store.save(THETA, COVARIANCE, SAMPLES)
        assert store.load().ts == clock.now()

    def test_the_parent_directory_is_created(self, store):
        assert store.save(THETA, COVARIANCE, SAMPLES) is True
        assert store.path.exists()

    def test_the_file_is_readable_json(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        assert json.loads(store.path.read_text(encoding="utf-8"))["version"] == (
            SCHEMA_VERSION
        )

    def test_no_temporary_file_is_left_behind(self, store):
        """The write goes beside the destination and is renamed into place."""
        store.save(THETA, COVARIANCE, SAMPLES)
        assert list(store.path.parent.glob("*.tmp")) == []


class TestNothingUsable:
    def test_a_missing_file_yields_nothing(self, store):
        assert store.load() is None

    def test_unparseable_content_yields_nothing(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text("{not json", encoding="utf-8")
        assert store.load() is None

    def test_a_non_object_document_yields_nothing(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text("[1, 2, 3]", encoding="utf-8")
        assert store.load() is None

    def test_a_different_schema_version_yields_nothing(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        payload["version"] = SCHEMA_VERSION + 1
        store.path.write_text(json.dumps(payload), encoding="utf-8")
        assert store.load() is None

    def test_a_missing_field_yields_nothing(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        del payload["theta"]
        store.path.write_text(json.dumps(payload), encoding="utf-8")
        assert store.load() is None

    def test_a_wrongly_shaped_theta_yields_nothing(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        payload["theta"] = [0.9, 0.1]
        store.path.write_text(json.dumps(payload), encoding="utf-8")
        assert store.load() is None

    def test_a_wrongly_shaped_covariance_yields_nothing(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        payload["covariance"] = [[1.0, 0.0], [0.0, 1.0]]
        store.path.write_text(json.dumps(payload), encoding="utf-8")
        assert store.load() is None

    def test_a_non_numeric_theta_yields_nothing(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        payload["theta"] = ["warm", "cool", "x", "y"]
        store.path.write_text(json.dumps(payload), encoding="utf-8")
        assert store.load() is None


class TestStaleness:
    """Section 7.1: older than a day and the estimate is discarded."""

    def test_a_fresh_estimate_is_adopted(self, store, clock):
        store.save(THETA, COVARIANCE, SAMPLES)
        clock.advance(MAX_AGE_S / 2.0)
        assert store.load() is not None

    def test_an_estimate_past_the_horizon_is_discarded(self, store, clock):
        store.save(THETA, COVARIANCE, SAMPLES)
        clock.advance(MAX_AGE_S + 1.0)
        assert store.load() is None

    def test_the_horizon_itself_is_still_adopted(self, store, clock):
        store.save(THETA, COVARIANCE, SAMPLES)
        clock.advance(MAX_AGE_S)
        assert store.load() is not None


class TestWriteCadence:
    def test_the_first_write_is_always_due(self, store):
        assert store.is_due() is True

    def test_a_write_is_not_due_again_immediately(self, store):
        store.save(THETA, COVARIANCE, SAMPLES)
        assert store.is_due() is False

    def test_a_write_is_due_once_the_interval_elapses(self, store, clock):
        store.save(THETA, COVARIANCE, SAMPLES)
        clock.advance(INTERVAL_S)
        assert store.is_due() is True

    def test_the_last_write_time_is_reported(self, store, clock):
        clock.advance(42.0)
        store.save(THETA, COVARIANCE, SAMPLES)
        assert store.last_written_ts == clock.now()


class TestFailureIsReportedNotRaised:
    def test_an_unwritable_destination_returns_false(self, clock, tmp_path):
        """Losing a snapshot must not stop the estimator, which still holds
        the live estimate in memory."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        config = PersistenceConfig(
            path=str(blocker / "coefficients.json"),
            interval_s=INTERVAL_S,
            max_age_s=MAX_AGE_S,
        )
        assert CoefficientStore(config, clock).save(THETA, COVARIANCE, SAMPLES) is False

    def test_a_failed_write_does_not_advance_the_cadence(self, clock, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        config = PersistenceConfig(
            path=str(blocker / "coefficients.json"),
            interval_s=INTERVAL_S,
            max_age_s=MAX_AGE_S,
        )
        store = CoefficientStore(config, clock)
        store.save(THETA, COVARIANCE, SAMPLES)
        assert store.is_due() is True
