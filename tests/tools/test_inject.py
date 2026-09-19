"""Unit tests for the injection command line (FR-31).

An examiner types these under pressure, so the tests are mostly about what
happens when the typing is wrong: a missing magnitude, one that would be
ignored, or a sensor that is not there.

Argument handling and validation are exercised directly. The publishing path
is covered where the injector lives; what is checked here is that nothing is
published when the request does not make sense.
"""

from pathlib import Path

import pytest

from src.common.config import load_config
from src.common.injection import InjectedFault
from tools.inject import FAULT_NAMES, build_parser, main


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


def _parse(*argv):
    return build_parser().parse_args(list(argv))


class TestFaultNames:
    def test_every_injectable_fault_has_a_short_name(self):
        assert set(FAULT_NAMES.values()) == set(InjectedFault)

    def test_ending_a_fault_is_spelled_clear_not_none(self):
        """What the operator is doing is ending a fault, not injecting an
        absence."""
        assert FAULT_NAMES["clear"] is InjectedFault.NONE

    def test_the_short_names_are_the_accepted_choices(self):
        arguments = _parse("temp_01", "stuck", "27.0")
        assert arguments.fault == "stuck"

    def test_an_unknown_fault_name_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            _parse("temp_01", "melted")


class TestArguments:
    def test_a_magnitude_is_read_as_a_number(self):
        assert _parse("temp_01", "stuck", "27.0").magnitude == 27.0

    def test_a_magnitude_is_optional(self):
        assert _parse("temp_01", "dropout").magnitude is None

    def test_a_negative_drift_rate_is_accepted(self):
        """A sensor can drift downwards."""
        assert _parse("temp_01", "drift", "-0.01").magnitude == -0.01

    def test_the_requester_defaults_to_the_operator(self):
        assert _parse("temp_01", "dropout").requester == "operator"

    def test_listing_needs_no_subject(self):
        assert _parse("--list").list is True


class TestRefusals:
    """Nothing is published when the request cannot take effect."""

    def test_a_fault_needing_a_magnitude_without_one_exits_two(self, capsys):
        assert main(["temp_01", "stuck"]) == 2

    def test_a_magnitude_that_would_be_ignored_exits_two(self):
        assert main(["temp_01", "dropout", "27.0"]) == 2

    def test_a_missing_fault_exits_two(self):
        assert main(["temp_01"]) == 2

    def test_a_missing_subject_exits_two(self):
        assert main([]) == 2

    def test_a_missing_config_file_exits_two(self):
        assert main(["--config", "config/nope.yaml", "temp_01", "dropout"]) == 2


class TestListing:
    def test_listing_exits_cleanly_without_a_broker(self):
        """It reads config and prints; it never connects."""
        assert main(["--list"]) == 0

    def test_listing_names_every_configured_sensor(self, capsys, config):
        main(["--list"])
        printed = capsys.readouterr().out
        for sensor in config.sensors.adapters:
            assert sensor.sensor_id in printed

    def test_listing_names_every_injectable_fault(self, capsys):
        main(["--list"])
        printed = capsys.readouterr().out
        for name in FAULT_NAMES:
            assert name in printed

    def test_listing_says_what_each_magnitude_means(self, capsys):
        """A number with no unit is how the wrong one gets typed."""
        main(["--list"])
        assert "units per second" in capsys.readouterr().out
