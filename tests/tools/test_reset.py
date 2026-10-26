"""Unit tests for the reset command line (section 5.6).

Argument handling and the refusal path. What the reset *does* is tested where
the mode manager and the detector bank live; what matters here is that a reset
with nobody's name on it is refused rather than published.
"""

from pathlib import Path

import pytest

from src.common.config import load_config
from tools.reset import build_parser, main


def _parse(*argv):
    return build_parser().parse_args(list(argv))


class TestArguments:
    def test_the_requester_defaults_to_the_operator(self):
        assert _parse().requester == "operator"

    def test_a_named_requester_is_carried(self):
        assert _parse("--requester", "udbhav").requester == "udbhav"

    def test_a_reason_is_optional(self):
        assert _parse().reason == ""

    def test_a_reason_is_kept_verbatim(self):
        parsed = _parse("--reason", "replaced the indoor sensor")
        assert parsed.reason == "replaced the indoor sensor"

    def test_the_config_path_can_be_overridden(self):
        assert _parse("--config", "other.yaml").config == Path("other.yaml")


class TestRefusals:
    def test_an_anonymous_reset_is_refused(self):
        """A hold cleared with no record of who cleared it is an audit trail
        with a hole exactly where the interesting thing happened."""
        assert main(["--requester", ""]) == 2

    def test_a_missing_config_file_exits_two(self):
        assert main(["--config", "config/nope.yaml"]) == 2


class TestHelpText:
    def test_the_help_says_it_re_tests_rather_than_overrides(self):
        """An operator who thinks this silences a fault will use it to silence
        a fault."""
        assert "re-tests" in build_parser().epilog

    def test_the_configured_broker_is_the_one_it_would_reach(self):
        config = load_config(Path("config/default.yaml"))
        assert config.mqtt.host and config.mqtt.port


class TestOutcome:
    """U-18: the reset says what happened, and what it does not promise."""

    @staticmethod
    def _mode(ts, mode="NORMAL", faults=()):
        from src.common.schemas import ModeState
        return ModeState(ts=ts, mode=mode, since_ts=ts, active_fault_ids=faults, reason="all clear")

    @staticmethod
    def _reset(ts=100.0):
        from src.common.schemas import ModeReset
        return ModeReset(ts=ts, requester="operator")

    def test_the_mode_the_reset_produced_is_reported(self):
        from tools.reset import describe_outcome
        text = describe_outcome(self._reset(), [self._mode(50.0, "SAFE_HOLD"), self._mode(101.0)])
        assert text.startswith("mode is now NORMAL")

    def test_a_reset_that_changed_nothing_says_the_mode_stays(self):
        from tools.reset import describe_outcome
        assert describe_outcome(self._reset(), [self._mode(50.0)]).startswith("mode stays NORMAL")

    def test_it_says_faults_are_re_tested(self):
        from tools.reset import describe_outcome
        assert "re-tested" in describe_outcome(self._reset(), [self._mode(101.0)])

    def test_faults_still_active_are_named(self):
        from tools.reset import describe_outcome
        text = describe_outcome(self._reset(), [self._mode(101.0, "DEGRADED_SENSOR", ("d2:temp_01",))])
        assert "still active: d2:temp_01" in text

    def test_silence_points_at_the_detectors_service(self):
        from tools.reset import describe_outcome
        assert "detectors service" in describe_outcome(self._reset(), [])
