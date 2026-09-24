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
