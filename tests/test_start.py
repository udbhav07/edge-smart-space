"""Unit tests for the service launcher.

Nothing here starts a process or opens a socket: the selection logic, the
dependency resolution and the preflight verdict are all pure given a config
and a reachability probe, and those are what can go wrong silently.
"""

import socket
from pathlib import Path

import pytest

import start
from src.common.config import load_config

CONFIG_PATH = Path("config/default.yaml")


@pytest.fixture(name="config")
def _config():
    return load_config(CONFIG_PATH)


@pytest.fixture(name="all_reachable")
def _all_reachable(monkeypatch):
    monkeypatch.setattr(start, "is_reachable", lambda host, port, **_: True)


@pytest.fixture(name="none_reachable")
def _none_reachable(monkeypatch):
    monkeypatch.setattr(start, "is_reachable", lambda host, port, **_: False)


class TestServiceSelection:
    def test_everything_runs_by_default(self):
        assert start.select(None, None) == start.SERVICES

    def test_only_narrows_to_one(self):
        chosen = start.select(["speech"], None)
        assert [service.name for service in chosen] == ["speech"]

    def test_skip_removes_one(self):
        chosen = start.select(None, ["speech"])
        assert "speech" not in [service.name for service in chosen]

    def test_only_and_skip_compose(self):
        assert start.select(["speech", "simulator"], ["speech"]) == (
            start.SERVICES[0],
        )

    def test_an_unknown_name_is_rejected_rather_than_starting_nothing(self):
        with pytest.raises(ValueError):
            start.select(["reasoning-agent"], None)

    def test_an_unknown_skip_name_is_also_rejected(self):
        with pytest.raises(ValueError):
            start.select(None, ["typo"])

    def test_every_service_declares_a_runnable_module(self):
        for service in start.SERVICES:
            command = service.command(CONFIG_PATH)
            assert command[1] == "-m"
            assert command[2] == service.module


class TestDependencies:
    def test_the_broker_is_taken_from_configuration(self, config):
        broker = dependencies_by_name(config)["mqtt broker"]
        assert (broker.host, broker.port) == (config.mqtt.host, config.mqtt.port)

    def test_the_inference_endpoint_is_parsed_from_its_url(self, config):
        server = dependencies_by_name(config)["inference server"]
        assert server.port == 11434

    def test_the_broker_is_required(self, config):
        """Every component reaches the others only through the blackboard."""
        assert dependencies_by_name(config)["mqtt broker"].required is True

    def test_the_inference_server_is_not_required(self, config):
        """FR-47: reasoning unavailability must not stop regulatory control."""
        assert dependencies_by_name(config)["inference server"].required is False

    @pytest.mark.parametrize("system", [start.LINUX, start.WINDOWS, start.MACOS])
    def test_every_platform_gets_a_remediation_hint(self, config, system):
        for dependency in start.dependencies(config):
            assert dependency.hint(system)

    def test_an_unknown_platform_falls_back_to_the_linux_hint(self, config):
        broker = dependencies_by_name(config)["mqtt broker"]
        assert broker.hint("Plan9") == broker.hint(start.LINUX)


def dependencies_by_name(config):
    return {dependency.name: dependency for dependency in start.dependencies(config)}


class TestPreflight:
    def test_passes_when_everything_answers(self, config, all_reachable):
        assert start.preflight(config, start.LINUX) is True

    def test_fails_when_the_broker_is_absent(self, config, none_reachable):
        assert start.preflight(config, start.LINUX) is False

    def test_passes_when_only_the_inference_server_is_absent(self, config, monkeypatch):
        broker_port = config.mqtt.port
        monkeypatch.setattr(
            start, "is_reachable", lambda host, port, **_: port == broker_port
        )
        assert start.preflight(config, start.LINUX) is True


class TestReachabilityProbe:
    def test_an_unbound_port_is_not_reachable(self):
        assert start.is_reachable("127.0.0.1", 1, timeout_s=0.2) is False

    def test_a_listening_port_is_reachable(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            _, port = server.getsockname()
            assert start.is_reachable("127.0.0.1", port, timeout_s=1.0) is True

    def test_a_malformed_host_is_not_reachable_rather_than_raising(self):
        assert start.is_reachable("", -1, timeout_s=0.2) is False


class TestCommandLine:
    def test_check_reports_success_without_starting_anything(
        self, config, all_reachable
    ):
        assert start.main(["--check", "--config", str(CONFIG_PATH)]) == 0

    def test_check_still_returns_zero_when_a_dependency_is_missing(
        self, config, none_reachable
    ):
        """--check reports; it is not a gate."""
        assert start.main(["--check", "--config", str(CONFIG_PATH)]) == 0

    def test_a_missing_broker_stops_the_launch(self, config, none_reachable):
        assert start.main(["--config", str(CONFIG_PATH)]) == 1

    def test_an_unknown_service_name_is_a_usage_error(self, config, all_reachable):
        assert start.main(["--only", "nope", "--config", str(CONFIG_PATH)]) == 2

    def test_a_missing_config_file_is_reported_clearly(self, tmp_path):
        assert start.main(["--config", str(tmp_path / "absent.yaml")]) == 2
