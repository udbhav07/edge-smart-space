"""The deployment must describe the system that actually exists.

Nothing here talks to a Jetson. What it checks is the class of mistake that
survives every other test and only appears at bring-up, at night, on hardware:
a unit file naming a module nobody wrote, a device topic that matches nothing
because an entity was renamed, or a component with no way to start.

These are cheap to check and expensive to discover, which is the whole
argument for checking them.
"""

import re
from pathlib import Path

import pytest

from src.common.config import Layer1Source, load_config

DEPLOY = Path("deploy")
SYSTEMD = DEPLOY / "systemd"
ESPHOME_NODE = DEPLOY / "esphome" / "room-node.yaml"
PROVISION = DEPLOY / "provision.sh"
LLAMA_UNIT = SYSTEMD / "llama-server.service"

#: Every unit that starts a Python component.
UNIT_FILES = sorted(SYSTEMD.glob("space-*.service"))


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


def _module_of(unit: Path) -> str:
    text = unit.read_text(encoding="utf-8")
    match = re.search(r"ExecStart=.*-m (\S+)", text)
    assert match, f"{unit.name} starts nothing"
    return match.group(1)


class TestUnitFiles:
    def test_there_are_units_to_check(self):
        assert UNIT_FILES

    @pytest.mark.parametrize("unit", UNIT_FILES, ids=lambda p: p.name)
    def test_every_unit_starts_a_module_that_exists(self, unit):
        """A unit naming a module nobody wrote fails at boot, on the Orin,
        with the room warm."""
        module = _module_of(unit)
        path = Path(module.replace(".", "/"))
        assert (path / "__main__.py").is_file() or path.with_suffix(
            ".py"
        ).is_file(), f"{unit.name} starts {module}, which does not exist"

    @pytest.mark.parametrize("unit", UNIT_FILES, ids=lambda p: p.name)
    def test_every_unit_restarts_itself(self, unit):
        """Restart independence is a requirement rather than a convenience
        (section 9.3): killing one process and watching the rest carry on is
        part of the demonstration."""
        assert "Restart=always" in unit.read_text(encoding="utf-8")

    @pytest.mark.parametrize("unit", UNIT_FILES, ids=lambda p: p.name)
    def test_no_unit_requires_another_component(self, unit):
        """Requires= would let one component's failure refuse to bring up
        another, which is the opposite of what the design asks for."""
        text = unit.read_text(encoding="utf-8")
        assert "Requires=" not in text
        assert "BindsTo=" not in text

    def test_every_startable_component_has_a_unit(self):
        """A component with no unit is one somebody has to remember to start
        by hand, which on the night nobody does."""
        started = {_module_of(unit) for unit in UNIT_FILES}
        for module in (
            "src.estimation",
            "src.control",
            "src.faults",
            "src.io",
            "src.reasoning",
            "src.assistance",
            "sim.run_sim",
        ):
            assert module in started

    def test_the_target_wants_rather_than_requires(self):
        """One failing unit should degrade the system, not refuse to start
        it -- the same rule the code follows for a missing speech stack."""
        text = (SYSTEMD / "space.target").read_text(encoding="utf-8")
        assert "Wants=" in text
        assert "Requires=" not in text


class TestLayerOneIsExclusive:
    def test_both_layer_one_units_exist(self):
        assert (SYSTEMD / "space-layer1@.service").is_file()
        assert (SYSTEMD / "space-simulator@.service").is_file()

    def test_the_target_wants_neither_statically(self):
        """A static Wants= would start that Layer 1 on every boot whatever
        was enabled; which one runs is decided by enabling it."""
        wants = [
            line
            for line in (SYSTEMD / "space.target").read_text(encoding="utf-8").splitlines()
            if line.startswith("Wants=")
        ]
        assert not any("layer1" in line or "simulator" in line for line in wants)

    @pytest.mark.parametrize(
        "unit", ["space-layer1@.service", "space-simulator@.service"]
    )
    def test_both_install_into_the_target(self, unit):
        assert "WantedBy=space.target" in (SYSTEMD / unit).read_text(encoding="utf-8")

    def test_provisioning_enables_exactly_the_one_asked_for(self):
        """Two Layer 1 implementations on the same topics would look from
        above like one sensor contradicting itself."""
        script = PROVISION.read_text(encoding="utf-8")
        simulated = script[script.index('if [[ "$SOURCE" == "simulated" ]]'):]
        branch, other = simulated.split("else", 1)
        assert "disable space-layer1@" in branch and "enable space-simulator@" in branch
        other = other.split("fi", 1)[0]
        assert "disable space-simulator@" in other and "enable space-layer1@" in other


class TestDeviceTopicsMatchTheNode:
    """The bridge and the node have to agree, and nothing else enforces it."""

    def test_the_node_definition_exists(self):
        assert ESPHOME_NODE.is_file()

    def test_every_configured_device_topic_appears_in_the_node(self, config):
        """A renamed entity moves its topic, and the bridge then subscribes to
        something no node publishes -- silently, because a subscription to a
        topic nobody uses looks exactly like a working one."""
        node = ESPHOME_NODE.read_text(encoding="utf-8")
        prefix = "esphome/room"
        for binding in config.io.devices:
            suffix = binding.topic.removeprefix(prefix)
            assert suffix in node, (
                f"{binding.sensor_id} expects {binding.topic}, which the node "
                f"definition does not publish"
            )

    def test_every_configured_sensor_has_a_device(self, config):
        """A sensor with no device binding publishes nothing, and D1 reports
        that as a dropped sensor when it is a configuration gap."""
        bound = {binding.sensor_id for binding in config.io.devices}
        for sensor in config.sensors.adapters:
            assert sensor.sensor_id in bound

    def test_the_node_is_marked_unverified(self):
        """It has never been flashed. Code that looks finished and silently
        matches nothing is worse than code that is obviously unfinished."""
        assert "UNVERIFIED" in ESPHOME_NODE.read_text(encoding="utf-8")


class TestTheBroker:
    def test_persistence_is_on(self):
        """Retained topics are how a late subscriber learns the current mode,
        the active faults and the last applied goal (FR-61). Losing them on a
        broker restart leaves every component correct and the system as a
        whole amnesiac."""
        text = (DEPLOY / "mosquitto.conf").read_text(encoding="utf-8")
        assert "persistence true" in text

    def test_the_compose_file_names_the_configured_port(self, config):
        text = (DEPLOY / "docker-compose.yml").read_text(encoding="utf-8")
        assert f"{config.mqtt.port}:{config.mqtt.port}" in text


class TestSimulationRemainsTheDefault:
    def test_the_shipped_configuration_runs_the_simulator(self, config):
        """Hardware is opt-in. A repository that defaulted to hardware would
        fail on every machine that has none, which is all of them until the
        Orin is wired."""
        assert config.io.source is Layer1Source.SIMULATED


class TestProvisioning:
    """Week 5: the Orin comes up on boot with nothing typed by hand."""

    def test_the_script_exists_and_stops_on_the_first_error(self):
        assert PROVISION.read_text(encoding="utf-8").count("set -euo pipefail") == 1

    def test_it_can_be_rehearsed_without_changing_anything(self):
        assert "--dry-run" in PROVISION.read_text(encoding="utf-8")

    def test_the_whole_system_is_enabled_at_boot(self):
        script = PROVISION.read_text(encoding="utf-8")
        assert "systemctl enable space.target" in script
        assert "systemctl enable llama-server.service" in script

    def test_the_target_comes_up_at_boot(self):
        text = (SYSTEMD / "space.target").read_text(encoding="utf-8")
        assert "WantedBy=multi-user.target" in text

    def test_the_power_mode_is_pinned_for_reproducible_latency(self):
        """Section 9.2: E7's numbers are not reproducible otherwise."""
        script = PROVISION.read_text(encoding="utf-8")
        assert "nvpmodel -m 0" in script and "jetson_clocks" in script

    def test_the_machine_zone_agrees_with_the_configured_offset(self, config):
        """Local time is epoch plus site.utc_offset_h; the machine should agree
        so its logs and ours read the same clock."""
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        import datetime as dt

        match = re.search(r'^TIMEZONE="([^"]+)"', PROVISION.read_text(encoding="utf-8"), re.M)
        try:
            zone = ZoneInfo(match.group(1))
        except ZoneInfoNotFoundError:
            pytest.skip("no time zone database on this machine")
        offset = dt.datetime(2026, 1, 1, tzinfo=zone).utcoffset().total_seconds() / 3600
        assert offset == config.site.utc_offset_h

    def test_the_model_server_listens_where_the_config_looks(self, config):
        from urllib.parse import urlparse

        match = re.search(r'^LLM_PORT="(\d+)"', PROVISION.read_text(encoding="utf-8"), re.M)
        assert int(match.group(1)) == urlparse(config.reasoning.base_url).port

    def test_every_unit_it_installs_is_in_the_repository(self):
        assert LLAMA_UNIT.is_file()
        assert (SYSTEMD / "space-reasoning@.service").is_file()


class TestTheModelServer:
    def test_it_uses_the_models_own_tool_template(self):
        """Section 5.7.3: without --jinja, tool calls come back as prose."""
        assert "--jinja" in LLAMA_UNIT.read_text(encoding="utf-8")

    def test_every_layer_is_offloaded_to_the_gpu(self):
        """GPU first; a silent CPU fallback looks like slow code (section 9.2)."""
        assert "--n-gpu-layers 999" in LLAMA_UNIT.read_text(encoding="utf-8")

    def test_it_listens_only_on_this_machine(self):
        """NFR-06 and FR-51 in spirit: nothing reasoning-related is reachable
        from outside the node."""
        assert "--host 127.0.0.1" in LLAMA_UNIT.read_text(encoding="utf-8")

    def test_it_serves_the_model_name_the_config_asks_for(self, config):
        assert f"--alias {config.reasoning.model}" in LLAMA_UNIT.read_text(encoding="utf-8")

    def test_it_restarts_itself(self):
        assert "Restart=always" in LLAMA_UNIT.read_text(encoding="utf-8")

    def test_the_reasoning_process_does_not_require_it(self):
        """FR-47: reasoning starts without it and records UNAVAILABLE."""
        text = (SYSTEMD / "space-reasoning@.service").read_text(encoding="utf-8")
        assert "Requires=llama-server" not in text
