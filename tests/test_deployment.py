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

    def test_the_target_wants_only_one_of_them(self):
        """Two Layer 1 implementations on the same topics would look from
        above like one sensor contradicting itself."""
        text = (SYSTEMD / "space.target").read_text(encoding="utf-8")
        assert "space-layer1@" in text
        assert "space-simulator@" not in text


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
