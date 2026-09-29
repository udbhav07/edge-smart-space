"""The Orin's provisioning, the model server and the node agree with the code.

Nothing here touches a Jetson. What it checks is the class of mistake that
only appears at bring-up: a script enabling a unit that does not exist, a
model server on a port the config does not look at, a node publishing on a
topic the bridge does not subscribe to.
"""

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

from src.common.config import load_config

DEPLOY = Path("deploy")
PROVISION = DEPLOY / "provision.sh"
LLAMA_UNIT = DEPLOY / "llama-server.service"
NODE = DEPLOY / "esphome" / "room-node.yaml"
UNITS = DEPLOY / "systemd"


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


@pytest.fixture(name="script")
def _script() -> str:
    return PROVISION.read_text(encoding="utf-8")


class TestProvisioning:
    def test_it_stops_on_the_first_error(self, script):
        assert "set -euo pipefail" in script

    def test_it_can_be_rehearsed_without_changing_anything(self, script):
        assert "--dry-run" in script

    def test_every_unit_it_enables_exists(self, script):
        for unit in re.findall(r"systemctl (?:enable|disable) ([\w@.-]+)", script):
            if unit.startswith(("jetson-clocks", "mosquitto")):
                continue
            assert (UNITS / unit).is_file() or (DEPLOY / unit).is_file(), unit

    def test_it_runs_the_services_as_the_user_the_units_name(self, script):
        user = re.search(r'^SERVICE_USER="([^"]+)"', script, re.M).group(1)
        for unit in UNITS.glob("*.service"):
            assert f"User={user}" in unit.read_text(encoding="utf-8"), unit.name

    def test_the_power_mode_is_pinned_for_reproducible_latency(self, script):
        """Section 9.2: E7's numbers are not reproducible otherwise."""
        assert "nvpmodel -m 0" in script and "jetson_clocks" in script

    def test_the_model_server_listens_where_the_config_looks(self, script, config):
        port = re.search(r'^LLM_PORT="(\d+)"', script, re.M).group(1)
        assert int(port) == urlparse(config.reasoning.base_url).port

    def test_it_enables_exactly_one_layer_1(self, script):
        """Two Layer 1s on the same topics would look from above like one
        sensor contradicting itself."""
        branch = script[script.index('if [[ "$SOURCE" == "simulated" ]]'):]
        simulated, rest = branch.split("else", 1)
        hardware = rest.split("fi", 1)[0]
        assert "enable smart-space-simulator" in simulated
        assert "disable smart-space-devices" in simulated
        assert "enable smart-space-devices" in hardware
        assert "disable smart-space-simulator" in hardware


class TestTheTarget:
    def test_it_names_no_layer_1(self):
        """A static Wants= would start that Layer 1 whatever was enabled."""
        wants = [
            line
            for line in (UNITS / "smart-space.target").read_text(encoding="utf-8").splitlines()
            if line.startswith("Wants=")
        ]
        assert not any("simulator" in line or "devices" in line for line in wants)

    @pytest.mark.parametrize("unit", ["smart-space-simulator", "smart-space-devices"])
    def test_both_layer_1s_install_into_it(self, unit):
        text = (UNITS / f"{unit}.service").read_text(encoding="utf-8")
        assert "WantedBy=smart-space.target" in text


class TestTheModelServer:
    def test_it_uses_the_models_own_tool_template(self):
        """Section 5.7.3: without --jinja, tool calls come back as prose."""
        assert "--jinja" in LLAMA_UNIT.read_text(encoding="utf-8")

    def test_every_layer_is_offloaded_to_the_gpu(self):
        assert "--n-gpu-layers 999" in LLAMA_UNIT.read_text(encoding="utf-8")

    def test_it_listens_only_on_this_machine(self):
        assert "--host 127.0.0.1" in LLAMA_UNIT.read_text(encoding="utf-8")

    def test_it_serves_the_model_name_the_config_asks_for(self, config):
        assert f"--alias {config.reasoning.model}" in LLAMA_UNIT.read_text(encoding="utf-8")


class TestTheNode:
    def test_it_is_marked_unverified(self):
        assert "UNVERIFIED" in NODE.read_text(encoding="utf-8")

    def test_it_publishes_every_topic_the_bridge_subscribes_to(self, config):
        node = NODE.read_text(encoding="utf-8")
        topics = list(config.devices.sensor_topics.values()) + [config.devices.door_topic]
        for topic in topics:
            assert topic in node, topic

    def test_it_listens_where_the_bridge_commands_the_unit(self, config):
        node = NODE.read_text(encoding="utf-8")
        assert config.devices.ac_mode_command_topic in node
        assert config.devices.ac_target_command_topic in node
