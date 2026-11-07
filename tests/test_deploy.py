"""The systemd units match the services start.py knows (NFR-08, section 9.3)."""

import configparser
from pathlib import Path

import pytest

import start

UNITS = Path("deploy/systemd")


def _unit(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.read(UNITS / f"smart-space-{name}.service", encoding="utf-8")
    return parser


@pytest.mark.parametrize("service", start.SERVICES, ids=lambda service: service.name)
def test_every_service_has_a_unit(service):
    assert (UNITS / f"smart-space-{service.name}.service").exists()


@pytest.mark.parametrize("service", start.SERVICES, ids=lambda service: service.name)
def test_every_unit_restarts_on_its_own(service):
    assert _unit(service.name)["Service"]["Restart"] == "always"


@pytest.mark.parametrize("service", start.SERVICES, ids=lambda service: service.name)
def test_every_unit_runs_the_module_start_py_runs(service):
    assert f"-m {service.module} " in _unit(service.name)["Service"]["ExecStart"]


def test_there_is_no_unit_for_a_service_that_does_not_exist():
    names = {f"smart-space-{service.name}.service" for service in start.SERVICES}
    assert {path.name for path in UNITS.glob("*.service")} == names


@pytest.mark.parametrize("service", start.SERVICES, ids=lambda service: service.name)
def test_every_unit_reads_the_nodes_own_layer(service):
    """site.yaml extends the calibration, which extends the reference (R-04)."""
    assert _unit(service.name)["Service"]["ExecStart"].endswith("/config/site.yaml")


# --- SearXNG (FR-101, FR-102) ----------------------------------------------

import yaml  # noqa: E402

from src.common.config import load_config  # noqa: E402

SEARXNG_SETTINGS = Path("deploy/searxng/settings.yml")


def _searxng() -> dict:
    return yaml.safe_load(SEARXNG_SETTINGS.read_text(encoding="utf-8"))


def test_searxng_answers_in_json_for_the_assistance_service():
    assert "json" in _searxng()["search"]["formats"]


def test_searxng_searches_in_the_region_the_config_asks_for():
    assert _searxng()["search"]["default_lang"] == load_config(Path("config/default.yaml")).web_search.region


def test_searxng_keeps_no_secret_in_the_repository():
    # start.py makes one on first run and passes it in the environment.
    assert "secret_key" not in _searxng().get("server", {})


def test_searxng_runs_without_the_limiter_a_public_instance_needs():
    assert _searxng()["server"]["limiter"] is False


def test_the_settings_directory_start_py_mounts_holds_the_settings():
    assert Path(load_config(Path("config/default.yaml")).web_search.searxng.settings_dir, "settings.yml") == SEARXNG_SETTINGS
