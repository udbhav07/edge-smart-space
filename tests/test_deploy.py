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
