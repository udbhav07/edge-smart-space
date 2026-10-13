"""The in-process system used by tests and experiments."""

from pathlib import Path

from eval.system import System
from src.common.clock import SimClock
from src.common.config import load_config


def test_it_never_writes_the_configured_state_file(tmp_path):
    """U-28: simulated time saved into state/ was discarded at the next live
    start as 410 days old. Persistence is exercised, but privately."""
    config = load_config(Path("config/default.yaml"))
    live = tmp_path / "coefficients.json"
    persistence = config.persistence.model_copy(update={"path": str(live), "interval_s": 60.0})
    system = System(config.model_copy(update={"persistence": persistence}), SimClock())
    system.run_for(600.0)
    assert not live.exists()


def test_its_own_estimate_is_still_persisted():
    system = System(load_config(Path("config/default.yaml")), SimClock())
    system.run_for(600.0)
    assert Path(system.config.persistence.path).exists()
