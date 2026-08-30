"""E1: does RLS converge to physically plausible coefficients?

DESIGN.md section 8.3 sets the method -- a 24 h simulated run against known
plant parameters -- and the metrics: coefficient error against ground truth,
and |a1 + a2 - 1|.

The comparison is only well posed because the ground truth can be *derived*
rather than assumed. The plant is parameterised by physical R, C and watts
and never by a1 to a4 (section 5.10), so this module discretises the
continuous equation itself and computes what the four coefficients must be
for that room. The estimator and the plant remain independently
parameterised; what is compared is the estimator's answer against
arithmetic, not against a number it was handed.

Two runs are reported, and the distinction is the honest part:

* **Identifiable.** Solar gain off, so the plant genuinely *is* the ARX
  model and a correct theta exists. This is the run that answers E1 as
  section 8.3 poses it.
* **With an unmodelled disturbance.** The shipped simulator drives a solar
  term the estimator has no regressor for (R-04). No theta is correct here,
  and the residual cannot reach zero however well the identification works.
  Reporting the first number without this one would overstate the result.

Run it: ``python -m eval.experiments.e1_convergence``
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.loopback import LoopbackTransport
from src.common.clock import SimClock
from src.common.config import Config, RoomConfig, load_config
from src.common.mqtt_client import Blackboard
from src.common import topics
from src.common.schemas import Command, CommandKind
from src.estimation.persistence import CoefficientStore
from src.estimation.rls import ThermalEstimator
from src.estimation.service import ThermalEstimatorService
from sim.run_sim import build_simulator

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

SECONDS_PER_HOUR = 3600.0
DEFAULT_HOURS = 24.0

#: Excitation schedule. A real room's air conditioner sits in a narrow band
#: and identifies a3 poorly (R-01), so the experiment drives deliberate
#: steps -- which is the mitigation the risk register already names.
EXCITATION_PERIOD_S = 900.0

#: Occupancy pattern, so a4 has something to identify against.
OCCUPANCY_PERIOD_S = 7200.0


def arx_ground_truth(room: RoomConfig, interval_s: float) -> np.ndarray:
    """The four coefficients this room must have, derived from R, C and watts.

    Zero-order-hold discretisation of

        C dT/dt = (T_out - T)/R + Q

    over one interval gives, with alpha = exp(-dt / RC):

        T[k+1] = alpha*T[k] + (1-alpha)*T_out + (1-alpha)*R*Q

    so a1 = alpha, a2 = 1 - alpha, and the driven terms carry (1-alpha)*R
    times their power. Note a1 + a2 = 1 exactly: steady-state consistency is
    not an approximation, it is what the discretisation guarantees when the
    model is correct (section 5.2.1).
    """
    time_constant_s = room.thermal_resistance_k_per_w * room.thermal_capacitance_j_per_k
    retained = math.exp(-interval_s / time_constant_s)
    coupling = 1.0 - retained
    gain = coupling * room.thermal_resistance_k_per_w
    return np.array(
        [
            retained,
            coupling,
            -gain * room.cooling_power_w,
            gain * room.occupant_gain_w,
        ]
    )


@dataclass(frozen=True)
class E1Result:
    """What one run measured."""

    label: str
    hours: float
    samples: int
    updates_fitted: int
    pairs_skipped: int
    truth: np.ndarray
    estimated: np.ndarray
    steady_state_residual: float
    residual_sigma: float
    model_confidence: float

    @property
    def errors(self) -> np.ndarray:
        return np.abs(self.estimated - self.truth)

    @property
    def worst_error(self) -> float:
        return float(np.max(self.errors))

    def report(self) -> str:
        names = ("a1", "a2", "a3", "a4")
        lines = [
            f"--- E1: {self.label} ---",
            f"  simulated {self.hours:.0f} h, {self.samples} samples, "
            f"{self.updates_fitted} fitted, {self.pairs_skipped} skipped",
        ]
        for name, truth, estimate, error in zip(
            names, self.truth, self.estimated, self.errors
        ):
            lines.append(
                f"  {name}  truth {truth: .6f}   estimated {estimate: .6f}   "
                f"error {error:.6f}"
            )
        lines.append(f"  worst coefficient error   {self.worst_error:.6f}")
        lines.append(f"  |a1 + a2 - 1|             {self.steady_state_residual:.6f}")
        lines.append(f"  residual sigma            {self.residual_sigma:.4f} C")
        lines.append(f"  model confidence          {self.model_confidence:.4f}")
        return "\n".join(lines)


def _excited_config(config: Config, identifiable: bool) -> Config:
    """Configuration for one run.

    The identifiable run silences the disturbance the estimator has no
    regressor for. Nothing else is softened: jitter, noise, quantisation,
    dropouts, dead-time and command loss all stay on, because a kind
    simulator makes a converged estimate mean nothing.
    """
    if not identifiable:
        return config
    room = config.sim.room.model_copy(update={"solar_gain_amplitude_w": 0.0})
    return config.model_copy(
        update={"sim": config.sim.model_copy(update={"room": room})}
    )


def run(config: Config, hours: float, identifiable: bool, label: str) -> E1Result:
    """Drive the simulator and the estimator through one another over MQTT."""
    run_config = _excited_config(config, identifiable)
    clock = SimClock()
    transport = LoopbackTransport()

    simulator_board = Blackboard(run_config.mqtt, transport)
    estimator_board = Blackboard(run_config.mqtt, transport)
    transport.attach(simulator_board)
    transport.attach(estimator_board)

    simulator = build_simulator(run_config, clock, simulator_board)
    simulator.subscribe()

    estimator = ThermalEstimator(run_config.estimator, clock)
    service = ThermalEstimatorService(
        config=run_config,
        clock=clock,
        blackboard=estimator_board,
        estimator=estimator,
        store=CoefficientStore(run_config.persistence, clock),
    )
    service.subscribe()

    interval_s = run_config.loop.sensor_period_s
    steps = int(hours * SECONDS_PER_HOUR / interval_s)
    start_ts = clock.now()

    for step in range(steps):
        elapsed_s = clock.now() - start_ts
        cooling = (elapsed_s % EXCITATION_PERIOD_S) < (EXCITATION_PERIOD_S / 2.0)
        simulator.occupied = (elapsed_s % OCCUPANCY_PERIOD_S) < (
            OCCUPANCY_PERIOD_S / 2.0
        )
        # Publish a real command rather than driving the actuator directly:
        # what the estimator identifies a3 against is the *published* state,
        # so bypassing the topic would leave u[k] at zero and a3 unidentified.
        kind = CommandKind.COOL if cooling else CommandKind.OFF
        command = Command(
            ts=clock.now(),
            actuator_id=topics.AIR_CONDITIONER_ID,
            kind=kind,
            setpoint_c=(
                run_config.controller.default_setpoint_c
                if kind is CommandKind.COOL
                else None
            ),
        )
        simulator_board.publish(
            topics.ACTUATOR_COMMAND,
            command,
            actuator_id=topics.AIR_CONDITIONER_ID,
        )
        simulator.step()
        clock.advance(interval_s)

    snapshot = estimator.snapshot()
    return E1Result(
        label=label,
        hours=hours,
        samples=steps,
        updates_fitted=estimator.samples_since_reset,
        pairs_skipped=service.skipped_pairs,
        truth=arx_ground_truth(run_config.sim.room, interval_s),
        estimated=estimator.coefficients,
        steady_state_residual=snapshot.steady_state_residual,
        residual_sigma=estimator.residual_sigma,
        model_confidence=estimator.model_confidence,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run experiment E1.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    config = load_config(arguments.config)

    identifiable = run(config, arguments.hours, True, "identifiable plant")
    disturbed = run(
        config, arguments.hours, False, "with an unmodelled solar disturbance"
    )

    print(identifiable.report())
    print()
    print(disturbed.report())
    print()
    print(
        "The second run is the honest one to quote alongside the first: the\n"
        "estimator has no regressor for solar gain (R-04), so no theta is\n"
        "correct there and the residual cannot reach zero however well the\n"
        "identification works."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
