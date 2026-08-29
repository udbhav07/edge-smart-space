"""Ground-truth thermal plant.

This module must not import from ``src.estimation`` (DESIGN.md section
5.10). The simulated plant and the estimator's internal model have to be
independently parameterised, or the evaluation degenerates into the model
predicting itself and every reported result is vacuous.

That independence is structural here, not a convention:

* The plant is parameterised by physical R, C and power in watts. The
  estimator identifies a1 to a4, a discrete-time ARX form. Neither can read
  the other's numbers.
* The plant integrates the continuous equation exactly over the step, using
  the analytic solution for a first-order lag. The estimator assumes a
  zero-order hold discretisation. The two therefore disagree slightly by
  construction, which is what a real plant does to a model.
* The plant is driven by a solar-gain disturbance the estimator has no
  regressor for at all (R-04), so a perfectly converged estimator still
  cannot achieve zero residual.

Continuous form (DESIGN.md section 5.2.1):

    C dT/dt = (T_out - T) / R + Q_hvac + Q_occ + Q_solar
"""

from __future__ import annotations

import math

from src.common.clock import Clock
from src.common.config import RoomConfig

#: Cooling command is a normalised fraction of the unit's rated power.
COMMAND_BOUNDS = (0.0, 1.0)

#: Solar gain is half-wave rectified: a daylight cycle heats or does nothing,
#: never cools.
_NO_GAIN_W = 0.0

_FULL_CYCLE_RADIANS = 2.0 * math.pi


class RoomModel:
    """The room, as physics rather than as a model of physics.

    State is the single zone temperature (A-01). ``step`` advances it by an
    arbitrary interval, so the caller controls the integration cadence and a
    scenario can run faster than wall-clock time.
    """

    def __init__(self, config: RoomConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._temperature_c = config.initial_temperature_c
        self._start_ts = clock.now()

    @property
    def temperature_c(self) -> float:
        """Current true zone temperature. Sensors observe this imperfectly."""
        return self._temperature_c

    @property
    def time_constant_s(self) -> float:
        """R times C. How long the room takes to forget its own state."""
        return (
            self._config.thermal_resistance_k_per_w
            * self._config.thermal_capacitance_j_per_k
        )

    def solar_gain_w(self) -> float:
        """Disturbance the estimator cannot represent (R-04).

        Half-wave rectified so it models daylight: a cycle that adds heat for
        part of the period and nothing for the rest.
        """
        elapsed_s = self._clock.now() - self._start_ts
        phase = _FULL_CYCLE_RADIANS * elapsed_s / self._config.solar_gain_period_s
        return self._config.solar_gain_amplitude_w * max(_NO_GAIN_W, math.sin(phase))

    def heat_flow_w(self, cooling_fraction: float, occupied: bool) -> float:
        """Net non-ambient heat into the zone, in watts.

        Cooling is negative. Occupancy and solar gain are positive.
        """
        cooling_w = -self._config.cooling_power_w * cooling_fraction
        occupancy_w = self._config.occupant_gain_w if occupied else _NO_GAIN_W
        return cooling_w + occupancy_w + self.solar_gain_w()

    def step(
        self,
        duration_s: float,
        cooling_fraction: float,
        occupied: bool,
        outdoor_c: float,
    ) -> float:
        """Advance the plant and return the new true temperature.

        Integrates exactly rather than by an Euler step: for a first-order lag
        with constant inputs over the interval the analytic solution is
        available, and using it keeps the plant stable at any step size
        instead of only at small ones.

        :param duration_s: interval to advance. Must be positive.
        :param cooling_fraction: normalised command in [0, 1].
        :param occupied: whether the internal gain is present.
        :param outdoor_c: ambient temperature held constant over the interval.
        :raises ValueError: if the duration is not positive or the command is
            outside its normalised range.
        """
        if duration_s <= 0.0:
            raise ValueError(f"duration must be positive, got {duration_s!r}")
        low, high = COMMAND_BOUNDS
        if not low <= cooling_fraction <= high:
            raise ValueError(
                f"cooling fraction must lie in [{low}, {high}], "
                f"got {cooling_fraction!r}"
            )

        heat_w = self.heat_flow_w(cooling_fraction, occupied)
        equilibrium_c = outdoor_c + self._config.thermal_resistance_k_per_w * heat_w
        decay = math.exp(-duration_s / self.time_constant_s)

        self._temperature_c = equilibrium_c + (self._temperature_c - equilibrium_c) * decay
        return self._temperature_c
