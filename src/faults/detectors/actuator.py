"""D5: the air conditioner is not actually cooling (FR-24).

The second detector that only exists because the model exists -- or more
precisely, because an *expectation* exists. R-02 says the IR path is
open-loop: commands go out and nothing comes back, so ``AckStatus.UNKNOWN`` is
the normal, correct answer to "did that land?". A system that treated a
missing acknowledgement as success would believe it was cooling a room it was
not touching, indefinitely.

So the test is not "was the command acknowledged" but "did the room respond".
The only honest evidence that an actuator works is the plant moving.

**Sustained cooling is tracked by state, not by counting commands.** After a
COOL the controller emits MAINTAIN on every tick while the compressor stays on
-- MAINTAIN means "keep doing what you are doing" -- so counting COOL messages
would see one command and conclude cooling had stopped. Only COOL and OFF
change the state; MAINTAIN and HOLD leave it alone. This is the same
distinction the validator draws for dwell and rate limits.

**The test is against the model's expectation, not a fixed number of
degrees.** This is what makes D5 one of the two detectors a thermostat cannot
have. A room sitting near its equilibrium cannot cool much however well the
air conditioner works, so a fixed threshold blames the actuator for physics:
with one, this detector raised a fault within an hour of every healthy run,
because the room had simply arrived where it was going. What the model
supplies is the counterfactual -- how much cooling *should* have happened over
this window -- and the fault is the room delivering a small fraction of it.

Expected cooling is accumulated one step at a time, and the arithmetic has to
be done carefully. A published estimate pairs ``t_pred`` -- the model's
prediction of *this* instant, made one step ago -- with ``t_in``, the
measurement of the same instant. Their difference is the prediction error, not
an expectation, and summing its negative half just accumulates noise: doing
exactly that produced an expected 9.5 C of cooling over a window in which the
room honestly moved 2 C, and D5 called a working unit dead. What the model
expects over a step is ``t_pred`` against the measurement it was predicted
*from*, so the previous reading is kept and differenced against the next
prediction.

Summing those across the window approximates the cooling expected across it.
It is an approximation, because it re-anchors to the measurement every step
rather than compounding, which is why the bar is a generous fraction rather
than a tight one. The fault being caught is an actuator doing nothing at all,
which delivers close to zero against an expectation of several tenths.

**The window is long because the room is slow.** Ten minutes, against a
thermal time constant measured in tens of minutes. An actuator fault is found
on the order of ten minutes and NFR-02 deliberately does not promise better.

**Where this can still be wrong.** The model is identified from the same room
the test is about, so a persistent unmodelled heat load -- a door propped open
all afternoon -- eventually becomes part of what the model expects, and the
test stops seeing it. R-04 settles the numbers with measured data during
bring-up. What cannot be settled is that this detector is only ever as good as
the model behind it.
"""

from __future__ import annotations

from src.common.clock import Clock
from src.common.config import ActuatorDetectorConfig
from src.common.schemas import (
    CommandKind,
    DetectorId,
    SensorReading,
    ThermalEstimate,
)
from src.faults.detectors.base import Finding, Judgment

#: Commands that change whether the compressor is running. MAINTAIN and HOLD
#: assert the status quo and must not restart the evaluation window.
_STATE_CHANGING = frozenset({CommandKind.COOL, CommandKind.OFF})


class ActuatorResponseDetector:
    """Watches whether sustained cooling actually cools the room.

    Holds the temperature at which the current cooling period began and the
    moment it began. Both reset whenever cooling stops, because a window that
    straddled an OFF would compare temperatures from two different regimes.
    """

    def __init__(
        self,
        subject: str,
        config: ActuatorDetectorConfig,
        clock: Clock,
    ) -> None:
        if not subject:
            raise ValueError("a detector must name the subject it watches")
        self._subject = subject
        self._config = config
        self._clock = clock
        self._cooling = False
        self._started_s: float | None = None
        self._start_temperature_c: float | None = None
        self._latest_temperature_c: float | None = None
        self._expected_cooling_c = 0.0
        self._estimates_seen = 0
        self._previous_t_in_c: float | None = None

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def cooling(self) -> bool:
        """Whether the compressor is believed to be running."""
        return self._cooling

    def observe_command(self, kind: CommandKind) -> None:
        """Note a command, if it changes whether the plant is being cooled."""
        if kind not in _STATE_CHANGING:
            return
        if kind is CommandKind.COOL:
            self._begin_cooling()
        else:
            self._stop_cooling()

    def observe_reading(self, reading: SensorReading) -> None:
        """Note the room temperature.

        Every reading offered is accepted. The bank hands this detector the
        indoor temperature and nothing else, and naming a sensor id here would
        put a second source of truth beside the one in configuration.

        The reading's value is what matters and its timestamp is not used:
        elapsed time comes from the injected clock, for the same reason D1
        keys on arrival. A node with an unset clock would otherwise make the
        window look either instantaneous or fifty years long.
        """
        self._latest_temperature_c = reading.value
        if self._cooling and self._start_temperature_c is None:
            self._start_temperature_c = reading.value
            self._started_s = self._clock.monotonic()

    def observe_estimate(self, estimate: ThermalEstimate) -> None:
        """Accumulate the cooling the model expects over this window.

        Only while cooling is being commanded. An expectation gathered while
        the compressor was off says nothing about whether the compressor
        works, and warming the model expects is not cooling it expects, so
        only the cooling half is accumulated.
        """
        self._estimates_seen += 1
        previous_c, self._previous_t_in_c = self._previous_t_in_c, estimate.t_in

        if not self._model_is_warm():
            # The expectation is still the prior, which over-predicts cooling
            # threefold. Judging against it blames a working air conditioner
            # for the estimator not having learned the room yet.
            return
        if not self._cooling or self._started_s is None or previous_c is None:
            return

        # The model's prediction is of this instant, formed from the previous
        # one, so the change it expected is measured against that.
        expected_change_c = estimate.t_pred - previous_c
        if expected_change_c < 0.0:
            self._expected_cooling_c += -expected_change_c

    def _model_is_warm(self) -> bool:
        """Whether the model behind the expectation can be trusted yet."""
        return self._estimates_seen > self._config.warmup_samples

    @property
    def expected_cooling_c(self) -> float:
        """Cooling the model has predicted across the current window."""
        return self._expected_cooling_c

    def reset(self) -> None:
        """Abandon the current evaluation window.

        Called when the fault retires, so the next verdict is measured from a
        fresh window rather than from temperatures recorded before whatever
        repair happened.
        """
        self._started_s = None
        self._start_temperature_c = None
        self._expected_cooling_c = 0.0

    def evaluate(self) -> Finding:
        """Judge the actuator on the current cooling period.

        UNKNOWN whenever there is nothing to judge: the plant is not being
        cooled, it has not been cooled for long enough to expect a measurable
        change, or no temperature has arrived to measure it with. An actuator
        nobody has asked to do anything has not been shown to work, and
        reporting it CLEAR would claim evidence that was never gathered --
        while reporting it FAULTED on a window with no temperatures in it
        would blame the air conditioner for a broken sensor.
        """
        if not self._model_is_warm():
            return self._finding(Judgment.UNKNOWN, cooled_c=0.0, elapsed_s=0.0)

        if not self._cooling or self._started_s is None:
            return self._finding(Judgment.UNKNOWN, cooled_c=0.0, elapsed_s=0.0)

        if self._start_temperature_c is None or self._latest_temperature_c is None:
            return self._finding(Judgment.UNKNOWN, cooled_c=0.0, elapsed_s=0.0)

        elapsed_s = self._clock.monotonic() - self._started_s
        if elapsed_s < self._config.evaluation_window_s:
            return self._finding(Judgment.UNKNOWN, cooled_c=0.0, elapsed_s=elapsed_s)

        # Captured before any re-anchor, because re-anchoring clears the
        # expectation and the evidence has to say what the decision was
        # actually made on. Reported zero for every CLEAR verdict until this
        # was fixed, which made the healthy case impossible to calibrate from
        # its own audit trail.
        expected_c = self._expected_cooling_c
        cooled_c = self._cooling_achieved_c()
        required_c = expected_c * self._config.response_fraction

        if expected_c < self._config.min_expected_cooling_c:
            # The model says barely anything should happen: the room is at its
            # equilibrium, or the ambient has the plant at capacity. There is
            # no test to run, so nothing is claimed, and the window restarts so
            # the next verdict is measured over a fresh stretch.
            self._anchor_window()
            return self._finding(Judgment.UNKNOWN, 0.0, elapsed_s, expected_c)

        if cooled_c >= required_c:
            # It works. Re-anchor, so the next window is a fresh test rather
            # than a verdict that stands on one success forever.
            self._anchor_window()
            return self._finding(Judgment.CLEAR, cooled_c, elapsed_s, expected_c)

        return self._finding(Judgment.FAULTED, cooled_c, elapsed_s, expected_c)

    def _cooling_achieved_c(self) -> float:
        """How much the room fell over the window. Negative means it rose.

        Signed rather than absolute: a room that got *warmer* under
        sustained cooling is the more certain actuator fault of the two, and
        ``|dT|`` would miss it.
        """
        if self._start_temperature_c is None or self._latest_temperature_c is None:
            return 0.0
        return self._start_temperature_c - self._latest_temperature_c

    def _begin_cooling(self) -> None:
        if self._cooling:
            return
        self._cooling = True
        self._anchor_window()

    def _stop_cooling(self) -> None:
        """End the cooling period, and the expectation gathered during it.

        Clearing the expectation matters as much as clearing the window. It is
        accumulated per window, and leaving it behind lets it grow across every
        cycle of an ordinary deadband: measured, it reached an expected 9.5 C
        of cooling over a window in which the room honestly moved 2 C, and D5
        called a healthy air conditioner dead.
        """
        self._cooling = False
        self._started_s = None
        self._start_temperature_c = None
        self._expected_cooling_c = 0.0

    def _anchor_window(self) -> None:
        """Start a fresh evaluation window from the temperature now known."""
        self._started_s = self._clock.monotonic()
        self._start_temperature_c = self._latest_temperature_c
        self._expected_cooling_c = 0.0

    def _finding(
        self,
        judgment: Judgment,
        cooled_c: float,
        elapsed_s: float,
        expected_c: float | None = None,
    ) -> Finding:
        expected = (
            self._expected_cooling_c if expected_c is None else expected_c
        )
        return Finding(
            detector=DetectorId.D5_ACTUATOR_NO_RESPONSE,
            subject=self._subject,
            judgment=judgment,
            confidence=1.0 if judgment is Judgment.FAULTED else 0.0,
            evidence={
                "cooled_c": cooled_c,
                "expected_cooling_c": expected,
                "required_cooling_c": expected * self._config.response_fraction,
                "elapsed_s": elapsed_s,
                "window_s": self._config.evaluation_window_s,
                "estimates_seen": float(self._estimates_seen),
                "start_temperature_c": (
                    self._start_temperature_c
                    if self._start_temperature_c is not None
                    else 0.0
                ),
                "latest_temperature_c": (
                    self._latest_temperature_c
                    if self._latest_temperature_c is not None
                    else 0.0
                ),
            },
        )
