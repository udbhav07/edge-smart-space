"""Run the speech pipeline as a service.

``python -m src.speech``

Wires the microphone to the wake word, the endpointer, on-device
transcription and the single-shot Personal Context call, then publishes what
comes out to ``space/context/preference``.

What it publishes is a *hint*, not a command (FR-53). Nothing here can move
an actuator: a spoken preference enters the blackboard exactly like any
other supervisory input and is gated by the validator downstream, which is
what FR-45 requires of every reasoning-layer output.

Speech is the lowest-priority feature in the system (R-05). This process
failing must never affect regulatory control, so it is a separate process
that publishes and nothing else; killing it changes nothing about the
control loop.
"""

from __future__ import annotations

import argparse
import logging
import queue
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, load_config
from src.common.device import resolve
from src.common.mqtt_client import Blackboard, build_transport
from src.reasoning.single_shot import PersonalContext, ReasoningUnavailableError
from src.speech.asr import Transcriber, UtteranceDetector
from src.speech.audio_capture import AudioCapture
from src.speech.pipeline import SpeechPipeline
from src.speech.wakeword import WakeWordDetector

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "speech-pipeline"

#: How long to wait for a frame before looping. Short enough that Ctrl-C is
#: responsive, long enough not to spin.
_FRAME_WAIT_S = 0.5


def build_pipeline(
    config: Config, clock: Clock, capture: AudioCapture
) -> SpeechPipeline:
    """Assemble the pipeline from configuration.

    Every model is constructed here rather than inside the pipeline, so the
    pipeline itself stays testable without a microphone, a GPU, or an
    inference server.
    """
    return SpeechPipeline(
        config=config.speech,
        clock=clock,
        capture=capture,
        detector=WakeWordDetector(config.speech),
        utterance=UtteranceDetector(config.speech),
        transcriber=Transcriber(config.speech),
        personal_context=PersonalContext(config.reasoning, clock),
    )


def _publish(blackboard: Blackboard, hint) -> None:
    blackboard.publish(topics.CONTEXT_PREFERENCE, hint)
    LOGGER.info("published preference hint: %s", hint.model_dump_json())


def run(
    config: Config,
    clock: Clock,
    capture: AudioCapture,
    pipeline: SpeechPipeline,
    blackboard: Blackboard,
    frames: int | None = None,
) -> int:
    """Pump audio frames through the pipeline until interrupted.

    :param frames: stop after this many frames. None runs forever; a count is
        what lets an integration test drive this without a microphone.
    """
    processed = 0
    while frames is None or processed < frames:
        try:
            frame = capture.frames.get(timeout=_FRAME_WAIT_S)
        except queue.Empty:
            continue

        processed += 1
        try:
            hint = pipeline.accept(frame)
        except ReasoningUnavailableError as exc:
            # The reasoning server is optional to this loop. Losing it costs a
            # preference hint, not the audio pipeline (FR-47).
            LOGGER.warning("reasoning unavailable, dropping utterance: %s", exc)
            continue

        if hint is not None:
            _publish(blackboard, hint)
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the speech pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--frames", type=int, default=None, help="Stop after N frames; default forever."
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(arguments.config)

    selection = resolve(config.speech.asr_device, config.speech.asr_compute_type)
    LOGGER.info("speech compute: %s/%s (%s)", selection.device, selection.compute_type, selection.reason)

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)

    capture = AudioCapture(config.speech, clock)
    pipeline = build_pipeline(config, clock, capture)

    blackboard.start()
    capture.start()
    LOGGER.info(
        "listening for %r; hints publish to %s",
        config.speech.wake_word,
        topics.CONTEXT_PREFERENCE.pattern,
    )
    try:
        run(config, clock, capture, pipeline, blackboard, frames=arguments.frames)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        capture.stop()
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
