"""Run the speech pipeline as a service.

``python -m src.speech``

Wires the microphone to the wake word, the endpointer and on-device
transcription, then publishes each transcript to ``space/context/utterance``.

What it publishes is *text*, not a request and not a command. Personal Context
runs in the reasoning process and answers it there (section 5.7.1); this
process is Layer 1 and does not import the reasoning layer. Nothing here can
move an actuator (FR-45), and a typed request on the same topic is
indistinguishable from a spoken one -- so everything speech can ask for can
be asked for without a microphone.

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
from src.common.schemas import Utterance, UtteranceSource
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
    pipeline itself stays testable without a microphone or a GPU.
    """
    return SpeechPipeline(
        config=config.speech,
        clock=clock,
        capture=capture,
        detector=WakeWordDetector(config.speech),
        utterance=UtteranceDetector(config.speech),
        transcriber=Transcriber(config.speech),
    )


def _publish(blackboard: Blackboard, clock: Clock, transcript: str) -> None:
    utterance = Utterance(
        ts=clock.now(), text=transcript[:2000], source=UtteranceSource.SPEECH
    )
    blackboard.publish(topics.CONTEXT_UTTERANCE, utterance)
    LOGGER.info("published utterance: %s", utterance.text)


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
        transcript = pipeline.accept(frame)
        if transcript:
            _publish(blackboard, clock, transcript)
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
        "listening for %r; transcripts publish to %s",
        config.speech.wake_word,
        topics.CONTEXT_UTTERANCE.pattern,
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
