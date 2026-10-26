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
from src.common.conversation import Conversations
from src.common.device import resolve
from src.common.mqtt_client import Blackboard, build_transport
from src.common.schemas import AmbientUtterance, AssistantReply, ModeState, SpeechActivity, SpeechSettings
from src.reasoning.devices import KnownDevices
from src.reasoning.single_shot import PersonalContext, ReasoningUnavailableError
from src.speech.asr import Transcriber, UtteranceDetector
from src.speech.audio_capture import AudioCapture
from src.speech.pipeline import SpeechPipeline
from src.speech.speaker_profile import SpeakerIdentifier
from src.speech.status import StatusPublisher
from src.speech.voice import Voice, build_speaker
from src.speech.wakeword import WakeWordDetector

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "speech-pipeline"

#: How long to wait for a frame before looping. Short enough that Ctrl-C is
#: responsive, long enough not to spin.
_FRAME_WAIT_S = 0.5


def build_pipeline(
    config: Config,
    clock: Clock,
    capture: AudioCapture,
    conversations: Conversations | None = None,
    on_activity=None,
    devices: KnownDevices | None = None,
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
        personal_context=PersonalContext(config.reasoning, clock, devices=devices),
        speaker=SpeakerIdentifier(config.speech),
        conversations=conversations,
        on_activity=on_activity,
    )


def _publish(blackboard: Blackboard, heard) -> None:
    """An addressed request is a hint (FR-53); anything else was overheard (FR-76)."""
    if isinstance(heard, AmbientUtterance):
        blackboard.publish(topics.CONTEXT_AMBIENT, heard)
        LOGGER.info("published overheard speech (%d words)", len(heard.transcript.split()))
        return
    blackboard.publish(topics.CONTEXT_PREFERENCE, heard)
    LOGGER.info("published preference hint: %s", heard.model_dump_json())


def run(
    config: Config,
    clock: Clock,
    capture: AudioCapture,
    pipeline: SpeechPipeline,
    blackboard: Blackboard,
    frames: int | None = None,
    status: StatusPublisher | None = None,
    voice: Voice | None = None,
) -> int:
    """Pump audio frames through the pipeline until interrupted.

    :param frames: stop after this many frames. None runs forever; a count is
        what lets an integration test drive this without a microphone.
    """
    processed = 0
    if status is not None:
        status.changed(status.activity)
    while frames is None or processed < frames:
        if status is not None:
            status.heartbeat()
        try:
            frame = capture.frames.get(timeout=_FRAME_WAIT_S)
        except queue.Empty:
            continue

        processed += 1
        held = _held(voice)
        if held is not None:
            # Speaking, or muted: the frame is dropped unheard (FR-79, FR-82).
            pipeline.pause()
            if status is not None and status.activity is not held:
                status.changed(held)
            continue
        if status is not None and status.activity in (SpeechActivity.SPEAKING, SpeechActivity.MUTED):
            status.changed(SpeechActivity.WAITING)
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


def _held(voice: Voice | None) -> SpeechActivity | None:
    """Why the microphone must be ignored right now, if it must."""
    if voice is None:
        return None
    if not voice.settings.listening:
        return SpeechActivity.MUTED
    if voice.busy:
        return SpeechActivity.SPEAKING
    return None


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
    conversations = Conversations()
    conversations.subscribe(blackboard)
    voice = Voice(
        build_speaker(config.speech),
        clock,
        utc_offset_h=config.site.utc_offset_h,
        tail_s=config.speech.speaking_tail_ms / 1000.0,
    )
    # Replies, the occupant's settings and changes of mode reach the voice;
    # it says what they ask for on a thread of its own (FR-79).
    blackboard.subscribe(topics.CONTEXT_REPLY, AssistantReply, voice.on_reply)
    blackboard.subscribe(topics.SPEECH_SETTINGS, SpeechSettings, voice.on_settings)
    blackboard.subscribe(topics.SYSTEM_MODE, ModeState, voice.on_mode)
    voice.start()
    status = StatusPublisher(config.speech, clock, blackboard, capture.microphone, voice=voice.engine)
    # The devices in the room, so "turn on the fan" is filed as being about the fan (FR-89).
    devices = KnownDevices()
    devices.subscribe(blackboard)
    pipeline = build_pipeline(config, clock, capture, conversations, on_activity=status.changed, devices=devices)

    blackboard.start()
    capture.start()
    LOGGER.info(
        "listening for %r (%s); hints publish to %s, overheard speech to %s",
        config.speech.wake_word,
        config.speech.listen_mode.value,
        topics.CONTEXT_PREFERENCE.pattern,
        topics.CONTEXT_AMBIENT.pattern,
    )
    try:
        run(config, clock, capture, pipeline, blackboard, frames=arguments.frames, status=status, voice=voice)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        capture.stop()
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
