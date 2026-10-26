"""Unit tests for recognising a voice, for personalisation only (FR-52).

Run on synthetic voices (``tests/speech/voices.py``), which test the
mechanism -- enrolment, matching, refusing a stranger, storing no audio --
and make no claim about accuracy on real people.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.speech.speaker_profile import SpeakerIdentifier, SpectralEmbedder, VoiceprintError
from tests.speech.voices import ANU, DEV, RAVI, SAMPLE_RATE_HZ, STRANGER, utterance

ENROLMENT_SEEDS = range(4)


@pytest.fixture(name="config")
def _config(tmp_path):
    speech = load_config(Path("config/default.yaml")).speech
    return speech.model_copy(update={"voiceprints_path": str(tmp_path / "voiceprints.json")})


@pytest.fixture(name="household", scope="module")
def _household(tmp_path_factory):
    """Three enrolled voices; module-scoped, since synthesis is the slow part."""
    speech = load_config(Path("config/default.yaml")).speech
    path = tmp_path_factory.mktemp("voices") / "voiceprints.json"
    identifier = SpeakerIdentifier(speech.model_copy(update={"voiceprints_path": str(path)}))
    for name, voice in {"ravi": RAVI, "anu": ANU, "dev": DEV}.items():
        for seed in ENROLMENT_SEEDS:
            identifier.enroll(name, utterance(*voice, seed))
    return identifier


class TestFeatures:
    def test_the_same_audio_gives_the_same_vector(self):
        audio = utterance(*RAVI, seed=1)
        embedder = SpectralEmbedder()
        assert np.array_equal(
            embedder.embed(audio, SAMPLE_RATE_HZ), embedder.embed(audio, SAMPLE_RATE_HZ)
        )

    def test_silence_has_no_voice_to_judge(self):
        silence = np.zeros(SAMPLE_RATE_HZ, dtype=np.float32)
        assert SpectralEmbedder().embed(silence, SAMPLE_RATE_HZ) is None

    def test_too_short_a_clip_has_no_voice_to_judge(self):
        assert SpectralEmbedder().embed(np.zeros(100, dtype=np.float32), SAMPLE_RATE_HZ) is None


class TestRecognition:
    @pytest.mark.parametrize("name,voice", [("ravi", RAVI), ("anu", ANU), ("dev", DEV)])
    def test_an_enrolled_voice_is_recognised(self, household, name, voice):
        recognised = [
            household.identify(utterance(*voice, seed)).profile_id for seed in range(300, 306)
        ]
        assert recognised.count(name) >= 4

    @pytest.mark.parametrize("voice", [RAVI, ANU, DEV])
    def test_nobody_is_taken_for_somebody_else(self, household, voice):
        """The costly error: a wrong name, and that person's memory."""
        names = {household.identify(utterance(*voice, seed)).profile_id for seed in range(300, 306)}
        assert len(names - {""}) <= 1

    def test_a_stranger_is_mostly_a_guest(self, household):
        guests = [
            household.identify(utterance(*STRANGER, seed)).profile_id == ""
            for seed in range(300, 310)
        ]
        assert sum(guests) >= 8

    def test_a_short_utterance_is_a_guest(self, household):
        assert not household.identify(utterance(*RAVI, seed=5, seconds=0.5)).judged

    def test_with_nobody_enrolled_everyone_is_a_guest(self, config):
        assert SpeakerIdentifier(config).identify(utterance(*RAVI, seed=1)).profile_id == ""


class TestEnrolment:
    def test_one_utterance_is_not_enough_to_be_recognised(self, config):
        """One utterance has no spread to measure a distance in."""
        identifier = SpeakerIdentifier(config)
        identifier.enroll("ravi", utterance(*RAVI, seed=0))
        assert not identifier.matchable("ravi")
        assert identifier.identify(utterance(*RAVI, seed=0)).profile_id == ""

    def test_enough_utterances_make_a_voice_matchable(self, config):
        identifier = SpeakerIdentifier(config)
        for seed in range(config.speaker_min_enrolment_utterances):
            identifier.enroll("ravi", utterance(*RAVI, seed))
        assert identifier.matchable("ravi")

    def test_an_unsafe_id_is_refused(self, config):
        with pytest.raises(VoiceprintError):
            SpeakerIdentifier(config).enroll("../x", utterance(*RAVI, seed=0))

    def test_silence_cannot_be_enrolled(self, config):
        with pytest.raises(VoiceprintError):
            SpeakerIdentifier(config).enroll(
                "ravi", np.zeros(2 * SAMPLE_RATE_HZ, dtype=np.float32)
            )

    def test_only_statistics_are_stored_never_audio(self, config):
        """FR-51: a voiceprint is a few dozen numbers, not a recording."""
        identifier = SpeakerIdentifier(config)
        audio = utterance(*RAVI, seed=0)
        identifier.enroll("ravi", audio)
        stored = json.loads(Path(config.voiceprints_path).read_text(encoding="utf-8"))
        entry = stored["voiceprints"]["ravi"]
        assert set(entry) == {"utterances", "mean", "squares"}
        assert len(entry["mean"]) < 100 < len(audio)

    def test_an_enrolment_by_another_process_is_picked_up(self, config):
        listening = SpeakerIdentifier(config)
        assert listening.enrolled == {}
        enrolling = SpeakerIdentifier(config)
        enrolling.enroll("ravi", utterance(*RAVI, seed=0))
        assert listening.enrolled == {"ravi": 1}

    def test_a_voiceprint_can_be_removed(self, config):
        identifier = SpeakerIdentifier(config)
        identifier.enroll("ravi", utterance(*RAVI, seed=0))
        assert identifier.remove("ravi") and identifier.enrolled == {}

    def test_an_unreadable_file_makes_everyone_a_guest(self, config):
        Path(config.voiceprints_path).write_text("{broken", encoding="utf-8")
        assert SpeakerIdentifier(config).identify(utterance(*RAVI, seed=0)).profile_id == ""


def test_a_fresh_identifier_knows_who_is_matchable(config):
    """Loaded from disk on first question, not only after an identify."""
    enrolling = SpeakerIdentifier(config)
    for seed in range(config.speaker_min_enrolment_utterances):
        enrolling.enroll("ravi", utterance(*RAVI, seed))
    assert SpeakerIdentifier(config).matchable("ravi")
