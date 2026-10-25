"""Each mode, said to a person (S-14, FR-79)."""

from src.common.plain import PLAIN_MODE, mode_sentence
from src.common.schemas import Mode


def test_every_mode_has_a_sentence():
    assert set(PLAIN_MODE) == set(Mode)


def test_no_sentence_is_a_detector_code():
    assert not any("_" in sentence for sentence in PLAIN_MODE.values())


def test_a_hold_says_what_to_do_next():
    assert "press Reset" in mode_sentence(Mode.DEGRADED_ACTUATOR)


def test_normal_needs_no_next_step():
    assert mode_sentence(Mode.NORMAL) == PLAIN_MODE[Mode.NORMAL]
