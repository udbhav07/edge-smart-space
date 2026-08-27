"""Unit tests for the Personal Context single-shot call.

Runs against a stub client, so no inference server is involved. What is
being tested is the part the design cares about: that the call has no tools,
and that post-decode validation actually discards bad output rather than
letting it through (FR-42, FR-44).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.clock import SimClock
from src.common.config import ReasoningConfig, load_config
from src.common.schemas import Comfort
from src.reasoning.single_shot import PersonalContext, ReasoningUnavailableError

TRANSCRIPT = "it's too warm in here, make it 23"


@pytest.fixture(name="config")
def _config() -> ReasoningConfig:
    return load_config(Path("config/default.yaml")).reasoning


class StubClient:
    """Answers with a fixed body, and records how it was called."""

    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _context(config, client) -> PersonalContext:
    return PersonalContext(config, SimClock(), client=client)


class TestNoTools:
    def test_the_call_is_never_given_tools(self, config):
        """Section 5.7.1 and FR-42: Personal Context has no tool loop."""
        client = StubClient('{"comfort": "cooler", "target_c": 23, "rationale": "x"}')
        _context(config, client).extract(TRANSCRIPT)
        assert "tools" not in client.calls[0]

    def test_the_decode_is_constrained_to_an_object(self, config):
        client = StubClient('{"comfort": "unchanged", "target_c": null, "rationale": ""}')
        _context(config, client).extract(TRANSCRIPT)
        assert client.calls[0]["response_format"] == {"type": "json_object"}

    def test_exactly_one_call_is_made(self, config):
        """Single-shot: no loop, no follow-up turn."""
        client = StubClient('{"comfort": "cooler", "target_c": null, "rationale": "x"}')
        _context(config, client).extract(TRANSCRIPT)
        assert len(client.calls) == 1

    def test_no_history_is_carried_between_utterances(self, config):
        client = StubClient('{"comfort": "cooler", "target_c": null, "rationale": "x"}')
        context = _context(config, client)
        context.extract(TRANSCRIPT)
        context.extract(TRANSCRIPT)
        assert client.calls[0]["messages"] == client.calls[1]["messages"]


class TestExtraction:
    def test_a_well_formed_preference_is_returned(self, config):
        client = StubClient(
            '{"comfort": "cooler", "target_c": 23.0, "rationale": "too warm"}'
        )
        hint = _context(config, client).extract(TRANSCRIPT)
        assert hint.comfort is Comfort.COOLER
        assert hint.target_c == 23.0

    def test_the_hint_is_timestamped_from_the_injected_clock(self, config):
        clock = SimClock()
        clock.advance(500.0)
        client = StubClient('{"comfort": "warmer", "target_c": null, "rationale": ""}')
        hint = PersonalContext(config, clock, client=client).extract(TRANSCRIPT)
        assert hint.ts == clock.now()

    def test_an_empty_transcript_makes_no_call_at_all(self, config):
        client = StubClient()
        assert _context(config, client).extract("   ") is None
        assert client.calls == []

    def test_no_preference_expressed_yields_nothing(self, config):
        client = StubClient('{"comfort": "unchanged", "target_c": null, "rationale": ""}')
        assert _context(config, client).extract("what time is it") is None


class TestPostDecodeValidation:
    """FR-44: output failing semantic validation is discarded."""

    @pytest.mark.parametrize(
        "body",
        [
            "not json at all",
            "[1, 2, 3]",
            '{"comfort": "freezing", "target_c": null, "rationale": ""}',
            '{"target_c": 23.0, "rationale": "no comfort key"}',
            '{"comfort": "cooler", "target_c": "warm", "rationale": ""}',
        ],
        ids=[
            "unparseable",
            "not-an-object",
            "comfort-outside-the-enum",
            "missing-comfort",
            "non-numeric-target",
        ],
    )
    def test_malformed_output_is_discarded_rather_than_published(self, config, body):
        assert _context(config, StubClient(body)).extract(TRANSCRIPT) is None

    def test_a_schema_valid_output_is_still_checked_semantically(self, config):
        """Parsing is guaranteed by construction and is not a result."""
        client = StubClient('{"comfort": "unchanged", "target_c": null, "rationale": "x"}')
        assert _context(config, client).extract(TRANSCRIPT) is None


class TestDegradation:
    def test_an_unreachable_server_raises_a_specific_error(self, config):
        client = StubClient(error=ConnectionError("refused"))
        with pytest.raises(ReasoningUnavailableError):
            _context(config, client).extract(TRANSCRIPT)

    def test_the_underlying_failure_is_preserved_for_diagnosis(self, config):
        client = StubClient(error=ConnectionError("refused"))
        with pytest.raises(ReasoningUnavailableError) as caught:
            _context(config, client).extract(TRANSCRIPT)
        assert isinstance(caught.value.__cause__, ConnectionError)
