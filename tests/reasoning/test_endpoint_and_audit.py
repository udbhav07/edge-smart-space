"""Unit tests for the shared endpoint and the reasoning audit (FR-46, FR-63)."""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import CallSite, ReasoningOutcome, ReasoningRecord
from src.reasoning.audit import ReasoningAudit, Trace
from src.reasoning.endpoint import ChatEndpoint, ReasoningUnavailableError
from tests.reasoning.fakes import ScriptedClient, calls, text


class RecordingTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []

    def connect(self, host, port, keepalive): ...
    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
    def subscribe(self, topic, qos): ...
    def loop_start(self): ...
    def loop_stop(self): ...
    def disconnect(self): ...

    def records(self) -> list[ReasoningRecord]:
        return [
            ReasoningRecord.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic == "space/audit/reasoning"
        ]


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


class TestEndpoint:
    def test_a_text_reply_is_returned_as_text(self, config):
        endpoint = ChatEndpoint(config.reasoning, SimClock(), ScriptedClient([text("hi")]))
        assert endpoint.complete([{"role": "user", "content": "x"}]).content == "hi"

    def test_tool_calls_keep_their_arguments_as_written(self, config):
        """Parsing is the caller's job: a malformed argument is a finding
        about the model, not something to smooth over here."""
        client = ScriptedClient([calls(("propose_setpoint", "{not json"))])
        completion = ChatEndpoint(config.reasoning, SimClock(), client).complete([])
        assert completion.tool_calls[0].arguments == "{not json"

    def test_token_counts_are_reported(self, config):
        client = ScriptedClient([text("hi", prompt_tokens=321, completion_tokens=7)])
        completion = ChatEndpoint(config.reasoning, SimClock(), client).complete([])
        assert (completion.prompt_tokens, completion.completion_tokens) == (321, 7)

    def test_a_server_that_reports_no_usage_counts_zero(self, config):
        reply = text("hi")
        reply.usage = None
        completion = ChatEndpoint(
            config.reasoning, SimClock(), ScriptedClient([reply])
        ).complete([])
        assert completion.prompt_tokens == 0

    def test_a_failing_server_is_unavailable_not_a_crash(self, config):
        client = ScriptedClient([ConnectionError("refused")])
        with pytest.raises(ReasoningUnavailableError):
            ChatEndpoint(config.reasoning, SimClock(), client).complete([])

    def test_a_reply_with_no_choices_is_unavailable(self, config):
        reply = text("hi")
        reply.choices = []
        with pytest.raises(ReasoningUnavailableError):
            ChatEndpoint(config.reasoning, SimClock(), ScriptedClient([reply])).complete([])

    def test_tools_are_offered_only_when_given(self, config):
        client = ScriptedClient([text("a")])
        ChatEndpoint(config.reasoning, SimClock(), client).complete([])
        assert "tools" not in client.requests[-1]

    def test_json_output_constrains_the_decode(self, config):
        client = ScriptedClient([text("{}")])
        ChatEndpoint(config.reasoning, SimClock(), client).complete([], json_output=True)
        assert client.requests[-1]["response_format"] == {"type": "json_object"}

    def test_decoding_is_deterministic_by_default(self, config):
        client = ScriptedClient([text("a")])
        ChatEndpoint(config.reasoning, SimClock(), client).complete([])
        assert client.requests[-1]["temperature"] == 0.0

    def test_a_tool_turn_replays_as_an_assistant_message(self, config):
        client = ScriptedClient([calls(("get_occupancy", {}))])
        message = ChatEndpoint(config.reasoning, SimClock(), client).complete([]).as_message()
        assert message["tool_calls"][0]["function"]["name"] == "get_occupancy"


class TestAudit:
    def _audit(self, config, max_chars=4000):
        transport = RecordingTransport()
        audit = ReasoningAudit(
            SimClock(), Blackboard(config.mqtt, transport), max_chars
        )
        return audit, transport

    def _completion(self, config, reply):
        return ChatEndpoint(
            config.reasoning, SimClock(), ScriptedClient([reply])
        ).complete([])

    def test_a_call_that_never_reached_the_model_is_recorded(self, config):
        """FR-46: every invocation, the failed ones included."""
        audit, transport = self._audit(config)
        audit.publish(
            Trace(CallSite.SUPERVISOR, "cadence", "decide"),
            ReasoningOutcome.UNAVAILABLE,
            reason="connection refused",
        )
        assert transport.records()[0].outcome is ReasoningOutcome.UNAVAILABLE

    def test_rounds_latency_and_tokens_accumulate(self, config):
        trace = Trace(CallSite.SUPERVISOR, "cadence", "decide")
        trace.add(self._completion(config, text("a", 100, 10)))
        trace.add(self._completion(config, text("b", 200, 20)))
        assert (trace.rounds, trace.prompt_tokens, trace.completion_tokens) == (
            2,
            300,
            30,
        )

    def test_the_tools_called_are_listed_in_order(self, config):
        trace = Trace(CallSite.SUPERVISOR, "cadence", "decide")
        trace.add(
            self._completion(
                config, calls(("get_thermal_state", {}), ("get_occupancy", {}))
            )
        )
        assert trace.tool_calls == ("get_thermal_state", "get_occupancy")

    def test_the_raw_output_includes_what_tools_were_asked_for(self, config):
        trace = Trace(CallSite.SUPERVISOR, "cadence", "decide")
        trace.add(self._completion(config, calls(("get_occupancy", {}))))
        assert "get_occupancy" in trace.raw_output

    def test_a_long_input_is_truncated_and_says_so(self, config):
        audit, transport = self._audit(config, max_chars=50)
        audit.publish(
            Trace(CallSite.PERSONAL_CONTEXT, "utterance", "x" * 500),
            ReasoningOutcome.NO_ACTION,
        )
        record = transport.records()[0]
        assert len(record.inputs) == 50 and record.inputs.endswith("[...]")

    def test_each_record_has_its_own_id(self, config):
        audit, transport = self._audit(config)
        for _ in range(2):
            audit.publish(
                Trace(CallSite.FAULT_DIAGNOSIS, "fault", ""), ReasoningOutcome.APPLIED
            )
        first, second = transport.records()
        assert first.invocation_id != second.invocation_id

    def test_the_audit_is_not_retained(self, config):
        """Append-only is the recorder's property; a retained record would be
        one record, overwritten."""
        audit, transport = self._audit(config)
        audit.publish(Trace(CallSite.SUPERVISOR, "cadence", ""), ReasoningOutcome.NO_ACTION)
        assert transport.published[-1][3] is False

    def test_a_bound_too_small_for_any_text_is_refused(self, config):
        with pytest.raises(ValueError):
            self._audit(config, max_chars=3)
