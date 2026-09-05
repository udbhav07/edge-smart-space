"""Unit tests for the tool-calling contract (DESIGN.md section 5.7.6).

Three properties carry the design and are tested hardest:

* A model is shown a declaration and nothing more -- no provider, no
  transport, no confirmation flow (FR-70, FR-71).
* Adding a tool, or swapping the implementation behind one, changes nothing
  else (FR-72).
* The model runs its own tools, except the ones that commit the occupant to
  an outside party; and nothing at all runs on arguments that do not satisfy
  the declaration (FR-73, FR-74).

The registry is tested through fake providers. A real one is a calendar
client, which belongs nowhere near ``src/common/``.
"""

from collections.abc import Mapping

import pytest

from src.common.clock import SimClock
from src.common.tools import (
    ASSISTANCE_TOOLS,
    BOOK_TRAVEL,
    GET_EVENTS,
    SCHEDULE_EVENT,
    ArgumentValue,
    ParameterType,
    ProviderOutcome,
    ToolArgumentError,
    ToolCatalogue,
    ToolEffect,
    ToolInvocation,
    ToolParameter,
    ToolRegistry,
    ToolRequester,
    ToolResult,
    ToolSpec,
    ToolStatus,
    UnknownToolError,
)

WINDOW_S = 300.0


class FakeProvider:
    """Stands in for whatever actually does the thing.

    Records what it was asked so a test can assert the registry hands over
    validated arguments and not the raw ones.
    """

    def __init__(self, *, name: str = "fake_calendar", simulated: bool = False) -> None:
        self._name = name
        self._simulated = simulated
        self.calls: list[tuple[str, Mapping[str, ArgumentValue]]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def simulated(self) -> bool:
        return self._simulated

    def invoke(
        self, tool: str, arguments: Mapping[str, ArgumentValue]
    ) -> ProviderOutcome:
        self.calls.append((tool, dict(arguments)))
        return ProviderOutcome(message="Done.", detail={"event_id": "e_1"})


class BrokenProvider:
    """A provider that fails the way a real one does: at the last moment."""

    name = "broken_calendar"
    simulated = False

    def invoke(
        self, tool: str, arguments: Mapping[str, ArgumentValue]
    ) -> ProviderOutcome:
        raise ConnectionError("calendar unreachable")


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="registry")
def _registry(clock) -> ToolRegistry:
    return ToolRegistry(clock=clock)


def _invocation(clock, tool: str = "schedule_event", **arguments) -> ToolInvocation:
    return ToolInvocation(
        ts=clock.now(),
        invocation_id="inv_1",
        tool=tool,
        arguments=arguments
        or {"starts_at": "2026-09-04T15:00:00", "subject": "review meeting"},
        requester=ToolRequester.PERSONAL_CONTEXT,
        rationale="put the review in my calendar",
        expires_ts=clock.now() + WINDOW_S,
    )


class TestWhatTheModelIsShown:
    """FR-70: the declaration is the whole instruction."""

    def test_a_tool_renders_as_a_function_schema(self):
        schema = SCHEDULE_EVENT.as_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "schedule_event"

    def test_the_purpose_becomes_the_description(self):
        assert "calendar" in SCHEDULE_EVENT.as_schema()["function"]["description"]

    def test_only_required_parameters_are_listed_as_required(self):
        required = SCHEDULE_EVENT.as_schema()["function"]["parameters"]["required"]
        assert required == ["starts_at", "subject"]

    def test_a_timestamp_is_offered_as_a_formatted_string(self):
        """A model can write an ISO-8601 datetime; it cannot write an epoch."""
        properties = SCHEDULE_EVENT.as_schema()["function"]["parameters"]["properties"]
        assert properties["starts_at"] == {
            "type": "string",
            "format": "date-time",
            "description": SCHEDULE_EVENT.parameters[0].description,
        }

    def test_choices_reach_the_schema_as_an_enum(self):
        properties = BOOK_TRAVEL.as_schema()["function"]["parameters"]["properties"]
        assert properties["kind"]["enum"] == ["flight", "hotel"]

    def test_an_undeclared_argument_is_refused_by_the_schema_too(self):
        parameters = SCHEDULE_EVENT.as_schema()["function"]["parameters"]
        assert parameters["additionalProperties"] is False

    def test_nothing_about_the_provider_appears_anywhere_in_the_surface(self):
        """FR-71: the model must not be able to tell which provider fulfils
        a tool, because then a provider swap would change its behaviour."""
        rendered = repr(ToolRegistry(clock=SimClock()).schemas())
        for leak in ("provider", "google", "microsoft", "local_calendar", "mqtt"):
            assert leak not in rendered.lower()

    def test_the_registry_offers_every_declared_tool(self, registry):
        assert len(registry.schemas()) == len(ASSISTANCE_TOOLS)


class TestArgumentValidation:
    """FR-73: refused before a provider is reached, with the reason."""

    def test_a_well_formed_call_is_accepted(self):
        accepted = SCHEDULE_EVENT.validate_arguments(
            {"starts_at": "2026-09-04T15:00:00", "subject": "standup"}
        )
        assert accepted["subject"] == "standup"

    def test_a_missing_required_argument_is_refused(self):
        with pytest.raises(ToolArgumentError, match="starts_at"):
            SCHEDULE_EVENT.validate_arguments({"subject": "standup"})

    def test_a_missing_optional_argument_is_simply_absent(self):
        accepted = SCHEDULE_EVENT.validate_arguments(
            {"starts_at": "2026-09-04T15:00:00", "subject": "standup"}
        )
        assert "duration_min" not in accepted

    def test_an_explicit_null_optional_is_treated_as_absent(self):
        """A model that fills every key with something writes null for the
        ones it has nothing for."""
        accepted = SCHEDULE_EVENT.validate_arguments(
            {
                "starts_at": "2026-09-04T15:00:00",
                "subject": "standup",
                "duration_min": None,
            }
        )
        assert "duration_min" not in accepted

    def test_an_explicit_null_required_argument_is_still_missing(self):
        with pytest.raises(ToolArgumentError, match="subject"):
            SCHEDULE_EVENT.validate_arguments(
                {"starts_at": "2026-09-04T15:00:00", "subject": None}
            )

    def test_an_unknown_argument_is_refused_rather_than_dropped(self):
        """Silently ignoring it is a tool that quietly did the wrong thing."""
        with pytest.raises(ToolArgumentError, match="attendees"):
            SCHEDULE_EVENT.validate_arguments(
                {
                    "starts_at": "2026-09-04T15:00:00",
                    "subject": "standup",
                    "attendees": "moksh",
                }
            )

    def test_an_unparseable_timestamp_is_refused(self):
        with pytest.raises(ToolArgumentError, match="ISO-8601"):
            SCHEDULE_EVENT.validate_arguments(
                {"starts_at": "next tuesday afternoon", "subject": "standup"}
            )

    def test_an_empty_string_is_not_a_subject(self):
        with pytest.raises(ToolArgumentError, match="subject"):
            SCHEDULE_EVENT.validate_arguments(
                {"starts_at": "2026-09-04T15:00:00", "subject": ""}
            )

    def test_a_value_outside_the_declared_choices_is_refused(self):
        with pytest.raises(ToolArgumentError, match="kind"):
            BOOK_TRAVEL.validate_arguments(
                {
                    "kind": "train",
                    "destination": "Hyderabad",
                    "depart_on": "2026-09-10T08:00:00",
                }
            )

    def test_an_integer_arriving_as_a_whole_float_is_accepted(self):
        """JSON has one number type, so 60 arrives as 60.0."""
        accepted = SCHEDULE_EVENT.validate_arguments(
            {
                "starts_at": "2026-09-04T15:00:00",
                "subject": "standup",
                "duration_min": 60.0,
            }
        )
        assert accepted["duration_min"] == 60

    def test_a_fractional_value_is_not_a_whole_number(self):
        with pytest.raises(ToolArgumentError, match="duration_min"):
            SCHEDULE_EVENT.validate_arguments(
                {
                    "starts_at": "2026-09-04T15:00:00",
                    "subject": "standup",
                    "duration_min": 12.5,
                }
            )

    def test_a_boolean_does_not_pass_as_a_number(self):
        """True is an int in Python and would otherwise slip through."""
        spec = ToolSpec(
            name="dim",
            purpose="Set a level.",
            effect=ToolEffect.WRITE,
            parameters=(
                ToolParameter(
                    name="level", type=ParameterType.NUMBER, description="0 to 1."
                ),
            ),
        )
        with pytest.raises(ToolArgumentError, match="boolean"):
            spec.validate_arguments({"level": True})

    def test_accepted_arguments_cannot_be_altered_afterwards(self):
        accepted = SCHEDULE_EVENT.validate_arguments(
            {"starts_at": "2026-09-04T15:00:00", "subject": "standup"}
        )
        with pytest.raises(TypeError):
            accepted["subject"] = "something else"


class TestDeclaration:
    def test_a_commit_tool_requires_confirmation(self):
        """Booking a flight is not ours to undo (FR-54)."""
        assert BOOK_TRAVEL.requires_confirmation

    def test_a_write_tool_does_not(self):
        """A calendar entry is the occupant's own and can be deleted again."""
        assert not SCHEDULE_EVENT.requires_confirmation

    def test_a_read_tool_does_not_either(self):
        assert not GET_EVENTS.requires_confirmation

    def test_only_a_commit_tool_asks(self):
        """The whole gate, stated once: prompting for everything trains
        someone to approve without reading."""
        for spec in ASSISTANCE_TOOLS:
            assert spec.requires_confirmation == (spec.effect is ToolEffect.COMMIT)

    def test_a_tool_cannot_declare_one_parameter_twice(self):
        parameter = ToolParameter(
            name="subject", type=ParameterType.STRING, description="Title."
        )
        with pytest.raises(ValueError, match="twice"):
            ToolSpec(
                name="dup",
                purpose="Two of the same.",
                effect=ToolEffect.READ,
                parameters=(parameter, parameter),
            )

    def test_choices_on_a_non_string_parameter_are_refused(self):
        with pytest.raises(ValueError, match="choices"):
            ToolParameter(
                name="nights",
                type=ParameterType.INTEGER,
                description="How many.",
                choices=("1", "2"),
            )

    def test_a_tool_name_must_be_a_plain_identifier(self):
        with pytest.raises(ValueError):
            ToolSpec(name="Book Travel", purpose="No.", effect=ToolEffect.WRITE)

    def test_a_spec_is_immutable(self):
        with pytest.raises(ValueError):
            SCHEDULE_EVENT.name = "something_else"


def _booking(clock) -> ToolInvocation:
    return _invocation(
        clock,
        "book_travel",
        kind="flight",
        destination="Hyderabad",
        depart_on="2026-09-10T08:00:00",
    )


class TestConfirmationGate:
    """FR-74. The gate is decided on the declared effect, not on the caller."""

    def test_a_commit_tool_is_not_run_without_confirmation(self, registry, clock):
        provider = FakeProvider(name="mock_travel", simulated=True)
        registry.bind("book_travel", provider)
        result = registry.invoke(_booking(clock), confirmed=False)
        assert result.status is ToolStatus.CONFIRMATION_REQUIRED
        assert provider.calls == []

    def test_the_same_booking_runs_once_confirmed(self, registry, clock):
        registry.bind("book_travel", FakeProvider(name="mock_travel", simulated=True))
        assert registry.invoke(_booking(clock), confirmed=True).status is ToolStatus.OK

    def test_a_write_tool_runs_without_confirmation(self, registry, clock):
        """The model puts the entry in the calendar itself. It is the
        occupant's own calendar and the entry can be removed again."""
        registry.bind("schedule_event", FakeProvider())
        assert registry.invoke(_invocation(clock), confirmed=False).status is (
            ToolStatus.OK
        )

    def test_a_read_tool_runs_without_confirmation(self, registry, clock):
        registry.bind("get_events", FakeProvider())
        invocation = _invocation(
            clock,
            "get_events",
            from_time="2026-09-04T00:00:00",
            to_time="2026-09-05T00:00:00",
        )
        assert registry.invoke(invocation, confirmed=False).status is ToolStatus.OK

    def test_bad_arguments_are_refused_before_confirmation_is_asked_for(
        self, registry, clock
    ):
        """Nobody should be asked to approve a malformed request."""
        registry.bind("book_travel", FakeProvider())
        invocation = _invocation(clock, "book_travel", kind="flight")
        result = registry.invoke(invocation, confirmed=False)
        assert result.status is ToolStatus.BAD_ARGUMENTS

    def test_an_expired_booking_is_refused_even_when_confirmed(self, registry, clock):
        invocation = _booking(clock)
        clock.advance(WINDOW_S + 1.0)
        assert registry.invoke(invocation, confirmed=True).status is ToolStatus.EXPIRED

    def test_a_booking_confirmed_inside_the_window_still_runs(self, registry, clock):
        registry.bind("book_travel", FakeProvider(name="mock_travel", simulated=True))
        invocation = _booking(clock)
        clock.advance(WINDOW_S - 1.0)
        assert registry.invoke(invocation, confirmed=True).status is ToolStatus.OK

    def test_a_new_commit_tool_is_gated_without_anyone_wiring_it(self, registry, clock):
        """The gate follows the declaration, so a tool added later is covered
        by having declared what it does."""
        registry.declare(
            ToolSpec(
                name="buy_tickets",
                purpose="Buy concert tickets.",
                effect=ToolEffect.COMMIT,
                parameters=(
                    ToolParameter(
                        name="event",
                        type=ParameterType.STRING,
                        description="Which one.",
                    ),
                ),
            )
        )
        registry.bind("buy_tickets", FakeProvider(name="mock_tickets", simulated=True))
        result = registry.invoke(
            _invocation(clock, "buy_tickets", event="something"), confirmed=False
        )
        assert result.status is ToolStatus.CONFIRMATION_REQUIRED


class TestInvocationOutcomes:
    """FR-75: every outcome is a result, never an exception nobody sees."""

    def test_an_unknown_tool_is_a_result_not_a_crash(self, registry, clock):
        result = registry.invoke(
            _invocation(clock, "launch_rocket", target="mars"), confirmed=True
        )
        assert result.status is ToolStatus.UNKNOWN_TOOL

    def test_a_declared_tool_with_nothing_bound_reports_unavailable(
        self, registry, clock
    ):
        result = registry.invoke(_invocation(clock), confirmed=True)
        assert result.status is ToolStatus.UNAVAILABLE

    def test_a_provider_failure_is_contained(self, registry, clock):
        """A broken calendar must not take the process down with it."""
        registry.bind("schedule_event", BrokenProvider())
        result = registry.invoke(_invocation(clock), confirmed=True)
        assert result.status is ToolStatus.FAILED

    def test_a_failure_says_what_went_wrong(self, registry, clock):
        registry.bind("schedule_event", BrokenProvider())
        result = registry.invoke(_invocation(clock), confirmed=True)
        assert "unreachable" in result.message

    def test_a_refusal_names_no_provider(self, registry, clock):
        result = registry.invoke(_invocation(clock), confirmed=False)
        assert result.provider == ""

    def test_every_result_carries_the_invocation_it_answers(self, registry, clock):
        registry.bind("schedule_event", FakeProvider())
        result = registry.invoke(_invocation(clock), confirmed=True)
        assert result.invocation_id == "inv_1"

    def test_a_result_is_stamped_from_the_injected_clock(self, registry, clock):
        registry.bind("schedule_event", FakeProvider())
        clock.advance(7.0)
        result = registry.invoke(_invocation(clock), confirmed=True)
        assert result.ts == clock.now()

    def test_the_provider_receives_validated_arguments(self, registry, clock):
        provider = FakeProvider()
        registry.bind("schedule_event", provider)
        registry.invoke(
            _invocation(
                clock,
                "schedule_event",
                starts_at="2026-09-04T15:00:00",
                subject="standup",
                duration_min=30.0,
            ),
            confirmed=True,
        )
        _, arguments = provider.calls[0]
        assert arguments["duration_min"] == 30

    def test_a_mock_provider_marks_its_result_as_simulated(self, registry, clock):
        """FR-55: a mock booking is identifiable as one without anybody
        remembering to write the word."""
        registry.bind("book_travel", FakeProvider(name="mock_travel", simulated=True))
        invocation = _invocation(
            clock,
            "book_travel",
            kind="flight",
            destination="Hyderabad",
            depart_on="2026-09-10T08:00:00",
        )
        result = registry.invoke(invocation, confirmed=True)
        assert result.simulated and result.provider == "mock_travel"

    def test_a_real_provider_does_not(self, registry, clock):
        registry.bind("schedule_event", FakeProvider())
        result = registry.invoke(_invocation(clock), confirmed=True)
        assert not result.simulated


class TestExtensibility:
    """FR-72, made testable: a new tool is one spec and one binding."""

    def test_a_tool_the_system_has_never_heard_of_can_be_added_and_used(
        self, registry, clock
    ):
        registry.declare(
            ToolSpec(
                name="order_groceries",
                purpose="Order a grocery delivery.",
                effect=ToolEffect.COMMIT,
                parameters=(
                    ToolParameter(
                        name="items",
                        type=ParameterType.STRING,
                        description="Comma-separated list.",
                    ),
                ),
            )
        )
        registry.bind("order_groceries", FakeProvider(name="mock_grocer"))
        result = registry.invoke(
            _invocation(clock, "order_groceries", items="milk, coffee"), confirmed=True
        )
        assert result.status is ToolStatus.OK

    def test_a_new_tool_declaring_commit_needs_no_extra_wiring_to_be_gated(
        self, registry, clock
    ):
        registry.declare(
            ToolSpec(
                name="hire_a_car",
                purpose="Hire a car.",
                effect=ToolEffect.COMMIT,
                parameters=(
                    ToolParameter(
                        name="city",
                        type=ParameterType.STRING,
                        description="Where to collect it.",
                    ),
                ),
            )
        )
        registry.bind("hire_a_car", FakeProvider(name="mock_cars", simulated=True))
        result = registry.invoke(
            _invocation(clock, "hire_a_car", city="Hyderabad"), confirmed=False
        )
        assert result.status is ToolStatus.CONFIRMATION_REQUIRED

    def test_a_new_tool_appears_in_what_the_model_is_shown(self, registry):
        before = len(registry.schemas())
        registry.declare(
            ToolSpec(name="lock_door", purpose="Lock it.", effect=ToolEffect.WRITE)
        )
        assert len(registry.schemas()) == before + 1

    def test_swapping_the_provider_changes_nothing_the_caller_does(
        self, registry, clock
    ):
        """The same invocation, fulfilled by a different implementation."""
        registry.bind("schedule_event", FakeProvider(name="local_calendar"))
        first = registry.invoke(_invocation(clock), confirmed=True)
        registry.bind("schedule_event", FakeProvider(name="hosted_calendar"))
        second = registry.invoke(_invocation(clock), confirmed=True)
        assert (first.status, second.status) == (ToolStatus.OK, ToolStatus.OK)
        assert (first.provider, second.provider) == (
            "local_calendar",
            "hosted_calendar",
        )

    def test_declaring_one_name_twice_is_refused(self, registry):
        with pytest.raises(ValueError, match="already declared"):
            registry.declare(SCHEDULE_EVENT)

    def test_binding_to_an_undeclared_tool_is_refused(self, registry):
        """Otherwise it sits there unreachable and looks like a broken
        provider rather than a wiring mistake."""
        with pytest.raises(UnknownToolError):
            registry.bind("feed_the_cat", FakeProvider())

    def test_asking_for_an_undeclared_spec_names_what_is_available(self, registry):
        with pytest.raises(UnknownToolError, match="schedule_event"):
            registry.spec("feed_the_cat")


class TestCatalogue:
    def test_the_catalogue_carries_every_declared_tool(self, registry):
        assert len(registry.catalogue().tools) == len(ASSISTANCE_TOOLS)

    def test_the_catalogue_is_stamped_from_the_injected_clock(self, registry, clock):
        clock.advance(3.0)
        assert registry.catalogue().ts == clock.now()

    def test_the_catalogue_survives_a_round_trip_through_json(self, registry):
        catalogue = registry.catalogue()
        assert ToolCatalogue.model_validate_json(catalogue.model_dump_json()) == (
            catalogue
        )

    def test_a_catalogue_declaring_one_tool_twice_is_refused(self, clock):
        with pytest.raises(ValueError, match="twice"):
            ToolCatalogue(ts=clock.now(), tools=(SCHEDULE_EVENT, SCHEDULE_EVENT))


class TestMessages:
    def test_an_invocation_is_immutable(self, clock):
        invocation = _invocation(clock)
        with pytest.raises(ValueError):
            invocation.tool = "book_travel"

    def test_an_invocations_arguments_are_immutable(self, clock):
        invocation = _invocation(clock)
        with pytest.raises(TypeError):
            invocation.arguments["subject"] = "something else"

    def test_an_invocation_survives_a_round_trip_through_json(self, clock):
        invocation = _invocation(clock)
        restored = ToolInvocation.model_validate_json(invocation.model_dump_json())
        assert restored == invocation

    def test_an_unexpected_field_on_an_invocation_is_refused(self, clock):
        with pytest.raises(ValueError):
            ToolInvocation(
                ts=clock.now(),
                invocation_id="inv_1",
                tool="get_events",
                requester=ToolRequester.CONSOLE,
                expires_ts=clock.now() + WINDOW_S,
                confirmed=True,
            )

    def test_a_successful_result_must_name_a_provider(self, clock):
        """Otherwise it is a claim that something happened with nothing
        behind it."""
        with pytest.raises(ValueError, match="provider"):
            ToolResult(
                ts=clock.now(),
                invocation_id="inv_1",
                tool="schedule_event",
                status=ToolStatus.OK,
            )

    def test_a_simulated_result_must_name_the_mock_that_answered(self, clock):
        with pytest.raises(ValueError, match="provider"):
            ToolResult(
                ts=clock.now(),
                invocation_id="inv_1",
                tool="book_travel",
                status=ToolStatus.FAILED,
                simulated=True,
            )

    def test_a_result_survives_a_round_trip_through_json(self, clock):
        result = ToolResult(
            ts=clock.now(),
            invocation_id="inv_1",
            tool="schedule_event",
            status=ToolStatus.OK,
            message="Added it.",
            detail={"event_id": "e_1"},
            provider="local_calendar",
        )
        assert ToolResult.model_validate_json(result.model_dump_json()) == result


class TestNoAuthority:
    """The registry decides whether a tool runs. It publishes nothing and
    controls nothing, so it cannot become a second path to the plant."""

    def test_it_holds_no_transport(self, registry):
        assert not any(
            name in vars(registry) for name in ("_transport", "_blackboard", "_client")
        )

    def test_no_declared_tool_can_touch_the_plant(self, registry):
        """FR-45 stands: the reasoning layer influences the room through a
        proposed setpoint and nothing else. A tool that took a temperature
        would be a second route past the validator."""
        parameters = [
            parameter.name
            for spec in registry.specs()
            for parameter in spec.parameters
        ]
        assert not any(
            name in parameters for name in ("setpoint_c", "temperature", "command")
        )
