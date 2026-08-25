"""Unit tests for the canonical topic definitions.

The literal topic strings asserted here are the contract in DESIGN.md
section 6.1. A change that breaks these tests is an interface change and
needs the interface freeze (R-06) reconsidered, not a test edit.
"""

import dataclasses

import pytest

from src.common import topics
from src.common.topics import Qos, TopicParameterError, TopicSpec

SENSOR_ID = "temp_01"

RETAINED_SPECS = (
    topics.SENSOR_HEALTH,
    topics.ESTIMATE_THERMAL,
    topics.ESTIMATE_COEFFICIENTS,
    topics.FAULT,
    topics.SYSTEM_MODE,
    topics.GOAL_ACTIVE,
    topics.ACTUATOR_STATE,
)

TRANSIENT_SPECS = (
    topics.SENSOR_STATE,
    topics.GOAL_PROPOSED,
    topics.ACTUATOR_COMMAND,
    topics.CONTEXT_PREFERENCE,
    topics.AUDIT_VALIDATION,
    topics.AUDIT_REASONING,
    topics.FAULT_INJECT,
)

ALL_SPECS = RETAINED_SPECS + TRANSIENT_SPECS


class TestContract:
    """The topic strings themselves, against DESIGN.md section 6.1."""

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            (topics.SENSOR_STATE, "space/sensor/{sensor_id}/state"),
            (topics.SENSOR_HEALTH, "space/sensor/{sensor_id}/health"),
            (topics.ESTIMATE_THERMAL, "space/estimate/thermal"),
            (topics.ESTIMATE_COEFFICIENTS, "space/estimate/coefficients"),
            (topics.FAULT, "space/fault/{fault_id}"),
            (topics.SYSTEM_MODE, "space/system/mode"),
            (topics.GOAL_PROPOSED, "space/goal/proposed"),
            (topics.GOAL_ACTIVE, "space/goal/active"),
            (topics.ACTUATOR_COMMAND, "space/actuator/{actuator_id}/command"),
            (topics.ACTUATOR_STATE, "space/actuator/{actuator_id}/state"),
            (topics.CONTEXT_PREFERENCE, "space/context/preference"),
            (topics.AUDIT_VALIDATION, "space/audit/validation"),
            (topics.AUDIT_REASONING, "space/audit/reasoning"),
        ],
    )
    def test_pattern_matches_the_design_document(self, spec, expected):
        assert spec.pattern == expected

    @pytest.mark.parametrize("spec", RETAINED_SPECS)
    def test_state_topics_are_retained_so_late_subscribers_see_them(self, spec):
        assert spec.retain is True

    @pytest.mark.parametrize("spec", TRANSIENT_SPECS)
    def test_event_topics_are_not_retained(self, spec):
        assert spec.retain is False

    @pytest.mark.parametrize(
        "spec", [topics.SENSOR_STATE, topics.ESTIMATE_THERMAL]
    )
    def test_high_rate_streams_use_at_most_once(self, spec):
        assert spec.qos is Qos.AT_MOST_ONCE

    @pytest.mark.parametrize(
        "spec",
        [
            topics.FAULT,
            topics.SYSTEM_MODE,
            topics.GOAL_ACTIVE,
            topics.ACTUATOR_COMMAND,
            topics.AUDIT_VALIDATION,
        ],
    )
    def test_must_not_miss_topics_use_at_least_once(self, spec):
        assert spec.qos is Qos.AT_LEAST_ONCE

    def test_every_topic_lives_under_the_blackboard_root(self):
        assert all(spec.pattern.startswith(f"{topics.TOPIC_ROOT}/") for spec in ALL_SPECS)

    def test_all_topics_wildcard_subscribes_to_the_whole_tree(self):
        assert topics.ALL_TOPICS == f"{topics.TOPIC_ROOT}/#"


class TestFormat:
    def test_substitutes_a_parameter(self):
        assert topics.SENSOR_STATE.format(sensor_id=SENSOR_ID) == (
            f"space/sensor/{SENSOR_ID}/state"
        )

    def test_air_conditioner_is_the_only_real_actuator_id(self):
        assert topics.ACTUATOR_COMMAND.format(
            actuator_id=topics.AIR_CONDITIONER_ID
        ) == "space/actuator/ac/command"

    @pytest.mark.parametrize("illegal", ["a/b", "a+b", "a#b"])
    def test_rejects_a_parameter_that_would_reshape_the_topic_tree(self, illegal):
        with pytest.raises(TopicParameterError):
            topics.SENSOR_STATE.format(sensor_id=illegal)

    def test_rejects_an_empty_parameter(self):
        with pytest.raises(TopicParameterError):
            topics.SENSOR_STATE.format(sensor_id="")

    def test_missing_parameter_is_an_error_not_a_silent_partial_topic(self):
        with pytest.raises(KeyError):
            topics.FAULT.format()


class TestWildcard:
    def test_replaces_a_parameter_with_the_single_level_wildcard(self):
        assert topics.SENSOR_STATE.wildcard() == "space/sensor/+/state"

    def test_leaves_a_parameterless_topic_unchanged(self):
        assert topics.SYSTEM_MODE.wildcard() == topics.SYSTEM_MODE.pattern

    def test_replaces_a_trailing_parameter(self):
        assert topics.FAULT.wildcard() == "space/fault/+"

    @pytest.mark.parametrize("spec", ALL_SPECS)
    def test_no_unsubstituted_parameter_remains(self, spec):
        assert "{" not in spec.wildcard()


class TestImmutability:
    def test_a_spec_cannot_be_altered_for_other_components(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            topics.SYSTEM_MODE.retain = False


class TestInjection:
    def test_injection_topic_is_parameterised_by_subject(self):
        assert topics.FAULT_INJECT.format(subject=SENSOR_ID) == (
            f"space/inject/{SENSOR_ID}"
        )

    def test_injection_is_not_retained_so_it_never_replays_on_restart(self):
        assert topics.FAULT_INJECT.retain is False
