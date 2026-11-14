"""The Module 2 normalizer must read attributes nested by Langfuse."""

from __future__ import annotations

import json

from analysis.helpers.normalization import _metadata, normalize_trace


def test_metadata_merges_nested_langfuse_attributes() -> None:
    record = {
        "metadata": {
            "scope": {},
            "attributes": json.dumps(
                {"cartwheel.scenario_id": "support-9", "cartwheel.user_role": "shopper"}
            ),
        }
    }
    merged = _metadata(record)
    assert merged["cartwheel.scenario_id"] == "support-9"
    assert merged["cartwheel.user_role"] == "shopper"


def test_metadata_keeps_flat_keys_authoritative() -> None:
    record = {
        "metadata": {
            "cartwheel.scenario_id": "flat",
            "attributes": {"cartwheel.scenario_id": "nested"},
        }
    }
    assert _metadata(record)["cartwheel.scenario_id"] == "flat"


def test_normalized_trace_keeps_cartwheel_session_id() -> None:
    trace = normalize_trace(
        {
            "id": "trace-1",
            "text": "hello",
            "metadata": {
                "attributes": {"cartwheel.session_id": "session-1"},
            },
        }
    )
    assert trace["meta"]["session_id"] == "session-1"


def test_flattened_text_names_the_tool_and_keeps_the_system_prompt() -> None:
    """A judge reads `text`; without the tool name it cannot tell a policy lookup
    from an order lookup, so both the call and its result carry the name. The
    system message, first in an HW5 judge input, is part of that text too."""
    trace = normalize_trace(
        {
            "trace_id": "t-tools",
            "trace": [
                {"role": "system", "text": "You are Cartwheel's support assistant."},
                {"role": "user", "text": "What is the return window?"},
                {"role": "tool_call", "name": "get_policy", "arguments": {"policy_id": "cw-returns"}},
                {"role": "tool_result", "name": "get_policy", "content": {"body": "30 days"}},
                {"role": "assistant", "text": "30 days (cw-returns)."},
            ],
        }
    )
    assert trace["text"].splitlines() == [
        "system: You are Cartwheel's support assistant.",
        "user: What is the return window?",
        'tool_call get_policy: {"policy_id": "cw-returns"}',
        'tool_result get_policy: {"body": "30 days"}',
        "assistant: 30 days (cw-returns).",
    ]
