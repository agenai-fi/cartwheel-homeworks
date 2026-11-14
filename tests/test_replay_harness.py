from __future__ import annotations

import pytest

from replay.harness import ReplayInfraError, replay_case, summarize_rollouts
from replay.rollout import judge_trace_text


def test_replay_resets_before_each_run_and_retries_only_infrastructure() -> None:
    resets = 0
    calls = 0

    def reset() -> None:
        nonlocal resets
        resets += 1

    def runner() -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReplayInfraError("temporary service error")
        return {"passed": calls % 2 == 0}

    records = replay_case(runner, reset, n=2, max_infra_retries=1)

    assert resets == 3
    assert calls == 3
    assert [record["rollout"] for record in records] == [0, 1]
    assert [record["passed"] for record in records] == [True, False]


def test_replay_does_not_retry_a_completed_failure() -> None:
    calls = 0

    def runner() -> dict:
        nonlocal calls
        calls += 1
        return {"passed": False}

    records = replay_case(runner, lambda: None, n=2)

    assert calls == 2
    assert len(records) == 2


def test_rollout_summary_is_deterministic() -> None:
    records = [
        {"passed": True, "steps": 2},
        {"passed": False, "failure_modes": ["judge:a"], "steps": 4},
        {
            "passed": False,
            "failure_modes": ["judge:a", "check:b"],
            "steps": 6,
        },
    ]

    first = summarize_rollouts(records, bootstrap_iterations=100, seed=7)
    second = summarize_rollouts(records, bootstrap_iterations=100, seed=7)

    assert first == second
    assert first["failures"] == 2
    assert first["failure_rate"] == pytest.approx(2 / 3)
    assert first["mode_counts"] == {"judge:a": 2, "check:b": 1}
    assert first["steps"] == {"min": 2, "median": 4, "max": 6}


def test_judge_trace_text_uses_the_hw5_normalized_roles() -> None:
    transcript = {
        "system_prompt": "You are Cartwheel's support assistant.",
        "turns": [
            {
                "user": "Where is my order?",
                "events": [
                    {"role": "assistant", "text": "I will look it up."},
                    {"role": "tool_call", "name": "get_order", "arguments": {"order_id": 42}},
                    {
                        "role": "tool_result",
                        "name": "get_order",
                        "content": {"status": "shipped", "ok": True},
                    },
                    {"role": "assistant", "text": "It shipped today."},
                ],
            }
        ],
    }

    assert judge_trace_text(transcript).splitlines() == [
        "system: You are Cartwheel's support assistant.",
        "user: Where is my order?",
        "assistant: I will look it up.",
        'tool_call get_order: {"order_id": 42}',
        'tool_result get_order: {"ok": true, "status": "shipped"}',
        "assistant: It shipped today.",
    ]


def test_judge_trace_text_refuses_a_transcript_without_turn_events() -> None:
    with pytest.raises(ValueError, match="events"):
        judge_trace_text({"turns": [{"user": "hi", "reply": "hello", "tool_calls": []}]})


def _transcript_from_hw5_messages(messages: list[dict]) -> dict:
    """Rebuild the runtime transcript shape from one recorded HW5 judge input."""
    transcript: dict = {"turns": []}
    for message in messages:
        if message["role"] == "system":
            transcript["system_prompt"] = message["text"]
        elif message["role"] == "user":
            transcript["turns"].append({"user": message["text"], "events": []})
        else:
            transcript["turns"][-1]["events"].append(message)
    return transcript


def test_judge_trace_text_reproduces_every_frozen_hw5_judge_input() -> None:
    """The HW6 verifier must give the frozen judge the input it was validated on
    (hw6.md:11, :31): byte-identical to analysis.helpers.scale.judge_rows."""
    import json
    from pathlib import Path

    from analysis.helpers.scale import judge_rows

    path = Path(__file__).resolve().parents[1] / "analysis" / "state" / "hw5_trace_inputs.json"
    if not path.exists():
        pytest.skip("no HW5 judge inputs in this checkout")
    records = json.loads(path.read_text())
    expected = {row["trace_id"]: row["content"] for row in judge_rows([r["trace_id"] for r in records])}
    mismatched = [
        r["trace_id"]
        for r in records
        if judge_trace_text(_transcript_from_hw5_messages(r["trace"])) != expected[r["trace_id"]]
    ]
    assert len(records) > 0
    assert mismatched == []


def test_extract_turn_keeps_the_order_of_text_and_tool_steps() -> None:
    from agents import Agent
    from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from replay.rollout import _extract_turn

    agent = Agent(name="t")

    def message(text: str, msg_id: str) -> MessageOutputItem:
        raw = ResponseOutputMessage(
            id=msg_id,
            role="assistant",
            status="completed",
            type="message",
            content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
        )
        return MessageOutputItem(agent=agent, raw_item=raw)

    call = ResponseFunctionToolCall(
        arguments='{"order_id": 42}', call_id="c1", name="get_order", type="function_call"
    )
    items = [
        message("I will look it up.", "m1"),
        ToolCallItem(agent=agent, raw_item=call),
        ToolCallOutputItem(
            agent=agent,
            raw_item={"call_id": "c1", "output": "x", "type": "function_call_output"},
            output={"ok": True},
        ),
        message("It shipped.", "m2"),
    ]

    turn = _extract_turn(items)

    assert turn["events"] == [
        {"role": "assistant", "text": "I will look it up."},
        {"role": "tool_call", "name": "get_order", "arguments": {"order_id": 42}},
        {"role": "tool_result", "name": "get_order", "content": {"ok": True}},
        {"role": "assistant", "text": "It shipped."},
    ]
    assert turn["reply"] == "I will look it up.\nIt shipped."
    assert turn["tool_calls"] == [{"name": "get_order", "args": {"order_id": 42}, "result": {"ok": True}}]
