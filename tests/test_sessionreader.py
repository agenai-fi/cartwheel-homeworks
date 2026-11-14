"""Contract tests for scripts/sessionreader.py.

These fail when the SDK, the provider, or the store's shape changes. They know
nothing about Cartwheel or any homework -- that belongs to the adapter's tests.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

from scripts import sessionreader as sr

SCHEMA = """
CREATE TABLE agent_sessions (session_id TEXT PRIMARY KEY);
CREATE TABLE agent_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_data TEXT NOT NULL
);
"""

USER = {"content": "check order 4127", "role": "user"}
CALL = {
    "type": "function_call",
    "name": "get_order",
    "arguments": '{"order_id":4127}',
    "call_id": "c1",
}
# Note the single quotes: the SDK stores str(result), not JSON.
OUT = {
    "type": "function_call_output",
    "call_id": "c1",
    "output": "{'ok': True, 'order': {'status': 'delivered'}}",
}
FINAL = {
    "content": [{"text": "Order #4127 was delivered.", "type": "output_text"}],
    "role": "assistant",
    "type": "message",
    "phase": "final_answer",
}
COMMENTARY = {
    "content": [{"text": "Let me check that for you.", "type": "output_text"}],
    "role": "assistant",
    "type": "message",
    "phase": "commentary",
}
PLAIN = {
    "content": [{"text": "Order #4127 was delivered.", "type": "output_text"}],
    "role": "assistant",
    "type": "message",
}


def make_db(
    tmp_path: Path, sessions: dict[str, list[dict]], name: str = "s.db"
) -> Path:
    path = tmp_path / name
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for session_id, messages in sessions.items():
        conn.execute(
            "INSERT INTO agent_sessions (session_id) VALUES (?)", (session_id,)
        )
        for message in messages:
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
                (session_id, json.dumps(message)),
            )
    conn.commit()
    conn.close()
    return path


def append(path: Path, session_id: str, messages: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)", (session_id,)
    )
    for message in messages:
        conn.execute(
            "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
            (session_id, json.dumps(message)),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def one_turn(tmp_path: Path) -> Path:
    return make_db(tmp_path, {"s-1": [USER, CALL, OUT, FINAL]})


# --- parsing ----------------------------------------------------------------


def test_a_tool_result_is_a_python_repr_not_json(one_turn: Path) -> None:
    """json.loads cannot read the stored result; the reader must still return a dict."""
    result = next(sr.turn_records(one_turn))["tool_calls"][0]["result"]
    assert isinstance(result, dict)
    assert result["ok"] is True
    assert result["order"]["status"] == "delivered"


def test_arguments_are_a_json_string(one_turn: Path) -> None:
    assert next(sr.turn_records(one_turn))["tool_calls"][0]["arguments"] == {
        "order_id": 4127
    }


def test_an_empty_arguments_string_becomes_an_object(tmp_path: Path) -> None:
    path = make_db(tmp_path, {"s-1": [USER, dict(CALL, arguments=""), OUT, FINAL]})
    assert next(sr.turn_records(path))["tool_calls"][0]["arguments"] == {}


def test_calls_pair_by_call_id_not_adjacency(tmp_path: Path) -> None:
    """The SDK interleaves calls and outputs; adjacency would mispair them."""
    second = dict(CALL, call_id="c2", arguments='{"order_id":3980}')
    second_out = dict(OUT, call_id="c2", output="{'ok': False}")
    path = make_db(tmp_path, {"s-1": [USER, CALL, second, OUT, second_out, FINAL]})
    calls = next(sr.turn_records(path))["tool_calls"]
    assert calls[0]["arguments"] == {"order_id": 4127}
    assert calls[0]["result"] == {"ok": True, "order": {"status": "delivered"}}
    assert calls[1]["arguments"] == {"order_id": 3980}
    assert calls[1]["result"] == {"ok": False}


def test_a_refusal_counts_as_a_reply(tmp_path: Path) -> None:
    """The SDK models a refusal as a content part with .refusal and no .text."""
    refusal = {
        "content": [{"type": "refusal", "refusal": "I can only help with orders."}],
        "role": "assistant",
        "type": "message",
        "phase": "final_answer",
    }
    path = make_db(tmp_path, {"s-1": [USER, refusal]})
    record = next(sr.turn_records(path))
    assert record["response"] == "I can only help with orders."
    assert record["status"] == "final"


def test_an_unparseable_result_falls_back_to_the_raw_string(tmp_path: Path) -> None:
    path = make_db(
        tmp_path, {"s-1": [USER, CALL, dict(OUT, output="<not a literal>"), FINAL]}
    )
    assert next(sr.turn_records(path))["tool_calls"][0]["result"] == "<not a literal>"


# --- turns ------------------------------------------------------------------


def three_turns() -> list[dict]:
    def reply(text: str) -> dict:
        return {
            "content": [{"text": text, "type": "output_text"}],
            "role": "assistant",
            "type": "message",
            "phase": "final_answer",
        }

    return [
        USER,
        CALL,
        OUT,
        reply("first"),
        {"content": "and 3980?", "role": "user"},
        reply("second"),
        {"content": "and 4455?", "role": "user"},
        reply("third"),
    ]


def test_a_session_splits_into_one_turn_per_request(tmp_path: Path) -> None:
    """One SQLiteSession per process, one turn per typed request."""
    path = make_db(tmp_path, {"s-1": three_turns()})
    records = list(sr.turn_records(path))
    assert [r["turn"] for r in records] == [1, 2, 3]
    assert all(r["turns_in_session"] == 3 for r in records)
    assert [r["response"] for r in records] == ["first", "second", "third"]
    assert [len(r["tool_calls"]) for r in records] == [1, 0, 0]


# --- completion -------------------------------------------------------------


def test_commentary_is_not_the_answer(tmp_path: Path) -> None:
    """A preamble persists a whole model inference before the real reply."""
    path = make_db(tmp_path, {"s-1": [USER, COMMENTARY, CALL, OUT]})
    record = next(sr.turn_records(path))
    assert record["status"] != "final"
    assert record["response"] == "Let me check that for you."


def test_a_phaseless_provider_is_judged_by_shape(tmp_path: Path) -> None:
    path = make_db(tmp_path, {"s-1": [USER, CALL, OUT, PLAIN]})
    assert next(sr.turn_records(path))["status"] == "final"


def test_a_phaseless_batch_ending_in_a_tool_output_is_unfinished(
    tmp_path: Path,
) -> None:
    preamble = {
        "content": [{"text": "Looking.", "type": "output_text"}],
        "role": "assistant",
        "type": "message",
    }
    path = make_db(tmp_path, {"s-1": [USER, preamble, CALL, OUT]})
    # The call HAS its result; what is missing is the closing message.
    assert "closing message" in next(sr.turn_records(path))["status"]


def test_a_call_with_no_result_is_unfinished(tmp_path: Path) -> None:
    path = make_db(tmp_path, {"s-1": [USER, CALL]})
    record = next(sr.turn_records(path))
    assert "mid-turn" in record["status"]
    assert record["tool_calls"][0]["result"] is None


def test_no_reply_is_unfinished(tmp_path: Path) -> None:
    path = make_db(tmp_path, {"s-1": [USER]})
    assert "before the agent replied" in next(sr.turn_records(path))["status"]


# --- labels -----------------------------------------------------------------


def test_a_label_pattern_names_a_session_without_the_reader_knowing_the_convention() -> (
    None
):
    pattern = re.compile(r"^cli-(?P<role>[a-z]+)-(?P<user>\d+)-")
    assert "role=shopper user=1" in sr.label("cli-shopper-1-1788822243", pattern)


def test_no_pattern_leaves_the_session_id_alone() -> None:
    assert sr.label("replay-e-001", None) == "replay-e-001"


def test_a_pattern_that_does_not_match_leaves_the_id_alone() -> None:
    assert (
        sr.label("replay-e-001", re.compile(r"^cli-(?P<role>[a-z]+)-"))
        == "replay-e-001"
    )


# --- rendering --------------------------------------------------------------


def test_rendering_never_exceeds_the_width(tmp_path: Path) -> None:
    path = make_db(tmp_path, {"s-1": [USER, CALL, OUT, FINAL]})
    record = next(sr.turn_records(path))
    record["response"] = "word " * 400
    record["tool_calls"][0]["result"] = {"body": "policy " * 300}
    for line in sr.render_turn(record, width=72, heading="-- 1. s-1 ").split("\n"):
        assert len(line) <= 72, line


def test_rendering_uses_json_spelling_for_scalars(one_turn: Path) -> None:
    text = sr.render_turn(next(sr.turn_records(one_turn)), width=88)
    assert "ok: true" in text  # not Python's True


def test_an_unfinished_turn_is_flagged_in_the_view(tmp_path: Path) -> None:
    path = make_db(tmp_path, {"s-1": [USER, CALL]})
    text = sr.render_turn(next(sr.turn_records(path)), width=88)
    assert "mid-turn" in text
    assert "(none stored" in text


# --- following --------------------------------------------------------------


def test_follow_announces_a_request_then_emits_the_finished_turn(
    tmp_path: Path,
) -> None:
    path = make_db(tmp_path, {})
    seen: dict = {}
    append(path, "s-1", [USER])
    events, wm = sr.tail_step(path, 0, seen)
    assert [e["kind"] for e in events] == ["request"]
    append(path, "s-1", [CALL, OUT, FINAL])
    events, wm = sr.tail_step(path, wm, seen)
    assert [e["kind"] for e in events] == ["turn"]
    assert events[0]["response"] == "Order #4127 was delivered."


def test_follow_never_latches_commentary_however_often_it_polls(tmp_path: Path) -> None:
    """The cadence that hides the bug: polls between the tool batch and the reply."""
    path = make_db(tmp_path, {})
    seen: dict = {}
    append(path, "s-1", [USER])
    events, wm = sr.tail_step(path, 0, seen)
    append(path, "s-1", [COMMENTARY, CALL, OUT])
    for _ in range(4):
        events, wm = sr.tail_step(path, wm, seen)
        assert [e for e in events if e["kind"] == "turn"] == []
    append(path, "s-1", [FINAL])
    events, wm = sr.tail_step(path, wm, seen)
    assert [e["kind"] for e in events] == ["turn"]
    assert events[0]["response"] == "Order #4127 was delivered."


def test_follow_never_repeats_a_turn(tmp_path: Path) -> None:
    path = make_db(tmp_path, {"s-1": [USER, CALL, OUT, FINAL]})
    seen: dict = {}
    events, wm = sr.tail_step(path, 0, seen)
    assert [e["kind"] for e in events] == ["request", "turn"]
    again, wm2 = sr.tail_step(path, wm, seen)
    assert again == [] and wm2 == wm


def test_follow_resets_for_a_replacement_store_with_more_rows(tmp_path: Path) -> None:
    """Comparing ids alone misses a restored backup and skips its history."""
    path = tmp_path / "live.db"
    make_db(tmp_path, {"s-1": [USER, FINAL]}, name="first.db").rename(path)
    seen: dict = {}
    _, wm = sr.tail_step(path, 0, seen)
    assert wm == 2
    other = tmp_path / "other"
    other.mkdir()
    make_db(other, {"s-9": three_turns()}, name="big.db").replace(path)
    events, _ = sr.tail_step(path, wm, seen)
    assert any(e["kind"] == "reset" for e in events)
    assert [e["turn"] for e in events if e["kind"] == "turn"] == [1, 2, 3]


def test_follow_survives_the_store_being_deleted(tmp_path: Path) -> None:
    path = make_db(tmp_path, {"s-1": [USER, FINAL]})
    seen: dict = {}
    _, wm = sr.tail_step(path, 0, seen)
    path.unlink()
    events, still = sr.tail_step(path, wm, seen)
    assert events == [] and still == wm
    assert not path.exists(), "a reader must never create a store"


def test_nothing_ever_creates_a_store(tmp_path: Path) -> None:
    missing = tmp_path / "absent.db"
    assert sr.current_watermark(missing) == 0
    assert sr.tail_step(missing, 0, {}) == ([], 0)
    assert not missing.exists()


# --- cli --------------------------------------------------------------------


def test_a_later_turn_carries_its_antecedents(tmp_path: Path) -> None:
    """ "and 3980?" is unintelligible alone; the record must say what preceded it."""
    path = make_db(tmp_path, {"s-1": three_turns()})
    records = list(sr.turn_records(path))
    assert records[0]["context"] == []
    assert records[1]["context"] == ["check order 4127"]
    assert records[2]["context"] == ["check order 4127", "and 3980?"]


def test_an_unanswered_call_is_unfinished_even_when_stamped_final(
    tmp_path: Path,
) -> None:
    """A dangling tool call outranks whatever phase the model stamped."""
    path = make_db(tmp_path, {"s-1": [USER, CALL, FINAL]})
    assert "mid-turn" in next(sr.turn_records(path))["status"]


def test_the_record_shape_is_the_whole_contract(one_turn: Path) -> None:
    """This module has no CLI; turn_records IS the boundary."""
    assert set(next(sr.turn_records(one_turn))) == {
        "session_id",
        "turn",
        "turns_in_session",
        "status",
        "request",
        "response",
        "tool_calls",
        "context",
    }
