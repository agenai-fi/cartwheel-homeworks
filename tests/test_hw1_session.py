"""Contract tests for scripts/hw1_session.py, the Homework 1 adapter.

These fail when hw1.md's requirements or Cartwheel's conventions change. The
session store's own behaviour is tested in tests/test_sessionreader.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import hw1_session as hw

TURN = {
    "session_id": "cli-shopper-1-1788822243",
    "turn": 1,
    "turns_in_session": 1,
    "status": "final",
    "request": "check order 4127",
    "response": "Order #4127 was delivered.",
    "tool_calls": [
        {"name": "get_order", "arguments": {"order_id": 4127}, "result": {"ok": True}}
    ],
}


def quiet(_message: str) -> None:
    return None


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A real session store holding the one turn TURN describes."""
    from tests.test_sessionreader import CALL, FINAL, OUT, USER, make_db

    return make_db(tmp_path, {"cli-shopper-1-1788822243": [USER, CALL, OUT, FINAL]})


# --- shaping ----------------------------------------------------------------


def test_a_turn_becomes_a_record_with_every_homework_field() -> None:
    record = hw.to_record(TURN, warn=quiet)
    assert set(hw.HOMEWORK_FIELDS) <= set(record)
    assert set(record) - set(hw.HOMEWORK_FIELDS) == set(hw.EXTRA_FIELDS)


def test_role_and_user_come_from_the_session_id() -> None:
    """The naming convention is agent/cli.py's, not the store's."""
    record = hw.to_record(TURN, warn=quiet)
    assert record["role"] == "shopper"
    assert record["user_id"] == 1


def test_judgments_start_unfilled() -> None:
    record = hw.to_record(TURN, warn=quiet)
    assert record["expected"] == ""
    assert record["requirement"] is None
    assert record["met_requirement"] is None
    assert record["problem_source"] is None


def test_a_non_cli_session_is_not_evidence() -> None:
    warned: list[str] = []
    assert hw.to_record(dict(TURN, session_id="replay-e-001"), warned.append) is None
    assert any("not an agent.cli session" in w for w in warned)


def test_an_unfinished_turn_is_not_evidence() -> None:
    warned: list[str] = []
    unfinished = dict(TURN, status="the run ended before the agent replied")
    assert hw.to_record(unfinished, warned.append) is None
    assert any("ended before the agent replied" in w for w in warned)


def test_every_turn_of_a_session_becomes_its_own_record() -> None:
    """hw1.md gives a record one request, so a 3-turn session is 3 records."""
    turns = [dict(TURN, turn=n, turns_in_session=3, request=f"q{n}") for n in (1, 2, 3)]
    records = [hw.to_record(t, warn=quiet) for t in turns]
    assert [r["turn"] for r in records] == [1, 2, 3]
    assert [r["request"] for r in records] == ["q1", "q2", "q3"]


def test_only_merchants_carry_a_store_id(world: dict) -> None:
    assert hw.to_record(TURN, warn=quiet)["store_id"] is None
    merchant = dict(TURN, session_id="cli-merchant-9002-1")
    assert hw.to_record(merchant, warn=quiet)["store_id"] == 2


def test_a_missing_world_database_warns_and_leaves_store_id_null(monkeypatch) -> None:
    """The branch that actually needs the database is the merchant one."""
    monkeypatch.setenv("CARTWHEEL_DB", "/nonexistent/cartwheel.db")
    warned: list[str] = []
    record = hw.to_record(dict(TURN, session_id="cli-merchant-9002-1"), warned.append)
    assert record["store_id"] is None
    assert any("world database missing" in w for w in warned)


def complete() -> dict:
    return {
        "session_id": "cli-shopper-1-1",
        "turn": 1,
        "role": "shopper",
        "user_id": 1,
        "store_id": None,
        "request": "check order 4127",
        "tool_calls": [{"name": "get_order", "arguments": {}, "result": {"ok": True}}],
        "response": "delivered",
        "expected": "the agent reports the order status",
        "requirement": "AUTH-1",
        "met_requirement": True,
        "problem_source": None,
    }


def test_spec_ids_include_the_tools_table() -> None:
    """TOOL-1..9 are a markdown table, not bold prose."""
    ids = hw.spec_requirement_ids()
    assert {"AUTH-1", "RESP-1", "ESC-4", "TOOL-7"} <= ids


def test_a_complete_record_validates() -> None:
    assert hw.validate([complete()]) == []


def test_a_recorded_failure_validates() -> None:
    record = complete()
    record["met_requirement"] = False
    record["problem_source"] = "prompt"
    assert hw.validate([record]) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r: r.pop("response"), "missing 'response'"),
        (lambda r: r.update(response=""), "response is empty"),
        (lambda r: r.update(request="  "), "request is empty"),
        (lambda r: r.update(role="admin"), "is not one of"),
        (lambda r: r.update(expected=""), "expected is not filled in"),
        (lambda r: r.update(requirement="AUTH-99"), "not in SPEC.md"),
        (lambda r: r.update(met_requirement=None), "met_requirement is not filled in"),
        (lambda r: r.update(met_requirement=False), "problem_source is required"),
        (
            lambda r: r.update(problem_source="model", met_requirement=False),
            "is not one of",
        ),
        (lambda r: r.update(problem_source="prompt"), "must be null when met is true"),
        (lambda r: r["tool_calls"][0].update(result=None), "has no result"),
    ],
)
def test_validation_catches(mutate, expected) -> None:
    record = complete()
    mutate(record)
    assert any(expected in problem for problem in hw.validate([record]))


def test_duplicate_turns_are_caught() -> None:
    assert any("duplicate" in p for p in hw.validate([complete(), complete()]))


# --- file io ----------------------------------------------------------------


def test_read_jsonl_names_the_line_it_cannot_parse(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    path.write_text('{"session_id": "s"}\n{not json}\n')
    with pytest.raises(SystemExit) as caught:
        hw.read_jsonl(path)
    assert ":2:" in str(caught.value)


def test_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "f.jsonl"
    hw.write_jsonl(path, [complete()])
    assert hw.read_jsonl(path) == [complete()]
    assert list(tmp_path.iterdir()) == [path]


# --- cli --------------------------------------------------------------------


def test_a_later_turn_carries_its_antecedents() -> None:
    """A record whose request is "4455" is unjudgeable without what preceded it."""
    turn = dict(
        TURN, turn=2, context=["I want a refund for order 4455"], request="4455"
    )
    assert hw.to_record(turn, warn=quiet)["context"] == [
        "I want a refund for order 4455"
    ]


def test_the_minimum_counts_conversations_not_turns(tmp_path, capsys) -> None:
    """hw1.md asks for 10 CONVERSATIONS; 12 turns from 2 sessions is not that."""
    out = tmp_path / "f.jsonl"
    records = []
    for session in range(2):
        for turn in range(6):
            record = complete()
            record["session_id"] = f"cli-shopper-1-{session}"
            record["turn"] = turn + 1
            records.append(record)
    hw.write_jsonl(out, records)
    assert hw.main(["dump", str(out), "--db", str(tmp_path / "absent.db")]) == 1
    assert "2 present" in capsys.readouterr().out


def test_a_merchant_record_needs_a_store_id() -> None:
    record = complete()
    record.update(role="merchant", user_id=9002, store_id=None)
    assert any("needs a store_id" in p for p in hw.validate([record]))


def test_a_shopper_record_must_not_carry_a_store_id() -> None:
    record = complete()
    record["store_id"] = 2
    assert any("must be null for shopper" in p for p in hw.validate([record]))


def test_a_complete_session_draws_no_note() -> None:
    records = [
        dict(complete(), session_id="s", turn=n, turns_in_session=2) for n in (1, 2)
    ]
    assert hw.missing_turns(records) == []


def test_a_record_carries_its_sessions_turn_count() -> None:
    turn = dict(TURN, turn=2, turns_in_session=6)
    assert hw.to_record(turn, warn=quiet)["turns_in_session"] == 6


# --- refresh ----------------------------------------------------------------


def test_refresh_rebuilds_the_transcript_and_keeps_the_judgment() -> None:
    """The whole reason `verify` is not needed: staleness is impossible."""
    existing = [
        dict(
            complete(),
            request="what an older reader produced",
            expected="the agent reports the status",
            met_requirement=True,
        )
    ]
    fresh = [dict(complete(), request="what the reader produces now", expected="")]
    merged = hw.refresh(existing, fresh)
    assert merged[0]["request"] == "what the reader produces now"
    assert merged[0]["expected"] == "the agent reports the status"
    assert merged[0]["met_requirement"] is True


def test_refresh_keeps_a_recorded_failure() -> None:
    """met_requirement False is falsy; a truthiness test would erase it."""
    existing = [dict(complete(), met_requirement=False, problem_source="prompt")]
    fresh = [dict(complete(), met_requirement=None, problem_source=None)]
    merged = hw.refresh(existing, fresh)
    assert merged[0]["met_requirement"] is False
    assert merged[0]["problem_source"] == "prompt"


def test_refresh_keeps_a_record_the_store_no_longer_holds() -> None:
    existing = [dict(complete(), session_id="vanished", expected="keep me")]
    assert hw.refresh(existing, []) == existing


def test_refresh_appends_new_turns() -> None:
    existing = [dict(complete(), session_id="s", turn=1)]
    fresh = [dict(complete(), session_id="s", turn=n) for n in (1, 2)]
    assert [hw.record_key(r) for r in hw.refresh(existing, fresh)] == [
        ("s", 1),
        ("s", 2),
    ]


def test_refresh_refuses_a_file_whose_keys_were_trimmed() -> None:
    existing = [{k: v for k, v in complete().items() if k != "turn"}]
    with pytest.raises(SystemExit) as caught:
        hw.refresh(existing, [])
    assert "cannot be matched" in str(caught.value)


# --- the two commands -------------------------------------------------------


def test_dump_writes_the_records(tmp_path, store, capsys) -> None:
    out = tmp_path / "records.jsonl"
    assert hw.main(["dump", str(out), "--db", str(store)]) == 1  # judgments unfilled
    records = hw.read_jsonl(out)
    assert len(records) == 1
    assert records[0]["role"] == "shopper"
    assert "wrote 1 records" in capsys.readouterr().out


def test_dump_twice_preserves_a_judgment_written_by_hand(tmp_path, store) -> None:
    out = tmp_path / "records.jsonl"
    hw.main(["dump", str(out), "--db", str(store)])
    records = hw.read_jsonl(out)
    records[0].update(
        expected="written by hand", met_requirement=False, problem_source="prompt"
    )
    hw.write_jsonl(out, records)
    hw.main(["dump", str(out), "--db", str(store)])
    again = hw.read_jsonl(out)
    assert again[0]["expected"] == "written by hand"
    assert again[0]["met_requirement"] is False


def test_dump_without_a_store_validates_the_file_instead(tmp_path, capsys) -> None:
    """A fresh clone has no store; judging what is on disk must still work."""
    out = tmp_path / "records.jsonl"
    hw.write_jsonl(out, [complete()])
    assert hw.main(["dump", str(out), "--db", str(tmp_path / "absent.db")]) == 1
    printed = capsys.readouterr().out
    assert "validating" in printed
    assert "1 records from 1 conversations" in printed


def test_dump_needs_a_filename() -> None:
    with pytest.raises(SystemExit):
        hw.main(["dump"])


def test_tail_renders_the_conversations(tmp_path, store, capsys) -> None:
    assert hw.main(["tail", "--db", str(store)]) == 0
    printed = capsys.readouterr().out
    assert "REQUEST" in printed and "role=shopper" in printed


def test_tail_can_pick_one_turn_by_session_and_number(tmp_path, store, capsys) -> None:
    """Addressed by the same key the records use, so they cannot desynchronise."""
    assert (
        hw.main(["tail", "--db", str(store), "--turn", "cli-shopper-1-1788822243:1"])
        == 0
    )
    assert "check order 4127" in capsys.readouterr().out
    assert hw.main(["tail", "--db", str(store), "--turn", "nope:1"]) == 1


def test_tail_without_a_store_fails_cleanly(tmp_path) -> None:
    assert hw.main(["tail", "--db", str(tmp_path / "absent.db")]) == 1
