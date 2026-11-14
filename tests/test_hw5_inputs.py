"""HW5 Part B judge inputs: label export, leakage, one record per conversation,
turn choice, dedupe, system prompt.

Offline: reads the committed trace export, the scenario file, the review app's
label log and the users table of data/cartwheel.db; writes only to tmp_path.
No keys, no network, no model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.agent import SYSTEM_PROMPT_TEMPLATE, prompt_version
from analysis import run_judges as rj
from analysis.review_app import loader

MESSAGE_KEYS = {"role", "text", "name", "arguments", "content"}
ROLES = {"system", "user", "assistant", "tool_call", "tool_result"}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("hw5")
    out, labels_path = tmp / "inputs.json", tmp / "labels.jsonl"
    records = rj.prepare_inputs(labels_path=labels_path, out=out, report=None)
    labels = rj.read_labels(labels_path)
    convs = rj._conversations(loader.EXPORT)
    where = {t["trace_id"]: (sid, t["turn"]) for sid, turns in convs.items() for t in turns}
    turn_labels: dict[str, dict[int, int]] = {}
    for tid, row in labels.items():
        sid, n = where[tid]
        turn_labels.setdefault(sid, {})[n] = row["label"]
    groups = rj.group_variants(list(turn_labels), rj.read_scenarios())
    return {"records": records, "file": json.loads(out.read_text()), "labels": labels,
            "convs": convs, "where": where, "turn_labels": turn_labels, "groups": groups}


def test_file_matches_return_value(built: dict) -> None:
    assert built["file"] == built["records"]


def test_no_label_or_metadata_in_records(built: dict) -> None:
    for record in built["file"]:
        assert set(record) == {"trace_id", "trace"}
        for message in record["trace"]:
            assert set(message) <= MESSAGE_KEYS, message.keys()
            assert message["role"] in ROLES
    blob = json.dumps(built["file"], ensure_ascii=False)
    for row in built["labels"].values():
        assert row["scenario_id"] not in blob
        if len(row["evidence"]) > 30:
            assert row["evidence"] not in blob, row["evidence"]


def test_one_record_per_conversation(built: dict) -> None:
    sessions = [built["where"][r["trace_id"]][0] for r in built["file"]]
    assert len(sessions) == len(set(sessions))
    ids = [r["trace_id"] for r in built["file"]]
    assert len(ids) == len(set(ids))


def test_choose_turn_rule() -> None:
    assert rj.choose_turn({1: 1, 2: 0}) == 2
    assert rj.choose_turn({1: 0, 2: 0}) == 1
    assert rj.choose_turn({1: 0, 2: 1, 3: 0}) == 1
    assert rj.choose_turn({1: 1, 2: 1, 3: 1}) == 3
    assert rj.choose_turn({2: 1}) == 2


def test_record_holds_turns_up_to_the_chosen_one(built: dict) -> None:
    by_sid = {built["where"][r["trace_id"]][0]: r for r in built["file"]}
    multi = [s for s in by_sid if len(built["turn_labels"][s]) > 1]
    assert multi, "expected a kept conversation with several labeled turns"
    for sid, record in by_sid.items():
        n = rj.choose_turn(built["turn_labels"][sid])
        turns = built["convs"][sid]
        assert record["trace_id"] == turns[n - 1]["trace_id"]
        users = [m["text"] for m in record["trace"] if m["role"] == "user"]
        assert users == [t["user"] for t in turns[:n]]
        assert record["trace"][-1] == {"role": "assistant", "text": turns[n - 1]["reply"]}


def test_dedupe_keeps_lowest_id_per_group(built: dict) -> None:
    kept = {built["where"][r["trace_id"]][0] for r in built["file"]}
    for ids in built["groups"].values():
        assert len(kept & set(ids)) == 1
        assert min(ids) in kept


def test_dedupe_is_label_blind(built: dict, tmp_path: Path) -> None:
    """Inverting every label changes no kept conversation."""
    flipped = tmp_path / "flipped.jsonl"
    flipped.write_text("".join(
        json.dumps({**row, "label": 1 - row["label"]}) + "\n" for row in built["labels"].values()))
    records = rj.prepare_inputs(labels_path=flipped, label_log=None,
                                out=tmp_path / "in.json", report=None)
    kept = {built["where"][r["trace_id"]][0] for r in records}
    assert kept == {built["where"][r["trace_id"]][0] for r in built["file"]}


def test_inputs_cover_eligible_labels(built: dict) -> None:
    eligible = rj.kept_scenarios(built["groups"])
    ids = {r["trace_id"] for r in built["file"]}
    assert ids <= set(built["labels"])
    assert {built["where"][t][0] for t in ids} == eligible
    for record in built["file"]:
        sid = built["where"][record["trace_id"]][0]
        assert built["labels"][record["trace_id"]]["label"] == rj.conversation_label(
            built["turn_labels"][sid])


def test_judge_runner_loads_the_inputs(built: dict, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis.helpers.scale import load_store_traces

    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(built["file"]))
    monkeypatch.setenv("CARTWHEEL_JUDGE_TRACE_SOURCE", str(path))
    traces = load_store_traces()
    assert [t["trace_id"] for t in traces] == [r["trace_id"] for r in built["file"]]
    assert all(t["text"].startswith("system: You are Cartwheel's support assistant.")
               for t in traces)
    tool_lines = [line for t in traces for line in t["text"].splitlines()
                  if line.startswith(("tool_call", "tool_result"))]
    assert tool_lines
    assert all(line.startswith(("tool_call ", "tool_result ")) for line in tool_lines)
    assert any(line.startswith("tool_call get_policy: ") for line in tool_lines)


def test_labeled_splits_are_judged_on_the_prepared_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without CARTWHEEL_JUDGE_TRACE_SOURCE, split traces still get the system prompt,
    and the store (Langfuse) is never consulted for them."""
    from analysis.helpers import scale

    monkeypatch.delenv("CARTWHEEL_JUDGE_TRACE_SOURCE", raising=False)

    def no_store():
        raise AssertionError("split traces must not fall back to the store")

    monkeypatch.setattr(scale, "load_store_traces", no_store)
    splits = json.loads((rj.REPO / "analysis/state/splits.json").read_text())[rj.MODE]
    ids = splits["train"] + splits["dev"] + splits["test"]
    rows = scale.judge_rows(ids)
    assert [r["trace_id"] for r in rows] == ids
    assert all(r["content"].startswith("system: You are Cartwheel's support assistant.")
               for r in rows)


# --------------------------------------------------------------------------- #
# variant rule: same seeded order or product AND same intent
# --------------------------------------------------------------------------- #

def _scenario(sid: str, intent: str, order: int | None = None, product: int | None = None,
              record_state: str | None = None) -> dict:
    return {"id": sid, "tuple": {"intent": intent, "order_id": order, "product_id": product,
                                 "record_state": record_state, "applicable_policy": None}}


def test_variant_key_needs_same_record_and_same_intent() -> None:
    key = rj.variant_key
    assert key(_scenario("a", "return_deadline", order=1)) == key(_scenario("b", "return_deadline", order=1))
    assert key(_scenario("a", "return_deadline", order=1)) != key(_scenario("b", "refund_request", order=1))
    assert key(_scenario("a", "product_question", product=2)) == key(_scenario("b", "product_question", product=2))
    assert key(_scenario("a", "product_question", product=2)) != key(_scenario("b", "stock_check", product=2))
    assert key(_scenario("a", "order_status", order=2)) != key(_scenario("b", "order_status", product=2))
    # no order or product: stands alone, even on the same policy page and intent
    assert (key(_scenario("a", "policy_question", record_state="policy_page"))
            != key(_scenario("b", "policy_question", record_state="policy_page")))


def test_every_group_shares_one_record_and_one_intent(built: dict) -> None:
    scenarios = rj.read_scenarios()
    for ids in built["groups"].values():
        tuples = [scenarios[s]["tuple"] for s in ids]
        assert len({t["intent"] for t in tuples}) == 1, ids
        if len(ids) > 1:
            records = {(t.get("order_id"), t.get("product_id")) for t in tuples}
            assert len(records) == 1 and records != {(None, None)}, ids


# --------------------------------------------------------------------------- #
# system prompt: first message, rendered for the session's role, user and store
# --------------------------------------------------------------------------- #

def test_first_message_is_the_sessions_rendered_system_prompt(built: dict) -> None:
    import sqlite3

    from agent.agent import render_system_prompt
    from agent.auth import AuthContext

    attrs = {t["id"]: loader.attrs(t) for t in loader.load_pool()}
    conn = sqlite3.connect(rj.USERS_DB)
    stores = set()
    for record in built["file"]:
        a = attrs[record["trace_id"]]
        assert a["cartwheel.prompt_version"] == prompt_version(SYSTEM_PROMPT_TEMPLATE)
        user_id = int(a["cartwheel.user_id"])
        role, store_id = conn.execute(
            "SELECT role, store_id FROM users WHERE id = ?", (user_id,)).fetchone()
        assert role == a["cartwheel.user_role"]
        expected = render_system_prompt(
            AuthContext(user_id=user_id, role=role, store_id=store_id), SYSTEM_PROMPT_TEMPLATE)
        assert record["trace"][0] == {"role": "system", "text": expected}
        assert [m["role"] for m in record["trace"]].count("system") == 1
        stores.add(store_id)
    assert None in stores and stores - {None}, "expected merchant and non-merchant sessions"


def test_unknown_prompt_version_stops() -> None:
    attrs = {"cartwheel.prompt_version": "deadbeef0000", "cartwheel.user_id": "76",
             "cartwheel.user_role": "shopper"}
    with pytest.raises(ValueError, match="deadbeef0000"):
        rj.system_message(attrs, rj.read_users())


# --------------------------------------------------------------------------- #
# export_labels: review-app log (1 = failure present) -> HW5 export (1 = Pass)
# --------------------------------------------------------------------------- #

EXPORT_FIELDS = {"trace_id", "scenario_id", "mode", "label", "evidence",
                 "source_label_id", "accepted_via", "ts"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_export_is_the_latest_log_record_per_trace_with_pass_as_one(tmp_path: Path) -> None:
    out = tmp_path / "export.jsonl"
    returned = rj.export_labels(rj.MODE, log=rj.LABEL_LOG, out=out)
    rows = _rows(out)
    assert rows == returned
    latest: dict[str, dict] = {}
    for row in _rows(rj.LABEL_LOG):
        latest[row["trace_id"]] = row
    assert [r["trace_id"] for r in rows] == [
        r["trace_id"] for r in sorted(latest.values(), key=lambda r: (r["scenario_id"], r["trace_id"]))]
    for row in rows:
        src = latest[row["trace_id"]]
        assert set(row) == EXPORT_FIELDS
        assert row["label"] == 1 - src["label"]
        assert row["source_label_id"] == src["label_id"]
        assert (row["scenario_id"], row["evidence"], row["ts"]) == (
            src["scenario_id"], src["evidence"], src["ts"])


def test_relabel_in_the_log_changes_the_export(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text(rj.LABEL_LOG.read_text())
    before = {r["trace_id"]: r for r in rj.export_labels(rj.MODE, log=log, out=tmp_path / "a.jsonl")}
    last = _rows(log)[-1]
    relabel = {**last, "label": 1 - last["label"], "label_id": f"{last['trace_id']}#99",
               "supersedes": last["label_id"], "ts": "2026-11-12T00:00:00+00:00"}
    with log.open("a") as f:
        f.write(json.dumps(relabel) + "\n")
    after = {r["trace_id"]: r for r in rj.export_labels(rj.MODE, log=log, out=tmp_path / "b.jsonl")}
    tid = last["trace_id"]
    assert after[tid]["label"] == 1 - before[tid]["label"]
    assert after[tid]["source_label_id"] == relabel["label_id"]
    assert {k: v for k, v in after.items() if k != tid} == {k: v for k, v in before.items() if k != tid}


def test_prepare_inputs_exports_the_labels_first(built: dict, tmp_path: Path) -> None:
    fresh = rj.export_labels(rj.MODE, log=rj.LABEL_LOG, out=tmp_path / "x.jsonl")
    assert {r["trace_id"]: r["label"] for r in fresh} == {
        t: row["label"] for t, row in built["labels"].items()}
