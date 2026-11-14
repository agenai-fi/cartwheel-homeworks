"""The HW4 review loader: grouping, ordering, scope and ground truth.

Offline: reads the committed trace export and the seeded database. No keys, no
network, no model calls.

Each test pins a behaviour that the course's own normalizer gets wrong for human
review, so a regression toward it fails loudly rather than quietly reintroducing
a fabricated conversation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from analysis.review_app import loader

RETRY_SCENARIO = "support-0144"
ABANDONED_TRACE = "1c828dff7e934ab059048deb2644d592"


@pytest.fixture(scope="module", autouse=True)
def _pin_the_run_database():
    """Point the loader at data/cartwheel.db, the database the traces ran against.

    `tests/conftest.py`'s session-scoped `world` fixture sets CARTWHEEL_DB by
    plain assignment, so it leaks for the rest of the session: without pinning,
    `pytest tests/test_review_app.py` reads data/cartwheel.db while a full
    `pytest` run reads a freshly seeded temp world, and the same tests see two
    different databases.

    A fresh seed is the WRONG database for reviewing these traces. It is the
    pre-run state. Refund rows 575-583 were created by the HW3 run itself -- in
    a fresh world order 8438 is `delivered` with no refund, while the trace's own
    `get_order` result shows `refunded`. Ground truth for a trace review has to
    be the state the traces ran against.

    Caveat this pins rather than solves: data/cartwheel.db is the state AFTER all
    250 scenarios ran, so for an order a later scenario mutated it can differ
    from what the agent saw mid-run. Order 8438 is exactly that -- support-0045
    refunded it before support-0075 read it.

    Scoped through MonkeyPatch.context so the variable unwinds. Setting it by
    plain assignment, as an earlier version did, leaks to every module collected
    afterwards -- the same defect this docstring criticises in conftest.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CARTWHEEL_DB", str(loader.REPO / "data" / "cartwheel.db"))
        yield


@pytest.fixture(scope="module")
def pool() -> list[dict]:
    return loader.load_pool()


def test_excluded_trace_is_absent_from_the_pool(pool: list[dict]) -> None:
    assert ABANDONED_TRACE in loader.EXCLUDED
    assert loader.EXCLUDED[ABANDONED_TRACE].strip(), "an exclusion must carry its reason"
    assert all(t["id"] != ABANDONED_TRACE for t in pool)


def test_one_session_per_scenario_holds_on_the_pool(pool: list[dict]) -> None:
    loader.check_one_session_per_scenario(pool)


def test_invariant_raises_when_a_scenario_has_two_sessions(pool: list[dict]) -> None:
    """Re-admitting the abandoned attempt must be caught, not rendered."""
    raw = json.loads(loader.EXPORT.read_text())["traces"]
    readmitted = [t for t in raw if t["id"] == ABANDONED_TRACE]
    assert readmitted, "the abandoned trace should still exist in the export"
    with pytest.raises(loader.InvariantViolation) as exc:
        loader.check_one_session_per_scenario(pool + readmitted)
    assert RETRY_SCENARIO in str(exc.value)


def test_grouping_key_is_the_session_not_the_scenario() -> None:
    """Tests the KEY, on the full export, independent of the exclusion.

    Written this way because the first version asserted only what EXCLUDED
    already guarantees: on the reduced pool both keys give the same answer, so
    swapping session_of for scenario_of did not fail it. Mutation testing caught
    that. Here the abandoned attempt is present, so the two keys must disagree.
    """
    raw = json.loads(loader.EXPORT.read_text())["traces"]
    retry = [t for t in raw if loader.scenario_of(t) == RETRY_SCENARIO]
    assert len(retry) == 3, "the export should hold all three attempts"

    by_session = loader.sessions(retry)
    assert len(by_session) == 2, "session grouping must keep the attempts apart"
    assert sorted(len(g) for g in by_session.values()) == [1, 2]

    by_scenario = {}
    for t in retry:
        by_scenario.setdefault(loader.scenario_of(t), []).append(t)
    assert len(by_scenario) == 1, "scenario grouping would fuse them into one"


def test_retry_scenario_renders_only_the_surviving_session(pool: list[dict]) -> None:
    view = loader.view_for(RETRY_SCENARIO, pool)
    assert view is not None
    assert view["totals"]["turns"] == 2, "the abandoned attempt must not be a third turn"
    assert len(view["trace_ids"]) == 2
    assert ABANDONED_TRACE not in view["trace_ids"]


def test_every_turn_keeps_its_own_trace_id(pool: list[dict]) -> None:
    """Part E writes one Langfuse score per trace, so no id may be discarded."""
    view = loader.view_for("support-0075", pool)
    assert len(view["trace_ids"]) == view["totals"]["turns"] == 3
    assert len(set(view["trace_ids"])) == 3
    assert [t["trace_id"] for t in view["turns"]] == view["trace_ids"]


def test_steps_are_chronological_even_though_the_api_is_not(pool: list[dict]) -> None:
    """10 of 11 inspected traces returned observations out of order."""
    for view in loader.all_views(pool)[:40]:
        for turn in view["turns"]:
            ats = [s["at"] for s in turn["steps"]]
            assert ats == sorted(ats), f"steps out of order in {turn['trace_id']}"

    unsorted_raw = sum(
        1 for t in pool
        if [o.get("startTime") for o in t.get("observations") or []]
        != sorted(o.get("startTime") for o in t.get("observations") or [])
    )
    assert unsorted_raw, "expected the export to contain unsorted observation lists"


def test_distinct_tools_is_computed_over_the_whole_conversation(pool: list[dict]) -> None:
    """The course normalizer leaves this at the first turn's value (reports 1)."""
    view = loader.view_for("support-0075", pool)
    names = {s["tool"] for t in view["turns"] for s in t["steps"] if s["kind"] == "tool"}
    assert view["totals"]["distinct_tools"] == len(names) >= 4


def test_order_facts_flag_a_partial_refund(pool) -> None:
    view = loader.view_for("support-0075", pool)
    order = next(o for o in view["facts"]["orders"] if o["order_id"] == 8438)
    assert order["status"] == "refunded"
    assert 0 < order["refunded_executed_usd"] < order["total_usd"]
    assert "PARTIALLY_REFUNDED" in order["flags"]


def test_a_stale_stamp_after_a_refund_is_not_called_a_data_defect(pool) -> None:
    """Order 8438 disagrees with the oracle only because the HW3 run refunded it.

    `agent/db.py` sets `status='refunded'` and never clears `refund_eligible`, so
    every order this run refunded carries a stale stamp. An earlier version
    badged that as ELIGIBILITY_FLAG_DISAGREES_WITH_ORACLE -- a red anomaly on the
    conversation that caused it -- and this test asserted the artifact as ground
    truth. On a freshly seeded world exactly one order disagrees: 8001.
    """
    order = next(o for o in loader.view_for("support-0075", pool)["facts"]["orders"]
                 if o["order_id"] == 8438)
    assert order["refund_eligible_flag"] != order["refund_eligible_by_function"]
    assert "ELIGIBILITY_STAMP_STALE_AFTER_REFUND" in order["flags"]
    assert "ELIGIBILITY_FLAG_DISAGREES_WITH_ORACLE" not in order["flags"]


def test_the_seeded_eligibility_defect_is_still_called_a_data_defect(pool) -> None:
    """8001 is the documented data_quality_cases row: not refunded, yet its stamp
    and the oracle disagree. It must keep the defect name."""
    conn = loader._connect()
    try:
        facts = loader.order_facts(8001, conn, loader.world_asof(conn))
        assert facts["status"] != "refunded"
        assert "ELIGIBILITY_FLAG_DISAGREES_WITH_ORACLE" in facts["flags"]
        assert "ELIGIBILITY_STAMP_STALE_AFTER_REFUND" not in facts["flags"]
    finally:
        conn.close()


def test_status_refunded_with_no_executed_money_is_flagged(pool: list[dict]) -> None:
    """The order record claiming `refunded` while no money moved is this
    homework's flagship conflation; a pending-looking flag alone does not say it."""
    hits = [o for v in loader.all_views(pool) for o in v["facts"]["orders"]
            if "STATUS_REFUNDED_BUT_NOTHING_EXECUTED" in o["flags"]]
    assert hits, "expected orders whose status says refunded with zero executed"
    for o in hits:
        assert o["status"] == "refunded"
        assert o["refunded_executed_usd"] == 0


def test_the_view_states_the_date_its_day_counts_are_measured_against(pool) -> None:
    """The rail shows `days_since_delivery_vs_asof`; without the anchor on the
    payload the reviewer cannot tell what "asof" is."""
    conn = loader._connect()
    try:
        expected = loader.world_asof(conn).isoformat()
    finally:
        conn.close()
    assert loader.view_for("support-0075", pool)["world_asof"] == expected


def test_eligibility_honours_a_store_return_window_override(pool: list[dict]) -> None:
    """Four stores override the 30-day platform window; ignoring them raised a
    red anomaly flag on consistent data."""
    conn = loader._connect()
    try:
        asof = loader.world_asof(conn)
        row = conn.execute(
            "SELECT o.id FROM orders o JOIN stores s ON s.id=o.store_id "
            "WHERE s.return_window_days_override IS NOT NULL AND o.status='delivered' "
            "LIMIT 1").fetchone()
        assert row, "the seeded world should have an order in an override store"
        facts = loader.order_facts(row["id"], conn, asof)
        assert facts["store_window_override_days"] is not None
        assert facts["return_window_days"] == facts["store_window_override_days"]
        assert facts["return_window_days"] != loader.PLATFORM_WINDOW_DAYS
    finally:
        conn.close()


def test_queued_refunds_are_not_counted_as_money_already_refunded(pool: list[dict]) -> None:
    """A queued refund has not moved money -- the exact conflation hw4's worked
    example is about."""
    conn = loader._connect()
    try:
        asof = loader.world_asof(conn)
        row = conn.execute(
            "SELECT order_id FROM refunds WHERE status='queued_for_approval' LIMIT 1").fetchone()
        assert row, "the seeded world should have a queued refund"
        facts = loader.order_facts(row["order_id"], conn, asof)
        assert facts["refunded_queued_usd"] > 0
        assert facts["refunded_executed_usd"] == 0 or "PARTIALLY_REFUNDED" in facts["flags"]
        assert "REFUND_QUEUED_NOT_EXECUTED" in facts["flags"]
    finally:
        conn.close()


def test_the_three_order_side_data_quality_defects_are_flagged(pool: list[dict]) -> None:
    """Reversed dates, a missing delivery date and a store mismatch are seeded
    defects that carried no flag at all in the first version."""
    conn = loader._connect()
    try:
        asof = loader.world_asof(conn)
        expected = {
            8001: "SHIPPED_AFTER_DELIVERED",
            8002: "DELIVERED_WITHOUT_DELIVERY_DATE",
            8003: "ORDER_STORE_NE_PRODUCT_STORE",
        }
        for order_id, flag in expected.items():
            facts = loader.order_facts(order_id, conn, asof)
            assert flag in facts["flags"], f"order {order_id} missing {flag}: {facts['flags']}"
    finally:
        conn.close()


def test_world_asof_comes_from_the_database_not_a_constant() -> None:
    conn = loader._connect()
    try:
        assert loader.world_asof(conn).isoformat() == conn.execute(
            "SELECT value FROM meta WHERE key='world_asof'").fetchone()["value"]
    finally:
        conn.close()


def test_a_trace_without_a_session_id_is_refused() -> None:
    """Grouping on a None key would fuse every such trace into one conversation."""
    with pytest.raises(ValueError, match="cartwheel.session_id"):
        loader.sessions([{"id": "x", "timestamp": "2026-01-01T00:00:00Z", "metadata": {}}])


def test_product_facts_flag_same_title_siblings(pool: list[dict]) -> None:
    """Found dynamically: pinning an id would break on a reseeded world."""
    for view in loader.all_views(pool):
        for product in view["facts"]["products"]:
            if product.get("same_title_in_same_store"):
                assert "SAME_TITLE_IN_SAME_STORE" in product["flags"]
                sibling = product["same_title_in_same_store"][0]
                assert sibling["product_id"] != product["product_id"]
                return
    pytest.fail("no duplicate-titled product reached any conversation's facts")


def test_citations_in_markdown_bold_are_detected(pool: list[dict]) -> None:
    """The agent's house style is bold, often bracketed: `**[cw-returns]**`.

    The first detector split the reply on whitespace and stripped `.,;:()[]`,
    which has no `*`, so it found citations in 32 of 250 conversations where a
    boundary-anchored match finds 121. Every miss rendered as "cited: none".
    """
    view = loader.view_for("support-0001", pool)
    reply = " ".join(t["reply"] for t in view["turns"])
    assert "**cw-returns**" in reply or "**[cw-returns]" in reply, "fixture changed"
    assert "cw-returns" in view["policy"]["cited"]


def test_citation_detection_is_whole_word_not_substring() -> None:
    """`cw-returns` must not be found inside `cw-returns-extended`, and vice versa."""
    turns = [{"reply": "see cw-returns-extended and **[cw-refunds]**", "steps": []}]
    cited = loader.policy_provenance(turns)["cited"]
    assert "cw-refunds" in cited
    assert "cw-returns" not in cited


def test_policy_provenance_separates_opened_from_merely_surfaced(pool: list[dict]) -> None:
    """A policy that only appeared in a search snippet was never read.

    Collapsing the two into one "retrieved" set made the verdict field empty on
    all 250 conversations. Separated, support-0110 cites a policy it never opened.
    """
    view = loader.view_for("support-0110", pool)
    p = view["policy"]
    assert "cw-returns" in p["opened"]
    assert "cw-cancellations" in p["surfaced"], "surfaced by search, never opened"
    assert "cw-cancellations" not in p["opened"]
    assert "cw-cancellations" in p["cited_not_opened"]
    assert len(p["known_policy_ids"]) == 18


def test_at_least_one_conversation_cites_a_policy_it_never_opened(pool: list[dict]) -> None:
    gaps = [v["scenario_id"] for v in loader.all_views(pool) if v["policy"]["cited_not_opened"]]
    assert len(gaps) >= 5, f"expected several, got {gaps}"


def test_census_reports_traces_and_conversations(pool: list[dict]) -> None:
    c = loader.census(pool)
    assert c["traces"] == 279
    assert c["conversations"] == c["scenarios"] == 250
    assert c["excluded"] == 1


# --------------------------------------------------------------------------- #
# the write side: server persistence and progress semantics
# --------------------------------------------------------------------------- #

def test_append_stamps_author_and_derives_the_handout_json(tmp_path, monkeypatch) -> None:
    """`author` is stamped server-side because the only downstream consumer
    selects on it: `analysis.helpers.failure_report` and tests/test_hw_holes.py
    filter `author == "human"`. A record without it is invisible."""
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    server.append_annotation({"trace_id": "t1", "turn": 1, "step": "reply", "note": "x"})
    records = server.read_annotations()
    assert len(records) == 1
    assert records[0]["author"] == "human"

    derived = json.loads((tmp_path / "annotations.json").read_text())
    assert list(derived) == ["annotations"], "the handout's shape is {'annotations': [...]}"
    assert derived["annotations"] == records


def test_append_is_a_log_not_an_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    for i in range(3):
        server.append_annotation({"trace_id": f"t{i}", "note": str(i)})
    assert [r["trace_id"] for r in server.read_annotations()] == ["t0", "t1", "t2"]


def test_a_malformed_line_fails_loudly_rather_than_vanishing(tmp_path, monkeypatch) -> None:
    """For a review log that cannot be regenerated, silently skipping a
    truncated line is the wrong failure."""
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    server.append_annotation({"trace_id": "t1"})
    with server.log_path().open("a") as fh:
        fh.write("{not json\n")
    with pytest.raises(json.JSONDecodeError):
        server.read_annotations()


def _write_manifest(batches: dict[str, list[str]]) -> None:
    """A sample_manifest.json in the current state dir: batch name -> trace ids."""
    from analysis.helpers import _state
    _state.write_json(_state.state_path("sample_manifest.json"), {
        "batches": [{"batch": b, "scenarios": []} for b in batches],
        "traces": {t: {"batch": b} for b, ts in batches.items() for t in ts}})


def test_progress_distinguishes_started_from_reviewed(tmp_path, monkeypatch) -> None:
    """One note on turn 1 must not mark a 3-turn conversation reviewed, or the
    coverage number the review summary reports is wrong."""
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server
    _write_manifest({"b1": ["a", "b", "c"]})

    monkeypatch.setattr(server, "VIEWS", [{
        "session_id": "s1", "scenario_id": "support-0000", "user_role": "shopper",
        "totals": {"turns": 3, "tool_calls": 0, "latency_s": 0.0},
        "trace_ids": ["a", "b", "c"],
    }])
    row = server.index_rows([{"trace_id": "a"}])[0]
    assert row["started"] is True
    assert row["reviewed"] is False
    assert row["turns_annotated"] == 1

    full = server.index_rows([{"trace_id": x} for x in "abc"])[0]
    assert full["reviewed"] is True

    p = server.progress([{"trace_id": "a"}])
    assert p["conversations_started"] == 1
    assert p["conversations_reviewed"] == 0
    assert p["conversations_incomplete"] == 1


def test_progress_counts_only_the_sample(tmp_path, monkeypatch) -> None:
    """The log also holds notes on traces outside the sample (the pool is
    served whole); counting them showed an unstarted batch with 17 of 16 traces
    annotated."""
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server
    _write_manifest({"b1": ["a"]})

    monkeypatch.setattr(server, "VIEWS", [{
        "session_id": "s1", "scenario_id": "support-0000", "user_role": "shopper",
        "totals": {"turns": 1, "tool_calls": 0, "latency_s": 0.0},
        "trace_ids": ["a"],
    }, {
        "session_id": "s2", "scenario_id": "support-0001", "user_role": "shopper",
        "totals": {"turns": 1, "tool_calls": 0, "latency_s": 0.0},
        "trace_ids": ["other"],
    }])
    p = server.progress([{"trace_id": "other", "no_failure": True},
                         {"trace_id": "a"}])
    assert p["traces_total"] == 1
    assert p["traces_annotated"] == 1
    assert p["annotations"] == 1
    assert p["no_failure_observed"] == 0


def test_the_cli_renderer_still_matches_the_view_shape() -> None:
    """The CLI broke silently when `policy.retrieved` was split into
    opened/surfaced: 28 loader tests passed while the front end crashed.
    Anything that consumes a view needs a test that runs it."""
    import importlib.util

    # The CLI lives outside the repo, so a grader legitimately skips this.
    path = Path(os.environ.get(
        "CARTWHEEL_CLI_RENDERER",
        Path.home() / "personal/code/courses/0-learn/notes/ai-evals-cohort-6"
                      "/hw-4/scripts/render_trace.py"))
    if not path.exists():
        pytest.skip(f"CLI front end not present at {path}")
    spec = importlib.util.spec_from_file_location("render_trace", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    view = loader.view_for("support-0007")
    for render in (mod.render_plain, mod.render_pretty):
        out = render(view)
        assert "support-0007" in out
        assert "POLICY" in out



# --------------------------------------------------------------------------- #
# retraction: append-only, collapse on read
# --------------------------------------------------------------------------- #

def test_an_edit_appends_and_the_original_stays_in_history(tmp_path, monkeypatch) -> None:
    """hw4.md line 106: the original open code must remain inspectable, so an
    edit may not rewrite or remove anything."""
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    first = server.append_annotation({"trace_id": "t", "step": "reply", "note": "first"})[-1]
    server.append_annotation({"trace_id": "t", "step": "reply", "note": "second",
                              "supersedes": first["id"]})

    live = server.read_annotations()
    assert [r["note"] for r in live] == ["second"]
    assert [r.get("note") for r in server.history()] == ["first", "second"]


def test_a_retraction_removes_the_note_and_its_own_tombstone(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    rec = server.append_annotation({"trace_id": "t", "step": "reply", "note": "oops"})[-1]
    server.append_annotation({"trace_id": "t", "step": "reply",
                              "supersedes": rec["id"], "retracted": True})

    assert server.read_annotations() == []
    assert len(server.history()) == 2, "both records stay in the log"


def test_an_edit_chain_resolves_to_the_last_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    a = server.append_annotation({"trace_id": "t", "step": "reply", "note": "a"})[-1]
    b = server.append_annotation({"trace_id": "t", "step": "reply", "note": "b",
                                  "supersedes": a["id"]})[-1]
    server.append_annotation({"trace_id": "t", "step": "reply", "note": "c",
                              "supersedes": b["id"]})
    assert [r["note"] for r in server.read_annotations()] == ["c"]


def test_a_record_written_before_ids_existed_can_still_be_superseded(
        tmp_path, monkeypatch) -> None:
    """The first four annotations were written without an id. A synthesised id has
    to be stable across reads or nothing could ever reference them."""
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    legacy = {"trace_id": "t", "turn": 1, "step": "reply", "note": "legacy",
              "ts": "2026-09-24T20:00:00Z", "author": "human"}
    server.log_path().parent.mkdir(parents=True, exist_ok=True)
    server.log_path().write_text(json.dumps(legacy) + "\n")

    first_read = server.history()[0]["id"]
    assert first_read.startswith("legacy-")
    assert server.history()[0]["id"] == first_read, "the id must be deterministic"

    server.append_annotation({"trace_id": "t", "step": "reply",
                              "supersedes": first_read, "retracted": True})
    assert server.read_annotations() == []


def test_retracted_notes_stop_counting_in_progress_and_the_graded_json(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server
    _write_manifest({"b1": ["a"]})

    monkeypatch.setattr(server, "VIEWS", [{
        "session_id": "s1", "scenario_id": "support-0000", "user_role": "shopper",
        "totals": {"turns": 1, "tool_calls": 0, "latency_s": 0.0},
        "trace_ids": ["a"],
    }])
    rec = server.append_annotation({"trace_id": "a", "step": "reply",
                                    "no_failure": True})[-1]
    assert server.progress(server.read_annotations())["no_failure_observed"] == 1

    server.append_annotation({"trace_id": "a", "step": "reply",
                              "supersedes": rec["id"], "retracted": True})
    p = server.progress(server.read_annotations())
    assert p["no_failure_observed"] == 0
    assert p["conversations_reviewed"] == 0

    derived = json.loads((tmp_path / "annotations.json").read_text())
    assert derived["annotations"] == [], "the graded file holds the LIVE set"


# --------------------------------------------------------------------------- #
# the HTTP layer. Nothing touched it before, so deleting the whole `supersedes`
# validation block left 34/34 green -- and that block is where the reachable
# defects were.
# --------------------------------------------------------------------------- #

@pytest.fixture
def live_server(tmp_path, monkeypatch):
    """A real server on a free port, state in tmp_path. Offline."""
    import socket
    import threading
    from http.server import ThreadingHTTPServer

    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    monkeypatch.setattr(server, "VIEWS", [{
        "session_id": "s1", "scenario_id": "support-0000", "user_role": "shopper",
        "totals": {"turns": 1, "tool_calls": 0, "latency_s": 0.0},
        "trace_ids": ["a"],
    }])
    monkeypatch.setattr(server, "BY_KEY", {"support-0000": {"world_asof": "2026-07-01"}})
    _write_manifest({"b1": ["a"]})   # "a" is the sample; progress counts it

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", server
    httpd.shutdown()


def _post(base: str, payload, path: str = "/api/annotations") -> tuple[int, dict]:
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _get(base: str, path: str):
    import urllib.request
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.load(r)


# --------------------------------------------------------------------------- #
# pool, sample, batch: the index lists the whole pool and reads batch
# membership from the manifest on every request.
# --------------------------------------------------------------------------- #

def _two_views(server, monkeypatch) -> None:
    """"a" (support-0000) is in batch b1; "z" (support-0001) is pool only."""
    monkeypatch.setattr(server, "VIEWS", server.VIEWS + [{
        "session_id": "s2", "scenario_id": "support-0001", "user_role": "merchant",
        "totals": {"turns": 1, "tool_calls": 0, "latency_s": 0.0},
        "trace_ids": ["z"],
    }])


def test_the_index_lists_the_whole_pool_with_its_batch(live_server, monkeypatch) -> None:
    base, server = live_server
    _two_views(server, monkeypatch)
    rows = {r["scenario_id"]: r for r in _get(base, "/api/index")}
    assert set(rows) == {"support-0000", "support-0001"}
    assert (rows["support-0000"]["batch"], rows["support-0000"]["in_sample"]) == ("b1", True)
    assert (rows["support-0001"]["batch"], rows["support-0001"]["in_sample"]) == (None, False)
    # progress still counts the sample only
    assert _get(base, "/api/progress")["conversations_total"] == 1


def test_a_batch_drawn_while_the_server_runs_appears_without_a_restart(
        live_server, monkeypatch) -> None:
    base, server = live_server
    _two_views(server, monkeypatch)
    assert {r["scenario_id"]: r["batch"] for r in _get(base, "/api/index")}["support-0001"] is None
    _write_manifest({"b1": ["a"], "b2": ["z"]})
    rows = {r["scenario_id"]: r for r in _get(base, "/api/index")}
    assert (rows["support-0001"]["batch"], rows["support-0001"]["in_sample"]) == ("b2", True)
    assert _get(base, "/api/progress")["conversations_total"] == 2


def test_a_note_on_a_trace_outside_the_sample_is_saved(live_server, monkeypatch) -> None:
    base, server = live_server
    _two_views(server, monkeypatch)
    status, _ = _post(base, {"trace_id": "z", "turn": 1, "step": "reply", "no_failure": False,
                             "note": "pool-only failure", "scenario_id": "support-0001"})
    assert status == 200
    assert [a["note"] for a in server.read_annotations() if a["trace_id"] == "z"] == ["pool-only failure"]
    row = next(r for r in _get(base, "/api/index") if r["scenario_id"] == "support-0001")
    assert row["reviewed"] is True and row["in_sample"] is False


def test_the_manifest_status_says_when_the_sample_is_unreadable(live_server) -> None:
    """tools.select_traces() rewrites the manifest without "traces"; the sample
    then reads as empty, so the page must be told rather than show 0 of 0."""
    base, _ = live_server
    ok = _get(base, "/api/manifest")
    assert (ok["manifest_ok"], ok["reason"]) == (True, None)
    from analysis.helpers import _state
    _state.write_json(_state.state_path("sample_manifest.json"),
                      {"source": "x", "k": 1, "strategy": "s", "picks": []})
    status = _get(base, "/api/manifest")
    assert status["manifest_ok"] is False and "traces" in status["reason"]
    _state.state_path("sample_manifest.json").unlink()
    status = _get(base, "/api/manifest")
    assert status["manifest_ok"] is False and "no sample manifest" in status["reason"]


def test_the_manifest_status_lists_batches_with_their_part_b_batch(live_server) -> None:
    """The batch filter groups batches by hw4.md Part B batch (1-4) and labels a
    depth batch with its mode; both come from the manifest, never from the UI."""
    base, _ = live_server
    from analysis.helpers import _state
    _state.write_json(_state.state_path("sample_manifest.json"), {
        "batches": [{"batch": "uniform-1", "part_b_batch": 1, "scenarios": ["s1"]},
                    {"batch": "depth-2", "part_b_batch": 3, "mode": "m2", "scenarios": ["s2"]},
                    {"batch": "recheck-1", "mode": "m2", "counts_toward_batch": False}],
        "traces": {"a": {"scenario_id": "s1", "batch": "uniform-1"}}})
    assert _get(base, "/api/manifest")["batches"] == [
        {"batch": "uniform-1", "part_b_batch": 1, "mode": None},
        {"batch": "depth-2", "part_b_batch": 3, "mode": "m2"},
        {"batch": "recheck-1", "part_b_batch": None, "mode": "m2"}]


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '{"traces": {"a": "b1"}}',
                                  '{"traces": ["a"]}'])
def test_a_malformed_manifest_is_reported_not_raised(live_server, text) -> None:
    """Invalid JSON, a non-object top level or non-object trace entries: the
    page still loads, with an empty sample and a reason to show."""
    base, _ = live_server
    from analysis.helpers import _state
    _state.state_path("sample_manifest.json").write_text(text)
    status = _get(base, "/api/manifest")
    assert status["manifest_ok"] is False and status["reason"]
    (row,) = _get(base, "/api/index")
    assert (row["batch"], row["in_sample"]) == (None, False)
    assert _get(base, "/api/progress")["conversations_total"] == 0


def test_a_trace_entry_without_a_batch_does_not_hide_the_conversations_batch(
        live_server, monkeypatch) -> None:
    base, server = live_server
    monkeypatch.setattr(server, "VIEWS", [{
        "session_id": "s1", "scenario_id": "support-0000", "user_role": "shopper",
        "totals": {"turns": 2, "tool_calls": 0, "latency_s": 0.0},
        "trace_ids": ["a0", "a"],
    }])
    from analysis.helpers import _state
    _state.write_json(_state.state_path("sample_manifest.json"), {
        "batches": [{"batch": "b1", "scenarios": []}],
        "traces": {"a0": {"scenario_id": "support-0000"}, "a": {"batch": "b1"}}})
    (row,) = _get(base, "/api/index")
    assert (row["batch"], row["in_sample"]) == ("b1", True)


def test_post_rejects_a_body_without_a_trace_id(live_server) -> None:
    base, _ = live_server
    assert _post(base, {})[0] == 400


def test_post_rejects_a_retraction_that_names_nothing(live_server) -> None:
    """A tombstone with no target appends a record that is dead on arrival."""
    base, _ = live_server
    code, body = _post(base, {"trace_id": "a", "step": "reply", "retracted": True})
    assert code == 400
    assert "supersedes" in body["error"]


def test_post_rejects_superseding_an_unknown_id(live_server) -> None:
    base, _ = live_server
    code, body = _post(base, {"trace_id": "a", "step": "reply", "supersedes": "nope"})
    assert code == 409
    assert "nope" in body["error"]


def test_post_rejects_superseding_an_already_superseded_id(live_server) -> None:
    """The defect this replaced: validating against the whole log let a stale tab
    supersede one id twice, leaving TWO live notes on one step."""
    base, server = live_server
    _post(base, {"trace_id": "a", "step": "reply", "note": "first"})
    first = server.read_annotations()[0]["id"]
    assert _post(base, {"trace_id": "a", "step": "reply", "note": "second",
                        "supersedes": first})[0] == 200
    code, _ = _post(base, {"trace_id": "a", "step": "reply", "note": "third",
                           "supersedes": first})
    assert code == 409, "a stale second edit of the same id must be refused"
    assert [r["note"] for r in server.read_annotations()] == ["second"]


def test_post_rejects_resurrecting_a_tombstone(live_server) -> None:
    base, server = live_server
    _post(base, {"trace_id": "a", "step": "reply", "note": "x"})
    rec = server.read_annotations()[0]["id"]
    _post(base, {"trace_id": "a", "step": "reply", "supersedes": rec, "retracted": True})
    tomb = [r for r in server.history() if r.get("retracted")][0]["id"]
    assert _post(base, {"trace_id": "a", "step": "reply", "note": "back",
                        "supersedes": tomb})[0] == 409


def test_post_ignores_a_client_supplied_id(live_server) -> None:
    """Two records sharing an id cannot be superseded independently."""
    base, server = live_server
    _post(base, {"trace_id": "a", "step": "reply", "note": "one", "id": "deadbeefcafe"})
    _post(base, {"trace_id": "a", "step": "other", "note": "two", "id": "deadbeefcafe"})
    ids = [r["id"] for r in server.history()]
    assert len(ids) == len(set(ids)) == 2
    assert "deadbeefcafe" not in ids


def test_history_endpoint_reports_superseded_counts(live_server) -> None:
    base, server = live_server
    _post(base, {"trace_id": "a", "step": "reply", "note": "v1"})
    first = server.read_annotations()[0]["id"]
    _post(base, {"trace_id": "a", "step": "reply", "note": "v2", "supersedes": first})
    h = _get(base, "/api/history")
    assert len(h["records"]) == 2, "the original stays in the log"
    assert h["superseded_counts"]["a|reply"] == 1


def test_a_duplicate_id_in_a_hand_edited_log_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    line = json.dumps({"id": "dup", "trace_id": "a", "step": "reply", "note": "x"})
    server.log_path().parent.mkdir(parents=True, exist_ok=True)
    server.log_path().write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate annotation id"):
        server.history()


def test_two_distinct_legacy_records_get_distinct_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    a = {"trace_id": "a", "step": "reply", "note": "one"}
    b = {"trace_id": "a", "step": "reply", "note": "two"}
    server.log_path().parent.mkdir(parents=True, exist_ok=True)
    server.log_path().write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    ids = [r["id"] for r in server.history()]
    assert len(set(ids)) == 2, "records differing only in note text must not collide"
    assert all(i.startswith("legacy-") for i in ids)


def test_a_real_id_can_never_look_like_a_legacy_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    rec = server.append_annotation({"trace_id": "a", "step": "reply", "note": "x"})[-1]
    assert not rec["id"].startswith("legacy-")
    assert len(rec["id"]) == 12


def test_the_graded_json_survives_supersede_records(tmp_path, monkeypatch) -> None:
    """tests/test_hw_holes.py reads annotations.json and filters author == 'human';
    tombstones must never reach it."""
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server

    a = server.append_annotation({"trace_id": "t", "step": "reply", "note": "v1"})[-1]
    server.append_annotation({"trace_id": "t", "step": "reply", "note": "v2",
                              "supersedes": a["id"]})
    derived = json.loads((tmp_path / "annotations.json").read_text())
    assert [r["note"] for r in derived["annotations"]] == ["v2"]
    assert all(r["author"] == "human" for r in derived["annotations"])
    assert not any(r.get("retracted") for r in derived["annotations"])


# --------------------------------------------------------------------------- #
# malformed supersedes graphs. Each of these used to collapse to a SMALLER live
# set -- silently deleting notes from annotations.json, the graded file, while
# history() still showed them on disk. A bad line raises; a bad edge must too.
# --------------------------------------------------------------------------- #

def _write_log(server, *records) -> None:
    server.log_path().parent.mkdir(parents=True, exist_ok=True)
    server.log_path().write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    from analysis.review_app import server
    return server


def test_a_self_supersede_is_refused(state) -> None:
    """Used to collapse to [] -- the record killed itself."""
    _write_log(state, {"id": "a", "trace_id": "t", "step": "reply",
                       "note": "x", "supersedes": "a"})
    with pytest.raises(ValueError, match="supersedes itself"):
        state.history()


def test_a_two_cycle_is_refused(state) -> None:
    """Used to collapse to [] -- both live notes vanished with no error."""
    _write_log(state,
               {"id": "a", "trace_id": "t", "step": "reply", "note": "x", "supersedes": "b"},
               {"id": "b", "trace_id": "t", "step": "reply", "note": "y", "supersedes": "a"})
    with pytest.raises(ValueError, match="cycle"):
        state.history()


def test_a_three_cycle_is_refused(state) -> None:
    _write_log(state,
               {"id": "a", "trace_id": "t", "step": "reply", "supersedes": "b", "note": "1"},
               {"id": "b", "trace_id": "t", "step": "reply", "supersedes": "c", "note": "2"},
               {"id": "c", "trace_id": "t", "step": "reply", "supersedes": "a", "note": "3"})
    with pytest.raises(ValueError, match="cycle"):
        state.history()


def test_superseding_one_record_twice_is_refused(state) -> None:
    """Used to leave BOTH replacements live -- one edit reading as two notes."""
    _write_log(state,
               {"id": "a", "trace_id": "t", "step": "reply", "note": "orig"},
               {"id": "b", "trace_id": "t", "step": "reply", "note": "e1", "supersedes": "a"},
               {"id": "c", "trace_id": "t", "step": "reply", "note": "e2", "supersedes": "a"})
    with pytest.raises(ValueError, match="superseded more than once"):
        state.history()


def test_a_tombstone_naming_no_target_is_refused(state) -> None:
    _write_log(state, {"id": "a", "trace_id": "t", "step": "reply", "retracted": True})
    with pytest.raises(ValueError, match="names no target"):
        state.history()


def test_a_supersedes_naming_a_missing_id_is_an_inert_no_op(state) -> None:
    """Defensible: a phantom edge kills nothing, so nothing is lost."""
    _write_log(state,
               {"id": "a", "trace_id": "t", "step": "reply", "note": "x"},
               {"id": "b", "trace_id": "t", "step": "reply", "note": "y",
                "supersedes": "ghost"})
    assert [r["note"] for r in state.read_annotations()] == ["x", "y"]


def test_a_long_edit_chain_stays_wellformed(state) -> None:
    _write_log(state,
               {"id": "a", "trace_id": "t", "step": "reply", "note": "1"},
               {"id": "b", "trace_id": "t", "step": "reply", "note": "2", "supersedes": "a"},
               {"id": "c", "trace_id": "t", "step": "reply", "note": "3", "supersedes": "b"},
               {"id": "d", "trace_id": "t", "step": "reply", "note": "4", "supersedes": "c"})
    assert [r["note"] for r in state.read_annotations()] == ["4"]


def test_startup_rederives_the_graded_file_after_a_hand_edit(state) -> None:
    """annotations.json is otherwise written only on POST, so a hand edit to the
    jsonl left the graded file a version behind."""
    _write_log(state, {"id": "a", "trace_id": "t", "step": "reply", "note": "typed by hand"})
    graded = state._state.state_path("annotations.json")
    assert not graded.exists()
    assert state.derive_graded_json() == 1
    assert json.loads(graded.read_text())["annotations"][0]["note"] == "typed by hand"


# --------------------------------------------------------------------------- #
# suggestions: agent depth-search hits, decided by the human. hw4.md Part D:
# "Record whether you accepted or rejected each suggestion. Your saved state
# must contain at least one rejected suggestion." So a decision is written back
# and a rejection is KEPT -- the reference interface drops dismissed ones.
# --------------------------------------------------------------------------- #

def _suggestion(i: str = "sg1", **kw) -> dict:
    return {"id": i, "trace_id": "a", "turn": 1, "step": "reply", "session_id": "s1",
            "scenario_id": "support-0000", "mode": "policy_claim_without_opening",
            "signal": "rail: cited but never opened", "evidence": "cites cw-returns",
            "author": "agent", "status": "pending", **kw}


@pytest.fixture
def with_suggestions(live_server):
    base, server = live_server
    from analysis.helpers import _state
    _state.write_json(_state.state_path("suggestions.json"),
                      [_suggestion("sg1"), _suggestion("sg2")])
    # An accept now also assigns the note to the suggestion's mode, and refuses
    # (404, nothing written) when that mode does not exist.
    _state.write_json(_state.state_path("patterns.json"),
                      {"modes": [_mode("policy_claim_without_opening")]})
    return base, server


def _decide(base, **payload):
    return _post(base, payload, "/api/suggestions/decision")


def test_suggestions_are_served(with_suggestions) -> None:
    base, _ = with_suggestions
    assert [x["id"] for x in _get(base, "/api/suggestions")] == ["sg1", "sg2"]


def test_deciding_an_unknown_suggestion_is_404(with_suggestions) -> None:
    base, _ = with_suggestions
    assert _decide(base, id="nope", decision="accepted", reason="x")[0] == 404


def test_a_decision_must_be_accept_or_reject(with_suggestions) -> None:
    base, _ = with_suggestions
    assert _decide(base, id="sg1", decision="maybe")[0] == 400


def test_a_rejection_needs_a_reason(with_suggestions) -> None:
    """The reason is the boundary that excluded the trace -- the video asks for it."""
    base, _ = with_suggestions
    assert _decide(base, id="sg1", decision="rejected", reason="  ")[0] == 400


def test_a_rejection_is_kept_with_its_reason(with_suggestions) -> None:
    base, server = with_suggestions
    assert _decide(base, id="sg1", decision="rejected", reason="opened it after all")[0] == 200
    saved = {x["id"]: x for x in server.read_suggestions()}
    assert saved["sg1"]["status"] == "rejected"
    assert saved["sg1"]["decision"]["reason"] == "opened it after all"
    assert server.read_annotations() == [], "a rejection is not an annotation"


def test_add_as_reason_without_a_reason_uses_the_ai_why(with_suggestions) -> None:
    """Only a rejection needs the reviewer's words; "Add as reason" with none
    takes the suggestion's own why, visibly marked as an accepted suggestion."""
    base, server = with_suggestions
    assert _decide(base, id="sg2", decision="accepted", reason="  ")[0] == 200
    (note,) = server.read_annotations()
    assert note["note"] == "[accepted suggestion, mode policy_claim_without_opening] cites cw-returns"
    assert {x["id"]: x for x in server.read_suggestions()}["sg2"]["decision"]["reason"] == "cites cw-returns"


def test_covered_by_my_reason_needs_no_reason(with_suggestions) -> None:
    base, server = with_suggestions
    n = _fail(server)
    assert _decide(base, id="sg1", decision="accepted", reason="", annotation_id=n["id"])[0] == 200
    assert [a["id"] for a in server.read_annotations()] == [n["id"]]
    assert {x["id"]: x for x in server.read_suggestions()}["sg1"]["decision"]["reason"] == \
        f"covered by reason {n['id']}"


def test_accepting_writes_a_human_annotation(with_suggestions) -> None:
    """An accepted suggestion becomes an observation the human accepted."""
    base, server = with_suggestions
    assert _decide(base, id="sg2", decision="accepted", reason="cited cw-returns unread")[0] == 200
    (note,) = server.read_annotations()
    assert note["author"] == "human" and note["source"] == "accepted_suggestion"
    assert note["no_failure"] is False and note["trace_id"] == "a"
    assert note["suggestion_id"] == "sg2" and "policy_claim_without_opening" in note["note"]
    assert "cited cw-returns unread" in note["note"]
    saved = {x["id"]: x for x in server.read_suggestions()}
    assert saved["sg2"]["status"] == "accepted"
    assert saved["sg2"]["decision"]["annotation_id"] == note["id"]


def test_the_same_decision_twice_is_409(with_suggestions) -> None:
    base, _ = with_suggestions
    assert _decide(base, id="sg1", decision="rejected", reason="x")[0] == 200
    assert _decide(base, id="sg1", decision="rejected", reason="x")[0] == 409


def test_a_rejection_reason_can_be_revised(with_suggestions) -> None:
    base, server = with_suggestions
    assert _decide(base, id="sg1", decision="rejected", reason="x")[0] == 200
    assert _decide(base, id="sg1", decision="rejected", reason="revised boundary")[0] == 200
    sg = {x["id"]: x for x in server.read_suggestions()}["sg1"]
    assert [d["reason"] for d in sg["decisions"]] == ["x", "revised boundary"]


def test_an_edited_accepted_note_is_still_retracted_on_reject(with_suggestions) -> None:
    """The UI edit path posts a new record without suggestion_id; the server keeps
    the lineage, or rejecting would leave a live failure note behind."""
    base, server = with_suggestions
    assert _decide(base, id="sg2", decision="accepted", reason="first words")[0] == 200
    (note,) = server.read_annotations()
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "no_failure": False,
                        "note": "edited words", "supersedes": note["id"]})[0] == 200
    (edited,) = server.read_annotations()
    assert edited["suggestion_id"] == "sg2" and edited["source"] == "accepted_suggestion"
    assert _decide(base, id="sg2", decision="rejected", reason="on reflection no")[0] == 200
    assert server.read_annotations() == []


def test_reaccepting_restores_a_hand_retracted_note(with_suggestions) -> None:
    base, server = with_suggestions
    assert _decide(base, id="sg2", decision="accepted", reason="yes")[0] == 200
    (note,) = server.read_annotations()
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply",
                        "supersedes": note["id"], "retracted": True})[0] == 200
    assert server.read_annotations() == []
    assert _decide(base, id="sg2", decision="accepted", reason="yes")[0] == 200
    assert len(server.read_annotations()) == 1


def test_redeciding_appends_and_retracts_the_accepted_note(with_suggestions) -> None:
    """Append-only like the annotation log: the first call stays in `decisions`,
    the latest wins, and un-accepting withdraws the note acceptance created."""
    base, server = with_suggestions
    assert _decide(base, id="sg1", decision="accepted", reason="yes")[0] == 200
    assert len(server.read_annotations()) == 1
    assert _decide(base, id="sg1", decision="rejected", reason="revised: snippet was enough")[0] == 200
    assert server.read_annotations() == [], "the accepted note is retracted"
    sg = {x["id"]: x for x in server.read_suggestions()}["sg1"]
    assert sg["status"] == "rejected"
    assert [d["status"] for d in sg["decisions"]] == ["accepted", "rejected"]
    assert len(server.history()) == 2, "note and tombstone both stay in the log"


def test_accepting_is_idempotent_on_an_existing_note(with_suggestions) -> None:
    """A retry after a partial failure must not append a second note."""
    base, server = with_suggestions
    server.append_annotation({"trace_id": "a", "turn": 1, "step": "reply", "no_failure": False,
                              "note": "earlier", "suggestion_id": "sg2"})
    assert _decide(base, id="sg2", decision="accepted", reason="again")[0] == 200
    assert len(server.read_annotations()) == 1


def test_patterns_are_served_for_mode_definitions(live_server) -> None:
    base, _ = live_server
    from analysis.helpers import _state
    _state.write_json(_state.state_path("patterns.json"), {"modes": [{"name": "m", "definition": "d"}]})
    assert _get(base, "/api/patterns")["modes"][0]["definition"] == "d"


def test_depth_search_emits_one_hit_per_matching_turn(pool) -> None:
    """support-0212 states an override rule in both turns; pinning the suggestion
    to the first match would leave the stronger turn-2 claim unjudged."""
    from analysis.review_app import depth_search

    views = {v["scenario_id"]: v for v in loader.all_views(pool)}
    strength, hits = depth_search.policy_claim_without_opening(views["support-0212"])
    assert strength == "weak" and [t["turn"] for t, _ in hits] == [1, 2]
    strength, hits = depth_search.policy_claim_without_opening(views["support-0161"])
    assert strength == "strong" and [t["turn"] for t, _ in hits] == [2]
    assert depth_search.policy_claim_without_opening(views["support-0109"]) is None, \
        "opened and cited cw-returns: no hit"


def test_an_accepted_note_cannot_be_flipped_to_no_failure(with_suggestions) -> None:
    """The suggestion would say accepted while its own note said no failure."""
    base, server = with_suggestions
    assert _decide(base, id="sg2", decision="accepted", reason="yes")[0] == 200
    (note,) = server.read_annotations()
    status, body = _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "no_failure": True,
                                "note": "", "supersedes": note["id"]})
    assert status == 409 and "reject the suggestion" in body["error"]
    assert server.read_annotations()[0]["id"] == note["id"]


# -- the merged verdict box: one click writes the verdict AND the decision ------

def test_reject_with_a_no_failure_verdict_writes_both_records(with_suggestions) -> None:
    base, server = with_suggestions
    assert _decide(base, id="sg1", decision="rejected", reason="opened the store page",
                   verdict={"no_failure": True})[0] == 200
    (note,) = server.read_annotations()
    assert note["no_failure"] is True and note["note"] == "opened the store page"
    assert "suggestion_id" not in note, "a verdict is not tied to the suggestion"
    assert {x["id"]: x for x in server.read_suggestions()}["sg1"]["status"] == "rejected"


def test_accepting_after_a_no_failure_reject_withdraws_that_verdict(with_suggestions) -> None:
    """support-0177: the turn held "No failure observed" and the accepted failure
    at once until the no-failure note was retracted by hand."""
    base, server = with_suggestions
    assert _decide(base, id="sg1", decision="rejected", reason="snippet was enough",
                   verdict={"no_failure": True})[0] == 200
    (verdict,) = server.read_annotations()
    assert _decide(base, id="sg1", decision="accepted", reason="cited unopened")[0] == 200
    (note,) = server.read_annotations()
    assert note["no_failure"] is False and note["suggestion_id"] == "sg1"
    sg = {x["id"]: x for x in server.read_suggestions()}["sg1"]
    assert sg["decision"]["retracted_verdict_id"] == verdict["id"]


def test_the_scan_flags_opposite_verdicts_on_one_step_only(live_server) -> None:
    """Same step: a conflict (support-0177 before the fix). Different steps of one
    turn: a step-level reclassification (support-0007), not flagged."""
    _, server = live_server
    note = lambda i, step, ok: {"id": i, "trace_id": "t", "turn": 1, "step": step, "no_failure": ok}
    rows = server.contradictory_verdicts([note("f", "reply", False), note("n", "reply", True),
                                          note("g", "tool:get_order#1", False), note("m", "agent#6", True)])
    assert rows == [{"trace_id": "t", "turn": 1, "step": "reply",
                     "failure_ids": ["f"], "no_failure_ids": ["n"]}]


def test_reject_with_a_different_failure_writes_both_records(with_suggestions) -> None:
    base, server = with_suggestions
    assert _decide(base, id="sg1", decision="rejected", reason="wrong date, not this mode",
                   verdict={"no_failure": False})[0] == 200
    (note,) = server.read_annotations()
    assert note["no_failure"] is False and note["note"] == "wrong date, not this mode"
    # re-accepting later must not retract the human's own verdict
    assert _decide(base, id="sg1", decision="accepted", reason="on reflection yes")[0] == 200
    assert len(server.read_annotations()) == 2


def test_a_malformed_verdict_is_400(with_suggestions) -> None:
    base, server = with_suggestions
    assert _decide(base, id="sg1", decision="rejected", reason="x", verdict={"no_failure": "yes"})[0] == 400
    assert server.read_annotations() == []


def test_a_quote_starts_at_a_sentence_boundary(pool) -> None:
    """The fixed +-110-char window started 9 of 14 stored quotes mid-word
    ("…ome Goods’ 14-day policy")."""
    from analysis.review_app import depth_search

    views = {v["scenario_id"]: v for v in loader.all_views(pool)}
    _, hits = depth_search.policy_claim_without_opening(views["support-0221"])
    (turn, hit), = [(t, h) for t, h in hits if t["turn"] == 1]
    q, reply = hit["quote"], turn["reply"]
    assert "14-day policy" in q and not q.startswith("…")
    at = reply.find(q)
    assert at >= 0, "the quote must be verbatim from the reply"
    assert at == 0 or reply[at - 1].isspace(), f"starts mid-word: {q[:40]!r}"


def test_a_quote_on_a_bullet_line_stays_on_that_line() -> None:
    from analysis.review_app import depth_search

    text = "Here is what applies:\n- Tracking: none\n- The store policy wins here\n- Refund: 5 days"
    q = depth_search._quote(text, depth_search._RULE)
    assert q == "The store policy wins here"
    assert "\n" not in q and "Refund" not in q and "Tracking" not in q


def test_a_quote_is_capped_on_a_long_unpunctuated_string() -> None:
    from analysis.review_app import depth_search

    text = "word " * 200 + "takes precedence" + " word" * 200
    q = depth_search._quote(text, depth_search._RULE)
    assert "takes precedence" in q and len(q) <= 302
    assert q.startswith("…") and q.endswith("…")


def test_a_hit_carries_reason_and_quote_and_the_fused_evidence(pool) -> None:
    from analysis.review_app import depth_search

    views = {v["scenario_id"]: v for v in loader.all_views(pool)}
    for sid, strength in (("support-0161", "strong"), ("support-0212", "weak")):
        got, hits = depth_search.policy_claim_without_opening(views[sid])
        assert got == strength
        for _, h in hits:
            assert h["reason"] and h["quote"]
            assert h["evidence"] == f"{h['reason']}: {h['quote']}"


def test_the_backfill_only_adds_reason_and_quote(pool, tmp_path, monkeypatch) -> None:
    """Records carry the reviewer's decisions; the backfill must not touch them."""
    from analysis.helpers import _state
    from analysis.review_app import depth_search

    real = json.loads((loader.REPO / "analysis" / "state" / "suggestions.json").read_text())
    by_id = {s["id"]: s for s in real}
    seed = [by_id[f"policy_claim_without_opening:r1:{k}"]
            for k in ("support-0221:t1", "support-0212:t2", "support-0123:t1")]
    seed = [{k: v for k, v in s.items() if k not in ("reason", "quote")} for s in seed]
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    _state.write_json(_state.state_path("suggestions.json"), seed)

    n, missed = depth_search.backfill_reason_quote(pool)
    after = _state.read_json(_state.state_path("suggestions.json"))
    assert (n, missed) == (3, [])
    for before, now in zip(seed, after):
        assert {k: now[k] for k in before} == before
        assert set(now) - set(before) == {"reason", "quote"}
        assert now["quote"] and now["reason"]


# --------------------------------------------------------------------------- #
# Taxonomy: assigning a failure note to a mode.
# --------------------------------------------------------------------------- #

def _mode(name: str, **kw) -> dict:
    return {"name": name, "definition": "d", "example_trace_ids": [], "close_negative_trace_ids": [],
            "source_annotation_ids": [], "scenarios": {"positive": [], "close_negative": []},
            "count": 0, **kw}


@pytest.fixture
def with_modes(live_server):
    base, server = live_server
    from analysis.helpers import _state
    _state.write_json(_state.state_path("patterns.json"), {"modes": [
        _mode("m", example_trace_ids=["in"], scenarios={"positive": ["support-0001"],
                                                         "close_negative": ["support-0000"]},
              close_negative_trace_ids=["a"], count=1),
        _mode("other"),
    ]})
    return base, server


def _assign(base, **payload):
    return _post(base, payload, "/api/patterns/assign")


def test_unassigned_failures_are_failure_notes_in_no_mode(live_server) -> None:
    _, server = live_server
    note = lambda i, t, ok: {"id": i, "trace_id": t, "turn": 1, "step": "reply",
                             "scenario_id": "s-" + t, "no_failure": ok, "note": "n" + i}
    rows = server.unassigned_failures(
        [note("1", "in", False), note("2", "out", False), note("3", "fine", True)],
        [_mode("m", example_trace_ids=["in"], source_annotation_ids=["1"])])
    assert rows == [{"trace_id": "out", "turn": 1, "step": "reply", "scenario_id": "s-out",
                     "note": "n2", "annotation_id": "2", "in_modes": []}]


def test_assigning_a_failure_adds_it_as_a_positive(with_modes) -> None:
    base, server = with_modes
    server.append_annotation({"trace_id": "a", "turn": 1, "step": "reply", "scenario_id": "support-0000",
                              "no_failure": False, "note": "cited unread"})
    (note,) = server.read_annotations()
    assert server.unassigned_failures(server.read_annotations(), [_mode("m")])[0]["trace_id"] == "a"
    assert _assign(base, trace_id="a", mode="m")[0] == 200
    from analysis.helpers import _state
    (m, _) = _state.read_json(_state.state_path("patterns.json"))["modes"]
    assert m["example_trace_ids"] == ["in", "a"]
    assert m["scenarios"]["positive"] == ["support-0001", "support-0000"]
    assert m["source_annotation_ids"] == [note["id"]]
    assert m["count"] == 2
    # Close negatives are curated by hand; assign never touches them.
    assert m["scenarios"]["close_negative"] == ["support-0000"] and m["close_negative_trace_ids"] == ["a"]
    tax = _get(base, "/api/taxonomy")
    assert tax["unassigned"] == []
    row = tax["modes"][0]["examples"][1]
    assert {k: row[k] for k in ("trace_id", "scenario_id", "turn", "note", "annotation_id")} == {
        "trace_id": "a", "scenario_id": "support-0000", "turn": 1, "note": "cited unread",
        "annotation_id": note["id"]}


def test_assigning_to_an_unknown_mode_is_404(with_modes) -> None:
    base, server = with_modes
    server.append_annotation({"trace_id": "a", "no_failure": False, "note": "x"})
    assert _assign(base, trace_id="a", mode="nope")[0] == 404


def test_assigning_a_trace_without_a_failure_note_is_409(with_modes) -> None:
    base, server = with_modes
    server.append_annotation({"trace_id": "a", "no_failure": True, "note": "fine"})
    assert _assign(base, trace_id="a", mode="m")[0] == 409


def test_assigning_a_trace_already_in_the_mode_is_409(with_modes) -> None:
    base, server = with_modes
    server.append_annotation({"trace_id": "in", "no_failure": False, "note": "x"})
    assert _assign(base, trace_id="in", mode="m")[0] == 409


# --------------------------------------------------------------------------- #
# Axial coding: a mode is attached to a NOTE (annotation id); counts are per
# trace and per conversation, recomputed after every patterns write.
# --------------------------------------------------------------------------- #

def _fail(server, trace="a", turn=1, step="reply", scenario="support-0000", note="x", **kw):
    """Append one live failure note and return it."""
    everything = server.append_annotation({"trace_id": trace, "turn": turn, "step": step,
                                           "scenario_id": scenario, "no_failure": False,
                                           "note": note, **kw})
    return everything[-1]


def _modes():
    from analysis.helpers import _state
    return {m["name"]: m for m in _state.read_json(_state.state_path("patterns.json"))["modes"]}


def _unassign(base, **payload):
    return _post(base, payload, "/api/patterns/unassign")


def test_assign_by_annotation_id_records_that_exact_note(with_modes) -> None:
    """Two notes on one trace, two different failures: each mode gets its own note,
    not the trace's latest one."""
    base, server = with_modes
    n1 = _fail(server, step="tool:get_order#0", note="wrong order read")
    n2 = _fail(server, note="cited cw-returns unread")
    assert _assign(base, annotation_id=n1["id"], mode="other")[0] == 200
    assert _assign(base, annotation_id=n2["id"], mode="m")[0] == 200
    modes = _modes()
    assert modes["other"]["source_annotation_ids"] == [n1["id"]]
    assert modes["m"]["source_annotation_ids"] == [n2["id"]]
    assert modes["other"]["example_trace_ids"] == ["a"]
    assert modes["m"]["example_trace_ids"] == ["in", "a"]
    tax = _get(base, "/api/taxonomy")
    other = next(m for m in tax["modes"] if m["name"] == "other")
    assert other["trace_count"] == 1 and other["conversation_count"] == 1
    (row,) = other["examples"]
    assert row["annotation_id"] == n1["id"] and row["note"] == "wrong order read"
    assert row["in_modes"] == ["m"] and row["from_suggestion"] is False
    assert other["closest_example"] == {"scenario_id": "support-0000", "note": "wrong order read"}


def test_assigning_one_note_twice_is_409(with_modes) -> None:
    base, server = with_modes
    n = _fail(server)
    assert _assign(base, annotation_id=n["id"], mode="other")[0] == 200
    assert _assign(base, annotation_id=n["id"], mode="other")[0] == 409


def test_assigning_a_no_failure_note_is_409(with_modes) -> None:
    base, server = with_modes
    ok = server.append_annotation({"trace_id": "a", "no_failure": True, "note": "fine"})[-1]
    assert _assign(base, annotation_id=ok["id"], mode="other")[0] == 409
    assert _modes()["other"]["source_annotation_ids"] == []


def test_assigning_a_note_to_an_unknown_mode_is_404(with_modes) -> None:
    base, server = with_modes
    n = _fail(server)
    assert _assign(base, annotation_id=n["id"], mode="nope")[0] == 404


def test_unassign_unknown_mode_is_404_and_a_note_not_in_it_is_409(with_modes) -> None:
    base, server = with_modes
    n = _fail(server)
    assert _unassign(base, annotation_id=n["id"], mode="nope")[0] == 404
    assert _unassign(base, annotation_id=n["id"], mode="other")[0] == 409


def test_removing_a_note_keeps_the_trace_only_while_another_source_note_is_on_it(with_modes) -> None:
    base, server = with_modes
    n1 = _fail(server, step="tool:get_order#0", note="first")
    n2 = _fail(server, note="second")
    assert _assign(base, annotation_id=n1["id"], mode="other")[0] == 200
    assert _assign(base, annotation_id=n2["id"], mode="other")[0] == 200
    assert _modes()["other"]["example_trace_ids"] == ["a"]
    assert _unassign(base, annotation_id=n1["id"], mode="other")[0] == 200
    m = _modes()["other"]
    assert m["source_annotation_ids"] == [n2["id"]] and m["example_trace_ids"] == ["a"]
    assert _unassign(base, annotation_id=n2["id"], mode="other")[0] == 200
    m = _modes()["other"]
    assert m["source_annotation_ids"] == [] and m["example_trace_ids"] == []
    assert m["scenarios"]["positive"] == [] and m["count"] == 0


def test_a_two_turn_conversation_counts_once_until_both_turns_leave(with_modes) -> None:
    """support-0072's shape: the same failure on two turns (two traces) of one
    conversation. Removing one turn keeps the conversation."""
    base, server = with_modes
    t1 = _fail(server, trace="t1", turn=1, scenario="support-0072")
    t2 = _fail(server, trace="t2", turn=2, scenario="support-0072")
    assert _assign(base, annotation_id=t1["id"], mode="other")[0] == 200
    assert _assign(base, annotation_id=t2["id"], mode="other")[0] == 200
    m = _modes()["other"]
    assert m["example_trace_ids"] == ["t1", "t2"]
    assert m["scenarios"]["positive"] == ["support-0072"] and m["count"] == 1
    other = next(x for x in _get(base, "/api/taxonomy")["modes"] if x["name"] == "other")
    assert (other["trace_count"], other["conversation_count"]) == (2, 1)
    assert _unassign(base, annotation_id=t1["id"], mode="other")[0] == 200
    m = _modes()["other"]
    assert m["example_trace_ids"] == ["t2"]
    assert m["scenarios"]["positive"] == ["support-0072"] and m["count"] == 1
    assert _unassign(base, annotation_id=t2["id"], mode="other")[0] == 200
    m = _modes()["other"]
    assert m["scenarios"]["positive"] == [] and m["count"] == 0


def test_a_failure_note_edited_to_no_failure_is_a_gone_example(with_modes) -> None:
    base, server = with_modes
    n = _fail(server, note="looked wrong")
    assert _assign(base, annotation_id=n["id"], mode="other")[0] == 200
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "no_failure": True,
                        "note": "actually fine", "supersedes": n["id"]})[0] == 200
    (row,) = [g for g in server.gone_examples(server.read_annotations(), list(_modes().values()))
              if g["mode"] == "other"]
    assert row["trace_id"] == "a" and row["scenario_id"] == "support-0000" and row["turn"] == 1
    assert row["kind"] == "note_gone" and row["annotation_id"] == n["id"]
    assert row["live_note"]["note"] == "actually fine" and row["live_note"]["no_failure"] is True
    assert [g["trace_id"] for g in _get(base, "/api/taxonomy")["gone"] if g["mode"] == "other"] == ["a"]
    assert _get(base, "/api/progress")["examples_with_note_gone"] >= 1
    assert _unassign(base, annotation_id=n["id"], mode="other")[0] == 200
    m = _modes()["other"]
    assert m["example_trace_ids"] == [] and m["source_annotation_ids"] == [] and m["count"] == 0
    assert [g for g in _get(base, "/api/taxonomy")["gone"] if g["mode"] == "other"] == []


def test_editing_an_assigned_failure_note_keeps_it_in_the_mode(with_modes) -> None:
    """An edit supersedes the note (new id). The mode follows the edit, or the
    rewritten note would read as unassigned and its old id as evidence."""
    base, server = with_modes
    n = _fail(server, note="first words")
    assert _assign(base, annotation_id=n["id"], mode="other")[0] == 200
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "no_failure": False,
                        "note": "better words", "supersedes": n["id"]})[0] == 200
    (edited,) = server.read_annotations()
    assert _modes()["other"]["source_annotation_ids"] == [edited["id"]]
    tax = _get(base, "/api/taxonomy")
    assert tax["unassigned"] == [] and [g for g in tax["gone"] if g["mode"] == "other"] == []


def test_a_second_failure_on_an_assigned_trace_is_still_unassigned(live_server) -> None:
    _, server = live_server
    note = lambda i, t: {"id": i, "trace_id": t, "turn": 1, "step": "reply",
                         "scenario_id": "s-" + t, "no_failure": False, "note": "n" + i}
    rows = server.unassigned_failures(
        [note("1", "a"), note("2", "a")],
        [_mode("m", example_trace_ids=["a"], source_annotation_ids=["1"])])
    assert [(r["annotation_id"], r["in_modes"]) for r in rows] == [("2", ["m"])]


def test_accept_assigns_the_note_to_the_mode_and_reject_removes_it(with_suggestions) -> None:
    base, server = with_suggestions
    status, body = _decide(base, id="sg2", decision="accepted", reason="cited unread")
    assert status == 200
    (note,) = server.read_annotations()
    m = _modes()["policy_claim_without_opening"]
    assert m["source_annotation_ids"] == [note["id"]] and m["example_trace_ids"] == ["a"]
    assert m["count"] == 1
    assert body["mode_update"] == [{"mode": "policy_claim_without_opening", "action": "added", "count": 1}]
    tax = _get(base, "/api/taxonomy")
    assert tax["modes"][0]["examples"][0]["from_suggestion"] is True
    status, body = _decide(base, id="sg2", decision="rejected", reason="on reflection no")
    assert status == 200
    m = _modes()["policy_claim_without_opening"]
    assert m["source_annotation_ids"] == [] and m["example_trace_ids"] == [] and m["count"] == 0
    assert body["mode_update"] == [{"mode": "policy_claim_without_opening", "action": "removed", "count": 0}]


def test_accept_keeps_an_existing_assignment_silently(with_suggestions) -> None:
    """A retry after the note is already in the mode is not an error."""
    base, server = with_suggestions
    assert _decide(base, id="sg2", decision="accepted", reason="yes")[0] == 200
    (note,) = server.read_annotations()
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply",
                        "supersedes": note["id"], "retracted": True})[0] == 200
    # re-accept restores a new note; the mode gains it, the dead id is left for cleanup
    assert _decide(base, id="sg2", decision="accepted", reason="yes")[0] == 200
    (again,) = server.read_annotations()
    assert again["id"] in _modes()["policy_claim_without_opening"]["source_annotation_ids"]


def test_accepting_into_a_missing_mode_is_404_and_writes_nothing(with_suggestions) -> None:
    base, server = with_suggestions
    from analysis.helpers import _state
    _state.write_json(_state.state_path("suggestions.json"), [_suggestion("sg3", mode="ghost")])
    before = _state.read_json(_state.state_path("patterns.json"))
    assert _decide(base, id="sg3", decision="accepted", reason="looks right")[0] == 404
    assert server.history() == []
    assert server.read_suggestions()[0]["status"] == "pending"
    assert "decisions" not in server.read_suggestions()[0]
    assert _state.read_json(_state.state_path("patterns.json")) == before


# --------------------------------------------------------------------------- #
# Review fixes: close negatives are hand-curated, evidence is checked per note,
# a rejected suggestion's note leaves every mode.
# --------------------------------------------------------------------------- #

def _close_negatives(name: str) -> tuple:
    m = _modes()[name]
    return list(m["close_negative_trace_ids"]), list(m["scenarios"]["close_negative"])


def test_assign_then_unassign_leaves_close_negatives_identical(with_modes) -> None:
    """A misclick on a close-negative trace must not lose curated data."""
    base, server = with_modes
    before = _close_negatives("m")
    n = _fail(server)          # trace "a" is a close negative of m
    assert _assign(base, annotation_id=n["id"], mode="m")[0] == 200
    assert _close_negatives("m") == before
    assert _unassign(base, annotation_id=n["id"], mode="m")[0] == 200
    assert _close_negatives("m") == before == (["a"], ["support-0000"])


def test_taxonomy_reports_a_trace_listed_as_both_example_and_close_negative(with_modes) -> None:
    base, server = with_modes
    n = _fail(server)
    assert _assign(base, annotation_id=n["id"], mode="m")[0] == 200
    modes = {m["name"]: m for m in _get(base, "/api/taxonomy")["modes"]}
    assert modes["m"]["conflicts"] == [{"trace_id": "a", "scenario_id": "support-0000", "turn": 1}]
    assert modes["other"]["conflicts"] == []


def test_a_retracted_source_note_is_gone_even_while_another_failure_is_live(with_modes) -> None:
    """Per note, not per trace: the trace still has a live failure note, but not
    the one that is the mode's evidence."""
    base, server = with_modes
    n1 = _fail(server, step="tool:get_order#0", note="wrong order read")
    n2 = _fail(server, note="cited unread")
    assert _assign(base, annotation_id=n1["id"], mode="other")[0] == 200
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "tool:get_order#0",
                        "supersedes": n1["id"], "retracted": True})[0] == 200
    rows = [g for g in _get(base, "/api/taxonomy")["gone"] if g["mode"] == "other"]
    assert len(rows) == 1
    (row,) = rows
    assert {k: row[k] for k in ("kind", "annotation_id", "trace_id", "scenario_id", "turn")} == {
        "kind": "note_gone", "annotation_id": n1["id"], "trace_id": "a",
        "scenario_id": "support-0000", "turn": 1}
    assert row["live_note"]["id"] == n2["id"]
    assert _get(base, "/api/progress")["examples_with_note_gone"] == 1
    # Remove by the dead id: it was the only source id of `other` on the trace.
    assert _unassign(base, annotation_id=n1["id"], mode="other")[0] == 200
    m = _modes()["other"]
    assert m["source_annotation_ids"] == [] and m["example_trace_ids"] == []
    assert [g for g in _get(base, "/api/taxonomy")["gone"] if g["mode"] == "other"] == []


def test_an_example_trace_with_no_source_note_is_a_no_note_row(with_modes) -> None:
    """m lists trace "in" as an example with no source id. A live failure note on
    it does not make it evidence until that note is assigned."""
    base, server = with_modes
    n = _fail(server, trace="in", scenario="support-0001")
    rows = [g for g in _get(base, "/api/taxonomy")["gone"] if g["mode"] == "m"]
    assert rows == [{"mode": "m", "kind": "no_note", "trace_id": "in",
                     "scenario_id": "support-0001", "turn": 1}]
    # "in" is not in the served set: progress counts only served traces.
    assert _get(base, "/api/progress")["examples_with_note_gone"] == 0
    assert _assign(base, annotation_id=n["id"], mode="m")[0] == 200
    assert [g for g in _get(base, "/api/taxonomy")["gone"] if g["mode"] == "m"] == []


def test_rejecting_removes_the_suggestion_note_from_every_mode(with_suggestions) -> None:
    base, server = with_suggestions
    from analysis.helpers import _state
    assert _decide(base, id="sg2", decision="accepted", reason="cited unread")[0] == 200
    (note,) = server.read_annotations()
    data = _state.read_json(_state.state_path("patterns.json"))
    data["modes"].append(_mode("other"))
    _state.write_json(_state.state_path("patterns.json"), data)
    assert _assign(base, annotation_id=note["id"], mode="other")[0] == 200
    status, body = _decide(base, id="sg2", decision="rejected", reason="on reflection no")
    assert status == 200
    for name in ("policy_claim_without_opening", "other"):
        m = _modes()[name]
        assert m["source_annotation_ids"] == [] and m["example_trace_ids"] == [] and m["count"] == 0
    assert sorted(u["mode"] for u in body["mode_update"]) == ["other", "policy_claim_without_opening"]


def test_removing_a_note_keeps_the_trace_while_a_dead_source_id_of_the_mode_is_on_it(with_modes) -> None:
    base, server = with_modes
    n1 = _fail(server, step="tool:get_order#0", note="first")
    n2 = _fail(server, note="second")
    assert _assign(base, annotation_id=n1["id"], mode="other")[0] == 200
    assert _assign(base, annotation_id=n2["id"], mode="other")[0] == 200
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply",
                        "supersedes": n2["id"], "retracted": True})[0] == 200
    assert _unassign(base, annotation_id=n1["id"], mode="other")[0] == 200
    m = _modes()["other"]
    assert m["source_annotation_ids"] == [n2["id"]] and m["example_trace_ids"] == ["a"]
    # And the unit: the remaining id is known only from the log, not the live set.
    mode = _mode("x", example_trace_ids=["t"], source_annotation_ids=["p", "q"])
    server._remove_note(mode, "p", {"p": {"trace_id": "t"}, "q": {"trace_id": "t", "retracted": True}})
    assert mode["source_annotation_ids"] == ["q"] and mode["example_trace_ids"] == ["t"]


# --------------------------------------------------------------------------- #
# withdrawal with a reason: the retracted note stays visible, struck through
# --------------------------------------------------------------------------- #

def test_a_retraction_stores_its_reason_and_the_note_is_listed_as_withdrawn(live_server) -> None:
    """support-0007 turn 1: a reviewer reclassifying a failure had no way to say
    why, so edited it to no_failure instead -- two contradictory verdicts."""
    base, server = live_server
    _post(base, {"trace_id": "a", "turn": 1, "step": "tool:get_order#0", "note": "wrong order",
                 "no_failure": False, "scenario_id": "support-0000", "ts": "2026-09-01T00:00:00Z"})
    note = server.read_annotations()[0]
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "tool:get_order#0",
                        "supersedes": note["id"], "retracted": True,
                        "reason": "  reclassified: the failure is the reply  ",
                        "ts": "2026-09-02T00:00:00Z"})[0] == 200
    tomb = [r for r in server.history() if r.get("retracted")][0]
    assert tomb["reason"] == "reclassified: the failure is the reply"
    assert server.read_annotations() == [], "the withdrawn note leaves the live set"
    assert server.withdrawn_notes(server.history()) == [{
        "trace_id": "a", "turn": 1, "step": "tool:get_order#0", "scenario_id": "support-0000",
        "annotation_id": note["id"], "note": "wrong order", "no_failure": False,
        "reason": "reclassified: the failure is the reply", "ts": "2026-09-02T00:00:00Z"}]


def test_a_retraction_without_a_reason_is_withdrawn_with_an_empty_reason(live_server) -> None:
    base, server = live_server
    _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "note": "x"})
    note = server.read_annotations()[0]
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply",
                        "supersedes": note["id"], "retracted": True})[0] == 200
    (row,) = server.withdrawn_notes(server.history())
    assert row["annotation_id"] == note["id"] and row["reason"] == ""
    tomb = [r for r in server.history() if r.get("retracted")][0]
    assert tomb.get("reason", "") == ""


def test_a_non_string_retraction_reason_is_400(live_server) -> None:
    base, server = live_server
    _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "note": "x"})
    note = server.read_annotations()[0]
    code, _ = _post(base, {"trace_id": "a", "turn": 1, "step": "reply",
                           "supersedes": note["id"], "retracted": True, "reason": 5})
    assert code == 400
    assert len(server.read_annotations()) == 1


def test_retracting_a_withdrawn_note_again_is_409(live_server) -> None:
    base, server = live_server
    _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "note": "x"})
    note = server.read_annotations()[0]
    body = {"trace_id": "a", "turn": 1, "step": "reply", "supersedes": note["id"],
            "retracted": True, "reason": "dup"}
    assert _post(base, body)[0] == 200
    assert _post(base, body)[0] == 409
    assert len(server.withdrawn_notes(server.history())) == 1


def test_an_edited_note_is_not_listed_as_withdrawn(live_server) -> None:
    base, server = live_server
    _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "note": "v1"})
    v1 = server.read_annotations()[0]["id"]
    _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "note": "v2", "supersedes": v1})
    assert server.withdrawn_notes(server.history()) == []
    # withdrawing the edit lists the edit's text, not the original's
    v2 = server.read_annotations()[0]["id"]
    _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "supersedes": v2,
                 "retracted": True, "reason": "passes"})
    rows = server.withdrawn_notes(server.history())
    assert [(r["annotation_id"], r["note"], r["reason"]) for r in rows] == [(v2, "v2", "passes")]


def test_withdrawn_endpoint_serves_the_rows(live_server) -> None:
    base, server = live_server
    assert _get(base, "/api/withdrawn") == []
    _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "note": "x"})
    note = server.read_annotations()[0]
    _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "supersedes": note["id"],
                 "retracted": True, "reason": "not a failure"})
    assert _get(base, "/api/withdrawn") == server.withdrawn_notes(server.history())
    assert _get(base, "/api/withdrawn")[0]["reason"] == "not a failure"


# --------------------------------------------------------------------------- #
# "Covered by my reason": accept attaches the reviewer's existing failure note
# to the suggestion's mode instead of writing a second note.
# --------------------------------------------------------------------------- #

M = "policy_claim_without_opening"


def _attached(server, sid="sg1") -> dict | None:
    return ({x["id"]: x for x in server.read_suggestions()}[sid]["decision"] or {}).get("attached")


def test_accept_with_annotation_id_attaches_the_existing_note(with_suggestions) -> None:
    base, server = with_suggestions
    n = _fail(server, note="cited cw-returns unread")
    assert _decide(base, id="sg1", decision="accepted", reason="same failure",
                   annotation_id=n["id"])[0] == 200
    assert [a["id"] for a in server.read_annotations()] == [n["id"]], "no new note"
    assert _attached(server) == {"annotation_id": n["id"], "mode": M, "added": True}
    assert _modes()[M]["source_annotation_ids"] == [n["id"]]


@pytest.mark.parametrize("case", ["no_failure", "other_trace", "other_turn", "other_mode"])
def test_attaching_an_unfit_note_is_409_and_writes_nothing(with_suggestions, case) -> None:
    base, server = with_suggestions
    from analysis.helpers import _state
    kw = {"no_failure": {"no_failure": True}, "other_trace": {"trace": "b"},
          "other_turn": {"turn": 2}, "other_mode": {}}[case]
    n = _fail(server, **kw)
    if case == "other_mode":
        _state.write_json(_state.state_path("patterns.json"),
                          {"modes": [_mode(M), _mode("other", source_annotation_ids=[n["id"]],
                                                     example_trace_ids=["a"])]})
    before = len(server.history())
    assert _decide(base, id="sg1", decision="accepted", reason="r", annotation_id=n["id"])[0] == 409
    assert len(server.history()) == before
    assert {x["id"]: x for x in server.read_suggestions()}["sg1"]["status"] == "pending"


def test_a_repeated_accept_of_an_attached_suggestion_is_409(with_suggestions) -> None:
    base, server = with_suggestions
    n = _fail(server)
    assert _decide(base, id="sg1", decision="accepted", reason="r1", annotation_id=n["id"])[0] == 200
    assert _decide(base, id="sg1", decision="accepted", reason="r2", annotation_id=n["id"])[0] == 409
    # "covered" -> "add as reason" needs a reject first
    assert _decide(base, id="sg1", decision="accepted", reason="r3")[0] == 409


@pytest.mark.parametrize("change", ["none", "edited", "retracted"])
def test_reject_after_an_attached_accept_detaches_from_the_mode_only(with_suggestions, change) -> None:
    base, server = with_suggestions
    n = _fail(server, note="v1")
    assert _decide(base, id="sg1", decision="accepted", reason="r", annotation_id=n["id"])[0] == 200
    if change == "edited":
        assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "no_failure": False,
                            "note": "v2", "supersedes": n["id"]})[0] == 200
    if change == "retracted":
        assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply",
                            "supersedes": n["id"], "retracted": True})[0] == 200
    live_before = server.read_annotations()
    assert _decide(base, id="sg1", decision="rejected", reason="not this mode")[0] == 200
    assert server.read_annotations() == live_before, "the note is never retracted"
    assert _modes()[M]["source_annotation_ids"] == []


def test_reject_leaves_a_note_the_accept_did_not_add(with_suggestions) -> None:
    base, server = with_suggestions
    n = _fail(server)
    assert server.assign_to_mode(n["id"], M)[0] == 200
    assert _decide(base, id="sg1", decision="accepted", reason="r", annotation_id=n["id"])[0] == 200
    assert _attached(server)["added"] is False
    assert _decide(base, id="sg1", decision="rejected", reason="no")[0] == 200
    assert _modes()[M]["source_annotation_ids"] == [n["id"]]


def test_switching_add_to_covered_retracts_the_produced_note(with_suggestions) -> None:
    """support-0123: one failure held twice, as the reviewer's note and as the
    accepted-suggestion note."""
    base, server = with_suggestions
    n = _fail(server, note="own words")
    assert _decide(base, id="sg1", decision="accepted", reason="also")[0] == 200
    produced = next(a for a in server.read_annotations() if a.get("suggestion_id") == "sg1")
    assert _decide(base, id="sg1", decision="accepted", reason="same failure",
                   annotation_id=n["id"])[0] == 200
    assert [a["id"] for a in server.read_annotations()] == [n["id"]]
    assert _modes()[M]["source_annotation_ids"] == [n["id"]]
    assert produced["id"] not in _modes()[M]["source_annotation_ids"]


def test_accept_retracts_a_plain_no_failure_note_on_the_turn(with_suggestions) -> None:
    base, server = with_suggestions
    assert _post(base, {"trace_id": "a", "turn": 1, "step": "reply", "no_failure": True,
                        "note": "passes"})[0] == 200
    assert _decide(base, id="sg1", decision="accepted", reason="cited unopened")[0] == 200
    assert [a["no_failure"] for a in server.read_annotations()] == [False]


# --------------------------------------------------------------------------- #
# Part E: the labeling view (one present/absent label per trace and final mode)
# --------------------------------------------------------------------------- #

@pytest.fixture
def labeling(tmp_path, monkeypatch):
    """Two sampled traces, one final mode and one dropped mode, Langfuse offline."""
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path))
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(var, raising=False)
    from analysis.helpers import _state
    from analysis.review_app import server
    _write_manifest({"b1": ["t1", "t2"]})
    monkeypatch.setattr(server, "VIEWS", [{"session_id": "s1", "scenario_id": "support-0001",
                                           "trace_ids": ["t1", "t2"]}])
    note = server.append_annotation({"trace_id": "t1", "turn": 1, "step": "reply", "note": "cited cw-x unopened"})[-1]
    _state.write_json(_state.state_path("patterns.json"), {"modes": [
        {"name": "m1", "status": "candidate", "definition": "d", "boundary": "b",
         "example_trace_ids": ["t1"], "source_annotation_ids": [note["id"]],
         "close_negative_trace_ids": ["t2"], "scenarios": {"positive": [], "close_negative": []}},
        {"name": "gone", "status": "dropped", "example_trace_ids": [], "scenarios": {}}]})
    return server


def test_labels_view_lists_only_final_modes_and_prefills_from_the_taxonomy(labeling) -> None:
    status, body = labeling.labels_view("m1")
    assert status == 200 and body["modes"] == ["m1"], "a dropped mode is not labeled"
    rows = {r["trace_id"]: r for r in body["rows"]}
    assert rows["t1"]["proposal"]["label"] == 1 and "cw-x unopened" in rows["t1"]["proposal"]["evidence"]
    assert rows["t2"]["proposal"]["label"] == 0
    assert all(r["label"] is None for r in rows.values()), "a proposal is never a label"
    assert labeling.labels_view("gone")[0] == 404


def test_a_flip_appends_and_the_last_label_wins_for_the_shared_loader(labeling) -> None:
    from analysis.helpers import tools
    assert labeling.save_label("m1", "t1", 1, "reply cites cw-x; never opened")[0] == 200
    assert labeling.save_label("m1", "t1", 0, "on reread: it was opened")[0] == 200
    lines = labeling._labels_file("m1").read_text().splitlines()
    assert len(lines) == 2, "a correction appends; the first judgment stays on disk"
    assert labeling.live_labels("m1")["t1"]["label"] == 0
    assert [r["label"] for r in tools._load_labels("m1")] == [0], "same rule as the HW5 loader"


def test_saving_the_same_judgment_twice_writes_nothing(labeling) -> None:
    labeling.save_label("m1", "t2", 0, "trigger absent")
    status, body = labeling.save_label("m1", "t2", 0, "trigger absent")
    assert status == 200 and body.get("unchanged")
    assert len(labeling._labels_file("m1").read_text().splitlines()) == 1


def test_a_flip_with_the_old_evidence_is_refused(labeling) -> None:
    """Evidence argues the saved answer: a changed answer needs new evidence (2026-11-09, 0004 saved
    No failure with the evidence written for Failure)."""
    assert labeling.save_label("m1", "t1", 1, "reply cites cw-x; never opened")[0] == 200
    status, body = labeling.save_label("m1", "t1", 0, "reply cites cw-x; never opened")
    assert status == 400 and "new evidence" in body["error"]
    assert len(labeling._labels_file("m1").read_text().splitlines()) == 1, "a refused flip appends nothing"
    assert labeling.live_labels("m1")["t1"]["label"] == 1


def test_save_label_rejects_what_is_not_a_judgment(labeling) -> None:
    assert labeling.save_label("gone", "t1", 1)[0] == 404
    assert labeling.save_label("m1", "t1", True)[0] == 400, "a bool is not a 0/1 label"
    assert labeling.save_label("m1", "t1", 2)[0] == 400
    assert labeling.save_label("m1", "not-sampled", 1)[0] == 409
    assert not labeling._labels_file("m1").exists()


def test_without_langfuse_env_the_label_saves_and_sync_writes_nothing(labeling) -> None:
    status, body = labeling.save_label("m1", "t1", 1, "x")
    assert status == 200 and body["label"]["langfuse"] == "offline"
    status, body = labeling.sync_labels("m1")
    assert status == 200 and body["offline"] and body["not_synced"] == 1
    assert len(labeling._labels_file("m1").read_text().splitlines()) == 1, "offline sync appends nothing"


def test_an_unreachable_langfuse_never_costs_the_label(labeling, monkeypatch) -> None:
    """The local append comes first and records 'pending'; the push result rides on the response only."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")   # discard port: nothing listens
    status, body = labeling.save_label("m1", "t1", 1, "reply cites cw-x; never opened")
    assert status == 200 and body["label"]["langfuse"] != "pending", "the response reports the push"
    on_disk = [json.loads(x) for x in labeling._labels_file("m1").read_text().splitlines()]
    assert [(r["label"], r["langfuse"]) for r in on_disk] == [(1, "pending")], "written before the push"


def test_a_label_records_how_it_was_accepted(labeling) -> None:
    """A bulk accept of a rule proposal must stay distinguishable from a pair read in
    context: HW5 builds its Pass/Fail sets from these records."""
    status, body = labeling.save_label("m1", "t2", 0, "trigger absent: x", "bulk", "rule: trigger absent")
    assert status == 200
    rec = body["label"]
    assert (rec["accepted_via"], rec["proposal_source"], rec["scenario_id"]) == \
        ("bulk", "rule: trigger absent", None), "scenario comes from the manifest entry"
    assert labeling.save_label("m1", "t1", 1, "x", "autopilot")[0] == 400


def test_focus_queue_holds_only_pairs_that_need_reading(labeling) -> None:
    """Taxonomy rows (the reviewer's own judgments) and rule rows (trigger absent)
    go in bulk; the queue holds agent Present/UNSURE proposals and unproposed pairs."""
    from analysis.helpers import _state
    _write_manifest({"b1": ["t1", "t2", "t3", "t4"]})
    labeling.VIEWS[0]["trace_ids"] = ["t1", "t2", "t3", "t4"]
    _state.write_json(_state.state_path("label_proposals", "m1.json"), [
        {"trace_id": "t3", "label": 0, "source": "agent proposal", "evidence": "UNSURE: maybe"},
        {"trace_id": "t4", "label": 0, "source": "rule: trigger absent", "evidence": "trigger absent: x"}])
    q = labeling.label_queue()
    assert [i["trace_id"] for i in q["items"]] == ["t3"], "t1/t2 are taxonomy, t4 is a rule row"
    assert (q["todo"], q["unlabeled"], q["bulk"]) == (1, 4, 3)
    labeling.save_label("m1", "t3", 0, "read it: absent", "focus", "agent proposal")
    q = labeling.label_queue()
    assert q["todo"] == 0 and q["unlabeled"] == 3, "the labeled pair no longer counts as to-read"
    assert q["items"][0]["label"]["label"] == 0, "it stays listed with the reviewer's label, so they see what they decided"


def test_careful_pass_queues_every_agent_proposal_of_one_mode(labeling) -> None:
    """The mode filter adds the confident agent 'absent' rows and the mode's taxonomy rows (marked, so
    the list can show them as such and the reviewer can re-check them); never rule rows."""
    from analysis.helpers import _state
    _write_manifest({"b1": ["t1", "t2", "t3", "t4"]})
    labeling.VIEWS[0]["trace_ids"] = ["t1", "t2", "t3", "t4"]
    _state.write_json(_state.state_path("label_proposals", "m1.json"), [
        {"trace_id": "t3", "label": 0, "source": "agent proposal", "evidence": "cites cw-x, opened"},
        {"trace_id": "t4", "label": 0, "source": "rule: trigger absent", "evidence": "trigger absent: x"}])
    assert labeling.label_queue()["items"] == [], "a confident agent absent is not a hard pair"
    items = labeling.label_queue("m1")["items"]
    assert [(i["trace_id"], i["taxonomy"]) for i in items] == [("t1", True), ("t2", True), ("t3", False)], \
        "t1/t2 are taxonomy rows, t4 a rule row"
    assert labeling.label_queue("gone")["items"] == [], "a dropped mode queues nothing"


def test_queue_items_carry_their_sample_batch(labeling) -> None:
    """The focus list keeps a batch's labeled pairs listed while that batch has pairs to do."""
    _write_manifest({"b1": ["t1", "t3"], "b2": ["t4"]})   # t3, t4: no proposal, so both are queued
    labeling.VIEWS[0]["trace_ids"] = ["t1", "t3", "t4"]
    assert {i["trace_id"]: i["batch"] for i in labeling.label_queue("m1")["items"]} == \
        {"t1": "b1", "t3": "b1", "t4": "b2"}, "t1 is a taxonomy row of the careful pass"


def test_a_label_without_evidence_is_refused(labeling) -> None:
    """Every path (focus, table, bulk, API) must say why; whitespace is not a reason."""
    status, body = labeling.save_label("m1", "t1", 1, "   ")
    assert status == 400 and "evidence is required" in body["error"]
    assert not labeling._labels_file("m1").exists()


def test_an_empty_close_negative_note_still_proposes_evidence(labeling) -> None:
    """Bulk-accepting the reviewer's own close negatives must not fail on an empty note."""
    labeling.append_annotation({"trace_id": "t2", "turn": 1, "step": "reply", "no_failure": True, "note": ""})
    p = labeling.label_proposals("m1", labeling.read_annotations())["t2"]
    assert p["label"] == 0 and p["evidence"].strip()


def test_the_first_focus_save_is_the_blind_answer(labeling) -> None:
    """The server records the blind answer itself: the pair's first focus save. A later save carries no
    blind fields, a bulk accept is not blind, and nothing writes `switched` (it is derived on read)."""
    first = labeling.save_label("m1", "t1", 1, "my case", "focus", "agent proposal")[1]["label"]
    assert (first["label"], first["blind_label"], first["blind_evidence"]) == (1, 1, "my case")
    assert "switched" not in first
    later = labeling.save_label("m1", "t1", 0, "on reread: opened")[1]["label"]
    assert later["accepted_via"] == "focus", "focus is the default"
    assert "blind_label" not in later and "switched" not in later
    plain = labeling.save_label("m1", "t2", 0, "trigger absent", "bulk", "rule: trigger absent")[1]["label"]
    assert "blind_label" not in plain, "a bulk accept is not a blind judgment"


def test_save_label_takes_no_client_blind_or_note(labeling) -> None:
    """The client cannot state the blind answer or a switch; the Note box is gone."""
    import inspect
    params = inspect.signature(labeling.save_label).parameters
    assert "blind" not in params and "note" not in params


def test_a_second_opinion_source_still_needs_reading():
    """Only taxonomy and rule rows skip the reading queue; a newer agent pass (its own source tag)
    is a second opinion for the reviewer, not a bulk accept, even when it is a confident 'absent'."""
    from analysis.review_app import server

    assert server.needs_reading({"label": 0, "source": "agent proposal (sonnet, 2026-11-09 boundary)", "evidence": "fine"})
    assert not server.needs_reading({"label": 0, "source": "agent proposal", "evidence": "fine"})
    assert server.needs_reading({"label": 1, "source": "agent proposal", "evidence": "x"})
    assert not server.needs_reading({"label": 0, "source": "taxonomy: close negative", "evidence": "x"})
    assert not server.needs_reading({"label": 0, "source": "rule: trigger absent", "evidence": "x"})


def _queue_setup(labeling) -> None:
    """t3 carries an agent proposal (queued), t4 none; t1/t2 stay taxonomy rows."""
    from analysis.helpers import _state
    _write_manifest({"b1": ["t1", "t2", "t3", "t4"]})
    labeling.VIEWS[0]["trace_ids"] = ["t1", "t2", "t3", "t4"]
    _state.write_json(_state.state_path("label_proposals", "m1.json"), [
        {"trace_id": "t3", "label": 1, "source": "agent proposal", "evidence": "cites cw-x, never opened"}])


def test_label_history_chains_each_save_to_the_one_it_replaces(labeling) -> None:
    """Blind answer, the agent's answer at that moment, the switch and every later change stay on
    disk; each record names the one it replaces, computed by the server, never by the client."""
    _queue_setup(labeling)
    ai = {"label": 1, "evidence": "cites cw-x, never opened", "source": "agent proposal"}
    first = labeling.save_label("m1", "t3", 0, "it was opened", "focus", "agent proposal")[1]
    assert first["label"]["ai"] == ai and "note" not in first["label"]
    assert "supersedes" not in first["label"], "the first judgment replaces nothing"
    switch = labeling.save_label("m1", "t3", 1, "cites cw-x, never opened", "focus", "agent proposal")[1]["label"]
    assert switch["supersedes"] == "t3#0"
    later = labeling.save_label("m1", "t3", 0, "reread: opened at step 2", "focus", None)[1]["label"]
    assert later["supersedes"] == "t3#1" and "blind_label" not in later, "blind fields are not copied forward"
    hist = labeling.label_history("m1", "t3")
    assert [r["label_id"] for r in hist] == ["t3#0", "t3#1", "t3#2"]
    assert [r["switched"] for r in hist] == [False, True, False], "derived: differs from blind, equals the AI"
    assert all(r["ai"] == ai for r in hist), "the agent's answer at save time rides on each record"
    assert labeling.label_history("m1", "t1") == []


def test_a_label_without_a_proposal_records_ai_null(labeling) -> None:
    _queue_setup(labeling)
    rec = labeling.save_label("m1", "t4", 0, "nothing claimed", "focus")[1]["label"]
    assert rec["ai"] is None


def test_same_label_and_evidence_is_unchanged_and_new_evidence_appends(labeling) -> None:
    """Same verdict with a new justification is a new record; the same pair of values again is a no-op."""
    labeling.save_label("m1", "t1", 1, "x")
    status, body = labeling.save_label("m1", "t1", 1, "x, and the policy never says so")
    assert status == 200 and not body.get("unchanged") and body["label"]["supersedes"] == "t1#0"
    assert labeling.save_label("m1", "t1", 1, "  x, and the policy never says so ")[1].get("unchanged")
    assert len(labeling._labels_file("m1").read_text().splitlines()) == 2


def _raw(labeling, *records) -> None:
    from analysis.helpers import _state
    for r in records:
        _state.append_jsonl(labeling._labels_file("m1"), {"mode": "m1", "ts": "t", "accepted_via": "focus", **r})


def test_switched_is_derived_from_the_blind_answer_and_the_ai_at_save_time(labeling) -> None:
    """The 0049/0078 shape: the switch was made by typing, so #1 has no `switched` field; it still reads
    as a switch (differs from the blind answer, equals the AI's answer stored on the record). The current
    proposal file never decides it: here it now says Failure, which would make #1 not a switch."""
    from analysis.helpers import _state
    _queue_setup(labeling)   # current proposal for t3: Failure
    ai_then = {"label": 0, "evidence": "dates come from get_order", "source": "agent proposal"}
    _raw(labeling,
         {"trace_id": "t3", "label": 1, "evidence": "ship before July 1", "label_id": "t3#0",
          "blind_label": 1, "blind_evidence": "ship before July 1", "switched": False, "ai": ai_then},
         {"trace_id": "t3", "label": 0, "evidence": "a caution, not a claim", "label_id": "t3#1",
          "supersedes": "t3#0", "ai": ai_then})
    assert [r["switched"] for r in labeling.label_history("m1", "t3")] == [False, True]
    item = next(i for i in labeling.label_queue("m1")["items"] if i["trace_id"] == "t3")
    assert [r["switched"] for r in item["history"]] == [False, True], "the card's history reads the same"
    assert "switched" not in _state.read_jsonl(labeling._labels_file("m1"))[1], "derived, never written"


def test_switched_falls_back_to_the_first_records_ai(labeling) -> None:
    _raw(labeling,
         {"trace_id": "t3", "label": 1, "evidence": "e", "label_id": "t3#0", "blind_label": 1,
          "blind_evidence": "e", "ai": {"label": 0, "evidence": "a", "source": "agent proposal"}},
         {"trace_id": "t3", "label": 0, "evidence": "f", "label_id": "t3#1", "supersedes": "t3#0", "ai": None})
    assert [r["switched"] for r in labeling.label_history("m1", "t3")] == [False, True]


def test_a_legacy_record_without_ai_keeps_its_stored_switched(labeling) -> None:
    """support-0043 shape: one record, blind No failure, label Failure, switched true, no `ai`."""
    _raw(labeling, {"trace_id": "t3", "label": 1, "evidence": "e", "label_id": "t3#0", "blind_label": 0,
                    "blind_evidence": "b", "switched": True})
    assert labeling.label_history("m1", "t3")[0]["switched"] is True


def test_a_switch_is_only_the_record_that_changes_to_the_ai_answer(labeling) -> None:
    """Blind Failure, AI No failure, then two No failure saves: only the first of them switched;
    the second repeats the previous label with new evidence, so it is not a second switch."""
    from analysis.helpers import _state
    _queue_setup(labeling)
    _state.write_json(_state.state_path("label_proposals", "m1.json"), [
        {"trace_id": "t3", "label": 0, "source": "agent proposal", "evidence": "dates come from get_order"}])
    labeling.save_label("m1", "t3", 1, "ship before July 1", "focus", "agent proposal")
    labeling.save_label("m1", "t3", 0, "a caution, not a claim", "focus", "agent proposal")
    labeling.save_label("m1", "t3", 0, "a caution; get_order has the date", "focus", "agent proposal")
    assert [r["switched"] for r in labeling.label_history("m1", "t3")] == [False, True, False]


def test_a_legacy_record_keeps_its_switched_when_a_later_record_has_ai(labeling) -> None:
    """support-0043 followed by a new save: the legacy record resolves no AI label, so it keeps its
    stored `switched` even though the chain now has an `ai` elsewhere."""
    _raw(labeling,
         {"trace_id": "t3", "label": 1, "evidence": "e", "label_id": "t3#0", "blind_label": 0,
          "blind_evidence": "b", "switched": True},
         {"trace_id": "t3", "label": 0, "evidence": "f", "label_id": "t3#1", "supersedes": "t3#0",
          "ai": {"label": 0, "evidence": "a", "source": "agent proposal"}})
    assert [r["switched"] for r in labeling.label_history("m1", "t3")] == [True, False]


def test_a_taxonomy_proposal_is_not_stored_as_the_ai_answer(labeling) -> None:
    """t1's proposal is the reviewer's own HW4 judgment (taxonomy: positive), not an AI answer."""
    assert labeling.label_proposals("m1", labeling.read_annotations())["t1"]["source"].startswith("taxonomy:")
    rec = labeling.save_label("m1", "t1", 0, "on reread: it was opened", "focus")[1]["label"]
    assert rec["ai"] is None


def test_history_skips_resync_repeats_and_keeps_legacy_records(labeling) -> None:
    """A legacy record (no supersedes) opens the chain; a resync repeat is not a judgment, so the
    next save supersedes the judgment it repeats."""
    from analysis.helpers import _state
    path = labeling._labels_file("m1")
    _state.append_jsonl(path, {"trace_id": "t1", "mode": "m1", "label": 1, "evidence": "old", "ts": "t0",
                               "accepted_via": "focus", "label_id": "t1#0", "langfuse": "pending"})
    _state.append_jsonl(path, {"trace_id": "t1", "mode": "m1", "label": 1, "evidence": "old", "ts": "t1",
                               "accepted_via": "focus", "label_id": "t1#1", "resync_of": "t1#0",
                               "langfuse": "synced"})
    rec = labeling.save_label("m1", "t1", 0, "new reading")[1]["label"]
    assert rec["supersedes"] == "t1#0"
    assert [r["label_id"] for r in labeling.label_history("m1", "t1")] == ["t1#0", "t1#2"]


@pytest.mark.parametrize("bad, message", [
    ([("t1#0", "t1", None), ("t1#1", "t1", "t1#0"), ("t1#2", "t1", "t1#0")], "superseded more than once"),
    ([("t1#0", "t1", None), ("t2#0", "t2", "t1#0")], "another trace"),
])
def test_label_history_refuses_a_malformed_chain(labeling, bad, message) -> None:
    """A hand-edited labels file: two replacements of one record, or a link across traces."""
    from analysis.helpers import _state
    path = labeling._labels_file("m1")
    for label_id, trace, sup in bad:
        _state.append_jsonl(path, {"trace_id": trace, "mode": "m1", "label": 1, "evidence": "e", "ts": "t",
                                   "label_id": label_id, **({"supersedes": sup} if sup else {})})
    with pytest.raises(ValueError, match=message):
        labeling.label_history("m1", "t1")


def test_queue_items_carry_a_compact_history(labeling) -> None:
    _queue_setup(labeling)
    _raw(labeling, {"trace_id": "t4", "label": 0, "evidence": "old", "label_id": "t4#0", "note": "legacy note"})
    labeling.save_label("m1", "t3", 0, "it was opened", "focus", "agent proposal")
    labeling.save_label("m1", "t3", 1, "cites cw-x, never opened", "focus", "agent proposal")
    items = {i["trace_id"]: i for i in labeling.label_queue("m1")["items"]}
    h = items["t3"]["history"]
    assert [(r["label"], r.get("blind_label"), r.get("switched")) for r in h] == [(0, 0, False), (1, None, True)]
    assert set(h[0]) == {"label", "evidence", "ts", "accepted_via", "ai", "blind_label", "blind_evidence", "switched"}
    assert h[1]["ai"]["label"] == 1
    assert items["t4"]["history"][0]["note"] == "legacy note", "a legacy note still shows"


def test_queue_reports_class_balance_by_label_and_conversation(labeling) -> None:
    """HW5 needs 30 Failure and 30 No failure from independent conversations: a conversation is Fail when
    any labeled turn is Failure, Pass when every sampled turn is labeled No failure, else partial."""
    _write_manifest({"b1": ["t1", "t2", "t3", "t4", "t5"]})
    labeling.VIEWS[:] = [{"session_id": "s1", "scenario_id": "support-0001", "trace_ids": ["t1", "t2"]},
                         {"session_id": "s2", "scenario_id": "support-0002", "trace_ids": ["t3", "t4"]},
                         {"session_id": "s3", "scenario_id": "support-0003", "trace_ids": ["t5"]}]
    for t, label in (("t1", 1), ("t2", 0), ("t3", 0), ("t5", 0)):
        labeling.save_label("m1", t, label, "why")
    q = labeling.label_queue("m1")
    assert q["balance"] == {"m1": {"labels": {"fail": 1, "pass": 3},
                                   "conversations": {"fail": 1, "pass": 1, "partial": 1}}}
    assert q["unsynced"] == 4, "offline labels are not verified in Langfuse"
    assert set(labeling.label_queue()["balance"]) == {"m1"}, "all modes: one entry per final mode"


# --------------------------------------------------------------------------- #
# HW5 Part C judge view (hw5.md:203): verdict and critique beside the label
# --------------------------------------------------------------------------- #

def test_judge_view_puts_each_dev_verdict_beside_the_label() -> None:
    from analysis.helpers import judge_alignment
    from analysis.review_app import server

    status, body = server.judge_view("unsupported_claim-v4", "dev")
    assert status == 200
    metrics = judge_alignment("unsupported_claim-v4", "dev")  # v4 is frozen: no iteration row is logged
    assert len(body["rows"]) == metrics["n"]
    assert sorted(r["trace_id"] for r in body["rows"] if r["agree"] is False) == sorted(metrics["disagreements"])
    assert all(r["critique"] and r["label"] in ("Pass", "Fail") for r in body["rows"])


def test_judge_view_hides_test_verdicts_of_an_unfrozen_judge() -> None:
    from analysis.review_app import server

    status, body = server.judge_view("unsupported_claim-v3", "test")
    assert status == 403 and "not frozen" in body["error"]


def test_judge_view_rejects_an_unknown_judge_or_split() -> None:
    from analysis.review_app import server

    assert server.judge_view("unsupported_claim-v99", "dev")[0] == 404
    assert server.judge_view("unsupported_claim-v4", "holdout")[0] == 400


def test_judge_list_names_model_and_status() -> None:
    from analysis.review_app import server

    judges = {j["judge_id"]: j for j in server.judge_list()}
    v5 = judges["unsupported_claim-v5"]
    assert (v5["model"], v5["status"]) == ("gpt-5.5", "frozen")
    assert judges["unsupported_claim-v3"]["input"].startswith("pre-fix input")


def test_run_judge_refuses_the_test_split_before_the_freeze() -> None:
    from analysis.helpers import guards, run_judge

    def never(prompt, ids):
        raise AssertionError("no model call may happen for an unfrozen judge on test")

    with pytest.raises(guards.GuardViolation):
        run_judge("unsupported_claim-v3", split="test", classify=never)


@pytest.mark.parametrize("judge_id,split", [("unsupported_claim-v4", "dev"), ("unsupported_claim-v4", "test"),
                                            ("unsupported_claim-v5", "test")])
def test_judge_view_metrics_equal_judge_alignment(judge_id: str, split: str) -> None:
    """The tab's confusion counts, TPR/TNR and Wilson intervals are the handout's (hw5.md:156-172)."""
    from analysis.helpers import judge_alignment
    from analysis.review_app import server

    shown = server.judge_view(judge_id, split)[1]["metrics"]
    official = judge_alignment(judge_id, split)  # frozen judges: no iteration row is logged
    for k in ("tp", "fn", "tn", "fp", "tpr", "tnr", "tpr_interval", "tnr_interval", "agreement"):
        assert shown[k] == official[k], k
    assert shown["human_pass"] + shown["human_fail"] == official["n"]
    view = server.judge_view(judge_id, split)[1]
    assert view["labels_changed_since_export"] == 0 and shown["no_label"] == 0
    assert all(isinstance(r["turn"], int) and r["turn"] >= 1 for r in view["rows"])  # the labeled turn the judge read up to


def test_judge_metrics_tell_a_missing_label_from_a_missing_verdict() -> None:
    from analysis.review_app import server

    rows = [{"label": "Pass", "verdict": "Pass"}, {"label": "Fail", "verdict": "Pass"},
            {"label": None, "verdict": "Fail"}, {"label": "Fail", "verdict": None}]
    m = server._judge_metrics(rows)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 0, 0, 1)
    assert (m["no_label"], m["no_verdict"], m["unjudged"]) == (1, 1, 2)
    assert m["tnr"] == 0.0 and m["tpr"] == 1.0
