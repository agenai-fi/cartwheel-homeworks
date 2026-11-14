"""Backend for the HW4 review interface.

Retained from `analysis/server.py`: the design -- a stdlib `ThreadingHTTPServer`
with no dependencies, serving a single-file page plus a small JSON API backed by
plain files under `analysis/state/`, so the whole review log is inspectable with
`cat`. Persistence goes through the repo's own `analysis/helpers/_state.py`
(`state_path`, `read_jsonl`, `append_jsonl`), which honours
`CARTWHEEL_ANALYSIS_STATE` and fails loudly on a line it cannot parse. An
earlier version imported the reference's `_read_json`/`_write_json` without
calling them and reimplemented the jsonl helpers instead.

Changed, from reviewing the reference against real traces:

* The reference POSTs the whole annotation array and overwrites the file, so one
  interrupted save loses every note. Here `POST /api/annotations` appends one
  record to `annotations.jsonl`. Append-only, crash-safe, and it diffs cleanly.
* Editing and retracting also append. A record carrying `supersedes` kills the
  record it names; one carrying `retracted` kills its target and itself. Reads
  collapse the log to the live set, `history()` returns the raw log. hw4.md line
  106 requires the original open code to stay in the saved history, so nothing is
  ever rewritten or removed -- the same convention
  `analysis/helpers/tools.py::_load_labels` uses for labels via `superseded_by`.
* The reference serves one trace per sample. This serves one CONVERSATION,
  grouped on `cartwheel.session_id`, with every turn's `trace_id` intact,
  because reading is per session and Part E scores per trace.
* A conversation view carries its own ground truth and policy provenance, so a
  reply claim can be checked without leaving the page.

* Suggestions (agent depth-search hits) are decided, not deleted. The
  reference drops a dismissed suggestion; hw4.md Part D requires the saved state
  to record each acceptance or rejection and to hold at least one rejection. A
  rejection keeps its reason (the boundary that excluded it); an acceptance
  appends a human annotation with `source: accepted_suggestion`, so the mode's
  evidence stays human-confirmed.

* The taxonomy view (`GET /api/taxonomy`) shows each mode with its supporting
  notes and the failure notes in no mode; `POST /api/patterns/assign` adds one
  to a mode, the only write path for patterns.json.

Terms, used the same way in this module and the page:

* pool -- all 250 conversations in the trace export. `/api/index` lists every one.
* sample -- the traces listed in `analysis/state/sample_manifest.json`.
* batch -- a named draw into the sample. Membership is read from the manifest on
  every request, so a batch drawn while the server runs appears without a restart.
* served set -- what the page's list shows after its filters.

Notes are accepted on any pool trace; `/api/progress` counts the sample only.

Not here yet: the structured-labeling view, and the Langfuse score
write. The reference's
`_sync_annotation_scores` is the intended reuse for that pass; importing it now
would be dead code, since v1 produces no binary labels to sync.

    uv run python analysis/review_app/server.py --port 8030
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from analysis.helpers import _state  # noqa: E402
from analysis.review_app import loader  # noqa: E402

UI_DIR = HERE / "ui"
# jsonl is the append-only write log; annotations.json is the shape the handout's
# commit list and `analysis.helpers.failure_report` expect, derived from it.
LOG_NAME = "annotations.jsonl"
JSON_NAME = "annotations.json"
SUGGESTIONS_NAME = "suggestions.json"
PATTERNS_NAME = "patterns.json"
DECISIONS = ("accepted", "rejected")
# One writer at a time for every state write. The server is threaded: without it
# two supersedes of one id can both pass the live-set check (and every later read
# then refuses the log), and a decision can race an annotation POST. depth_search
# is a separate process -- run it with the server stopped.
WRITE_LOCK = threading.RLock()

# Built once at startup: the whole pool, 250 conversations, each with database facts, is a few
# seconds of sqlite and does not change while the server runs.
VIEWS: list[dict] = []
BY_KEY: dict[str, dict] = {}


def derive_graded_json() -> int:
    """Re-write annotations.json from the log.

    It is otherwise written only inside `append_annotation`, so a hand edit to the
    jsonl leaves the graded file -- the one `failure_report` and
    `tests/test_hw_holes.py` read -- silently a version behind. Called at startup.
    """
    live = read_annotations()
    _state.write_json(_state.state_path(JSON_NAME), {"annotations": live})
    return len(live)


def build_cache(pool: list[dict] | None = None) -> None:
    global VIEWS, BY_KEY
    pool = pool if pool is not None else loader.load_pool()
    VIEWS = loader.all_views(pool)
    BY_KEY = {}
    for v in VIEWS:
        BY_KEY[v["session_id"]] = v
        BY_KEY[v["scenario_id"]] = v


def log_path() -> Path:
    return _state.state_path(LOG_NAME)


def _with_id(record: dict) -> dict:
    """Give a record a stable id, synthesising one for records written before ids.

    The first four annotations were written without an id. Deriving it from
    (trace_id, step, ts) is deterministic, so those records keep the same id on
    every read and can be superseded like any other.
    """
    if record.get("id"):
        return record
    seed = json.dumps(record, sort_keys=True)
    return {**record, "id": "legacy-" + hashlib.sha1(seed.encode()).hexdigest()[:12]}


def history() -> list[dict]:
    """The raw append-only log, every record ever written, ids filled in.

    `_state.read_jsonl` lets a parse error propagate naming the line, rather than
    skipping it. For a 100-trace review log that cannot be regenerated, silently
    dropping a truncated line is the wrong failure.
    """
    path = log_path()
    records = [_with_id(r) for r in (_state.read_jsonl(path) if path.exists() else [])]
    _check_wellformed(records, path)
    return records


def _check_wellformed(records: list[dict], path: Path) -> None:
    """Refuse a log whose supersedes graph cannot be collapsed honestly.

    `collapse` treats `supersedes` as a set of dead targets, which cannot tell one
    edit from two and has no notion of reachability. So a malformed graph silently
    deletes live notes instead of raising -- the opposite of this module's policy
    for a malformed line. Every case below is refused by the POST path; these
    checks catch a hand-edited file.
    """
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(
            f"duplicate annotation id(s) in {path}: {dupes}. Two records sharing an "
            "id cannot be superseded independently; the log was edited by hand.")

    edges = {r["id"]: r["supersedes"] for r in records if r.get("supersedes")}
    for rid, target in edges.items():
        if rid == target:
            raise ValueError(f"{path}: record {rid} supersedes itself")

    targets = [t for t in edges.values()]
    if len(targets) != len(set(targets)):
        dupes = sorted({t for t in targets if targets.count(t) > 1})
        raise ValueError(
            f"{path}: {dupes} superseded more than once. Two replacements for one "
            "record would both stay live, so one edit would read as two notes.")

    for start in edges:
        seen, node = {start}, edges[start]
        while node in edges:
            if node in seen:
                raise ValueError(
                    f"{path}: supersedes cycle through {sorted(seen)}. A cycle marks "
                    "every record in it dead, silently deleting live notes.")
            seen.add(node)
            node = edges[node]

    for r in records:
        if r.get("retracted") and not r.get("supersedes"):
            raise ValueError(
                f"{path}: record {r['id']} is retracted but names no target")


def superseded_counts() -> dict[str, int]:
    """(trace_id, step) -> how many dead records it accumulated.

    hw4.md line 106 requires the original open code to stay inspectable. Without
    surfacing this, the history is only reachable by reading the file by hand.
    """
    records = history()
    live = {r["id"] for r in collapse(records)}
    out: dict[str, int] = {}
    for r in records:
        if r["id"] in live:
            continue
        key = f"{r.get('trace_id')}|{r.get('step')}"
        out[key] = out.get(key, 0) + 1
    return out


def collapse(records: list[dict]) -> list[dict]:
    """The live set: drop anything superseded, and drop tombstones themselves.

    An edit appends `{**new, supersedes: <old id>}` -- the old record dies, the new
    one lives. A retraction appends `{supersedes: <id>, retracted: true}` -- the
    target dies and so does the tombstone, leaving that step with no live note.
    Chains resolve naturally: A superseded by B superseded by C leaves only C.
    """
    dead = {r["supersedes"] for r in records if r.get("supersedes")}
    return [r for r in records if r["id"] not in dead and not r.get("retracted")]


def withdrawn_notes(records: list[dict]) -> list[dict]:
    """Notes withdrawn by a retraction, with the retraction's reason.

    One row per tombstone whose chain ends there: its target is not live and the
    tombstone itself was not superseded. An edited note is not withdrawn -- its
    replacement is live. The page shows these struck through, so a reclassified
    failure stays visible instead of being edited into a contradictory
    "no failure" (support-0007 turn 1)."""
    by_id = {r["id"]: r for r in records}
    live = {r["id"] for r in collapse(records)}
    superseded = {r["supersedes"] for r in records if r.get("supersedes")}
    rows = []
    for t in records:
        target = by_id.get(t.get("supersedes")) if t.get("retracted") else None
        if target is None or target["id"] in live or t["id"] in superseded:
            continue
        rows.append({k: target.get(k) for k in ("trace_id", "turn", "step", "scenario_id")}
                    | {"annotation_id": target["id"], "note": target.get("note"),
                       "no_failure": target.get("no_failure"),
                       "reason": t.get("reason") or "", "ts": t.get("ts")})
    return rows


def read_annotations() -> list[dict]:
    """The live annotations -- what the page shows and what gets graded."""
    return collapse(history())


def append_annotation(record: dict) -> list[dict]:
    with WRITE_LOCK:
        return _append_unlocked(record)


def _append_unlocked(record: dict) -> list[dict]:
    """Append to the log, then re-derive annotations.json from the whole log.

    Returns the live set after the append. `author` is stamped here rather than
    trusted from the client, because `tests/test_hw_holes.py` selects human
    origin on `author == "human"`; a record without it would not count.
    """
    # The id is minted here, never taken from the client: a duplicate id would
    # make `collapse` kill both records sharing it.
    record = {
        **record,
        "id": uuid.uuid4().hex[:12],
        "author": record.get("author") or "human",
    }
    _state.append_jsonl(log_path(), record)
    # One read, reused for the derived file and the response count. Re-deriving
    # the whole json per POST is O(n^2) -- measured at 1.55s over a 600-annotation
    # review, which is acceptable; reading the log twice per POST was not.
    everything = read_annotations()
    _state.write_json(_state.state_path(JSON_NAME), {"annotations": everything})
    return everything


def read_suggestions() -> list[dict]:
    data = _state.read_json(_state.state_path(SUGGESTIONS_NAME), default=[])
    if not isinstance(data, list):
        raise ValueError(f"{SUGGESTIONS_NAME} must hold a list of suggestions")
    return data


def _attachable(ident: str, target: dict, modes: list[dict]) -> tuple[dict | None, str]:
    """The reviewer's note an accept may attach ("covered by my reason"): a live
    failure note on the suggestion's trace and turn, in no other mode -- one
    failure, one mode. Returns (note, "") or (None, why not)."""
    note = next((a for a in read_annotations() if a.get("id") == ident), None)
    if note is None or note.get("no_failure"):
        return None, f"{ident} is not a live failure note"
    if note.get("trace_id") != target.get("trace_id") or note.get("turn") != target.get("turn", 1):
        return None, f"{ident} is not on this suggestion's trace and turn"
    other = [m["name"] for m in modes if m.get("name") != target.get("mode")
             and ident in m.get("source_annotation_ids", [])]
    if other:
        return None, f"your reason is in mode {other[0]}; move it at axial coding first"
    return note, ""


def _accept_reason(target: dict, annotation_id: str | None) -> str:
    """The reason an accept records when the reviewer wrote none. "Add as
    reason" takes the AI's own why -- the note stays marked as an accepted
    suggestion, so it never reads as the reviewer's words; "covered by my
    reason" names the note that already says it."""
    if annotation_id is not None:
        return f"covered by reason {annotation_id}"
    return (target.get("reason") or target.get("evidence") or "").strip()


def _edit_chain(ident: str, records: dict[str, dict]) -> set[str]:
    """`ident` and every record that edited it, live or dead: a mode holds
    whichever id of the chain was current when the mode last followed it."""
    succ = {r["supersedes"]: i for i, r in records.items() if r.get("supersedes")}
    chain = {ident}
    while ident in succ:
        ident = succ[ident]
        chain.add(ident)
    return chain


def decide_suggestion(sid: str, decision: str, reason: str,
                      verdict: dict | None = None,
                      annotation_id: str | None = None) -> tuple[int, dict]:
    """Record the human's decision on one suggestion. Returns (http status, body).

    Decisions append to the suggestion's `decisions` list and the latest wins, the
    same append-only convention as the annotation log, so a reviewer can change a
    call after a definition revision without erasing the first one. Accepting is
    idempotent (an existing live note for this suggestion is reused); moving an
    accepted suggestion to rejected retracts the note it created.

    `verdict` ({"no_failure": bool}) lets one reviewer action on a REJECT also
    record the turn's open-coding verdict, with the reason as its note, in the
    same locked step -- the merged verdict box. It is a plain human verdict, not
    tied to the suggestion, so a later decision never retracts a Failure verdict.
    A No-failure verdict is the exception: a later accept contradicts it on the
    same turn, so the accept withdraws it (support-0177 held both until retracted
    by hand). On an accept the verdict is implied (a failure note is written
    anyway), so it is ignored. Any accept now retracts every live no-failure
    note on the turn, in the same write, however it was written.

    `annotation_id` on an accept ("covered by my reason") attaches that existing
    note to the mode instead of writing one, and records `attached`
    {annotation_id, mode, added}. It also retracts a note an earlier accept of
    this suggestion produced (the support-0123 duplicate). A reject after it
    detaches the note from that mode only, and only if the accept added it; the
    note itself is never retracted.
    """
    reason = (reason or "").strip()
    if verdict is not None and not (isinstance(verdict, dict)
                                    and isinstance(verdict.get("no_failure"), bool)):
        return 400, {"error": 'verdict must be {"no_failure": true|false}'}
    if decision not in DECISIONS:
        return 400, {"error": f"decision must be one of {DECISIONS}"}
    if not reason and decision == "rejected":
        # A rejection's reason is the boundary that excluded the trace. An accept
        # may come without one: see _accept_reason.
        return 400, {"error": "write a note first: it is your reason for this decision"}
    if annotation_id is not None and decision != "accepted":
        return 400, {"error": "annotation_id applies to an accept only"}
    with WRITE_LOCK:
        items = read_suggestions()
        target = next((x for x in items if x.get("id") == sid), None)
        if target is None:
            return 404, {"error": f"unknown suggestion: {sid}"}
        reason = reason or _accept_reason(target, annotation_id)
        if not reason:
            return 400, {"error": "write a note first: it is your reason for this decision"}
        prev = target.get("decision") or {}
        attached = prev.get("attached") if prev.get("status") == "accepted" else None
        if decision == "accepted" and attached:
            return 409, {"error": f"{sid} is already accepted as covered by note "
                                  f"{attached['annotation_id']}; reject it first to change the answer"}
        live = [a for a in read_annotations() if a.get("suggestion_id") == sid]
        same = target.get("status", "pending") == decision
        # Same status is a no-op unless it changes something: a new reason, or an
        # acceptance whose note was retracted by hand and should be restored.
        if same and annotation_id is None and prev.get("reason") == reason \
                and not (decision == "accepted" and not live):
            return 409, {"error": f"{sid} is already {decision} for that reason"}
        patterns = read_patterns()
        mode = _find_mode(patterns, target.get("mode", ""))
        if decision == "accepted" and mode is None:
            # Checked before any write: an accepted note outside its mode is a half state.
            return 404, {"error": f"unknown mode: {target.get('mode')}; add it to patterns.json first"}
        note = None
        if annotation_id is not None:
            note, why = _attachable(str(annotation_id), target, patterns.get("modes", []))
            if note is None:
                return 409, {"error": why}
        mode_update: list[dict] = []
        touched: list[dict] = []
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        record = {"status": decision, "reason": reason, "decided_at": now}
        if decision == "accepted":
            # A failure on the turn ends any "no failure observed" on it.
            for a in read_annotations():
                if a.get("trace_id") == target["trace_id"] and a.get("no_failure") is True:
                    _append_unlocked({"trace_id": a["trace_id"], "turn": a.get("turn"), "step": a.get("step"),
                                      "supersedes": a["id"], "retracted": True, "ts": now})
                    record["retracted_verdict_id"] = a["id"]
        if note is not None:
            # "Covered by my reason": the note an earlier accept produced is the
            # same failure written twice; it goes, with its mode entries.
            records = _records_by_id()
            for a in live:
                for m in patterns.get("modes", []):
                    if a["id"] in m.get("source_annotation_ids", []):
                        _remove_note(m, a["id"], records)
                _append_unlocked({"trace_id": a["trace_id"], "turn": a.get("turn"), "step": a.get("step"),
                                  "supersedes": a["id"], "retracted": True, "ts": now})
                record["retracted_annotation_id"] = a["id"]
            added = note["id"] not in mode.get("source_annotation_ids", [])
            if added:
                _add_note(mode, note)
            _write_patterns(patterns)
            record["attached"] = {"annotation_id": note["id"], "mode": mode["name"], "added": added}
            if added:
                mode_update = [{"mode": mode["name"], "action": "added", "count": mode["count"]}]
        elif decision == "accepted":
            if live:
                record["annotation_id"] = live[0]["id"]
            else:
                written = _append_unlocked({
                    "trace_id": target["trace_id"], "turn": target.get("turn", 1),
                    "step": target.get("step", "reply"), "session_id": target.get("session_id"),
                    "scenario_id": target.get("scenario_id"), "no_failure": False,
                    "note": f"[accepted suggestion, mode {target['mode']}] {reason}",
                    "source": "accepted_suggestion", "suggestion_id": sid, "ts": now})
                record["annotation_id"] = next(r["id"] for r in written if r.get("suggestion_id") == sid)
            if record["annotation_id"] not in mode.setdefault("source_annotation_ids", []):
                _add_note(mode, next(a for a in read_annotations() if a["id"] == record["annotation_id"]))
                _write_patterns(patterns)
                mode_update = [{"mode": mode["name"], "action": "added", "count": mode["count"]}]
        elif attached:
            # Undo only what the accept did: detach the note from this mode,
            # under whichever id of its edit chain the mode holds.
            m = _find_mode(patterns, attached["mode"])
            if attached.get("added") and m is not None:
                records = _records_by_id()
                for i in _edit_chain(attached["annotation_id"], records) & set(m.get("source_annotation_ids", [])):
                    _remove_note(m, i, records)
                    touched = [m]
        else:
            # Every note this suggestion ever produced (edits keep suggestion_id)
            # is retracted now, so it is evidence in no mode -- not only its own.
            records = _records_by_id()
            mine = {i for i, r in records.items() if r.get("suggestion_id") == sid} \
                | {(target.get("decision") or {}).get("annotation_id")}
            for m in patterns.get("modes", []):
                drop = mine & set(m.get("source_annotation_ids", []))
                for i in drop:
                    _remove_note(m, i, records)
                if drop:
                    touched.append(m)
            for a in live:
                _append_unlocked({"trace_id": a["trace_id"], "turn": a.get("turn"), "step": a.get("step"),
                                  "supersedes": a["id"], "retracted": True, "ts": now})
                record["retracted_annotation_id"] = a["id"]
            if verdict is not None:
                written = _append_unlocked({
                    "trace_id": target["trace_id"], "turn": target.get("turn", 1),
                    "step": target.get("step", "reply"), "session_id": target.get("session_id"),
                    "scenario_id": target.get("scenario_id"), "no_failure": verdict["no_failure"],
                    "note": reason, "ts": now})
                record["verdict_annotation_id"] = written[-1]["id"]
        target.setdefault("decisions", []).append(record)
        target["status"] = decision
        target["decision"] = record
        _state.write_json(_state.state_path(SUGGESTIONS_NAME), items)
        if decision == "rejected" and touched:
            _write_patterns(patterns)
            mode_update = [{"mode": m["name"], "action": "removed", "count": m["count"]} for m in touched]
    body = {"ok": True, "suggestion": target}
    if mode_update:
        body["mode_update"] = mode_update
    return 200, body


def read_patterns() -> dict:
    return _state.read_json(_state.state_path(PATTERNS_NAME), default={"modes": []})


# A mode is attached to a NOTE (annotation id), counted per trace and per
# conversation. `source_annotation_ids` is the evidence; `example_trace_ids`,
# `scenarios.positive` and `count` are derived from it on every write.

def _records_by_id() -> dict[str, dict]:
    """Every record ever written, live or dead, by id -- a dead note still knows
    its trace and scenario, which is what gone-note cleanup needs."""
    return {r["id"]: r for r in history()}


def _find_mode(data: dict, name: str) -> dict | None:
    return next((m for m in data.get("modes", []) if m.get("name") == name), None)


def _scenario_of(trace_id: str, annotations: list[dict], records: dict[str, dict]) -> str | None:
    """The trace's scenario: from a live note, else any logged record, else the
    served views. None when nothing names it (a hand-written example)."""
    for pool in (reversed(annotations), reversed(list(records.values()))):
        for a in pool:
            if a.get("trace_id") == trace_id and a.get("scenario_id"):
                return a["scenario_id"]
    return next((v["scenario_id"] for v in VIEWS if trace_id in v.get("trace_ids", [])), None)


def _recompute(mode: dict, annotations: list[dict], records: dict[str, dict]) -> None:
    """scenarios.positive = ordered unique scenarios of example_trace_ids; count =
    its length. A trace whose scenario resolves nowhere keeps the existing entry
    at its position (the next old entry no resolved trace claims)."""
    scen = mode.setdefault("scenarios", {})
    traces = mode.setdefault("example_trace_ids", [])
    resolved = {t: _scenario_of(t, annotations, records) for t in traces}
    spare = [s for s in scen.get("positive", []) if s not in set(resolved.values())]
    positive: list[str] = []
    for t in traces:
        s = resolved[t] or (spare.pop(0) if spare else None)
        if s and s not in positive:
            positive.append(s)
    scen["positive"] = positive
    scen.setdefault("close_negative", [])
    mode["count"] = len(positive)


def _write_patterns(data: dict, annotations: list[dict] | None = None,
                    records: dict[str, dict] | None = None) -> None:
    """The only patterns writer: derived fields are recomputed for every mode."""
    annotations = read_annotations() if annotations is None else annotations
    records = _records_by_id() if records is None else records
    for m in data.get("modes", []):
        _recompute(m, annotations, records)
    _state.write_json(_state.state_path(PATTERNS_NAME), data)


def _add_note(mode: dict, note: dict) -> None:
    """Attach a live failure note. Close negatives are hand-curated and never
    touched here; a trace in both lists is reported by taxonomy() as a conflict."""
    mode.setdefault("source_annotation_ids", []).append(note["id"])
    trace_id = note.get("trace_id")
    if trace_id not in mode.setdefault("example_trace_ids", []):
        mode["example_trace_ids"].append(trace_id)


def _remove_note(mode: dict, note_id: str, records: dict[str, dict]) -> None:
    """Detach one note; its trace leaves only if no other source note of this mode is on it."""
    ids = mode["source_annotation_ids"] = [i for i in mode.get("source_annotation_ids", [])
                                           if i != note_id]
    trace_id = (records.get(note_id) or {}).get("trace_id")
    if trace_id and not any((records.get(i) or {}).get("trace_id") == trace_id for i in ids):
        mode["example_trace_ids"] = [t for t in mode.get("example_trace_ids", []) if t != trace_id]


def unassigned_failures(annotations: list[dict], modes: list[dict]) -> list[dict]:
    """Live failure NOTES in no mode's source_annotation_ids. A trace already in
    mode X with a second, different failure note still appears; `in_modes` names
    the modes whose examples already hold its trace."""
    assigned = {i for m in modes for i in m.get("source_annotation_ids", [])}
    return [{"trace_id": a.get("trace_id"), "turn": a.get("turn"), "step": a.get("step"),
             "scenario_id": a.get("scenario_id"), "note": a.get("note"), "annotation_id": a.get("id"),
             "in_modes": [m["name"] for m in modes if a.get("trace_id") in m.get("example_trace_ids", [])]}
            for a in annotations if not a.get("no_failure") and a.get("id") not in assigned]


def gone_examples(annotations: list[dict], modes: list[dict]) -> list[dict]:
    """Mode evidence that needs a decision, checked per note:

    - kind `note_gone`: a source id that is not a live failure note (retracted,
      edited to no-failure, or superseded). `live_note` is the trace's latest
      live note, if any, so the reviewer sees why it went.
    - kind `no_note`: an example trace with no source id on it."""
    records = _records_by_id()
    live = {a.get("id"): a for a in annotations}

    def turn_of(t: str | None) -> Any:
        on = [a for a in annotations if a.get("trace_id") == t] or \
             [r for r in records.values() if r.get("trace_id") == t and r.get("turn") is not None]
        return on[-1].get("turn") if t and on else None

    rows = []
    for m in modes:
        ids = m.get("source_annotation_ids", [])
        for i in ids:
            if i in live and not live[i].get("no_failure"):
                continue
            t = (records.get(i) or {}).get("trace_id")
            on = [a for a in annotations if t and a.get("trace_id") == t]
            last = on[-1] if on else None
            rows.append({"mode": m.get("name"), "kind": "note_gone", "annotation_id": i, "trace_id": t,
                         "scenario_id": _scenario_of(t, annotations, records) if t else None,
                         "turn": (records.get(i) or {}).get("turn", turn_of(t)),
                         "live_note": None if last is None else {
                             k: last.get(k) for k in ("id", "turn", "step", "note", "no_failure")}})
        sourced = {(records.get(i) or {}).get("trace_id") for i in ids}
        for t in m.get("example_trace_ids", []):
            if t not in sourced:
                rows.append({"mode": m.get("name"), "kind": "no_note", "trace_id": t,
                             "scenario_id": _scenario_of(t, annotations, records), "turn": turn_of(t)})
    return rows


def taxonomy(annotations: list[dict]) -> dict:
    """Modes with their supporting notes resolved, the failure notes in no mode,
    and the examples whose failure note is gone.

    A positive shows one row per live source note on its trace; a trace with no
    live source note shows its latest live failure note (`annotation_id` None).
    A close negative shows its latest live no-failure note; `note` is None when
    the trace has none."""
    latest: dict[tuple, dict] = {}
    by_id = {a.get("id"): a for a in annotations}
    for a in annotations:
        latest[(a.get("trace_id"), bool(a.get("no_failure")))] = a
    modes = read_patterns().get("modes", [])
    records = _records_by_id()

    def row(t: str, a: dict, source: bool) -> dict:
        return {"trace_id": t, "scenario_id": a.get("scenario_id"), "turn": a.get("turn"),
                "note": a.get("note"), "annotation_id": a.get("id") if source else None}

    def examples(m: dict) -> list[dict]:
        sources = [by_id[i] for i in m.get("source_annotation_ids", []) if i in by_id]
        out = []
        for t in m.get("example_trace_ids", []):
            mine = [a for a in sources if a.get("trace_id") == t]
            rs = [row(t, a, True) for a in mine] or [row(t, latest.get((t, False), {}), False)]
            for r in rs:
                src = by_id.get(r["annotation_id"]) or {}
                r["in_modes"] = [o["name"] for o in modes
                                 if o is not m and t in o.get("example_trace_ids", [])]
                r["from_suggestion"] = bool(src.get("suggestion_id"))
            out += rs
        return out

    def conflicts(m: dict) -> list[dict]:
        """Traces in both example_trace_ids and close_negative_trace_ids: the
        reviewer resolves these by hand."""
        neg = set(m.get("close_negative_trace_ids", []))
        out = []
        for t in m.get("example_trace_ids", []):
            if t in neg:
                a = latest.get((t, False)) or latest.get((t, True)) or {}
                out.append({"trace_id": t, "scenario_id": a.get("scenario_id") or _scenario_of(
                    t, annotations, records), "turn": a.get("turn")})
        return out

    def closest(m: dict) -> dict | None:
        a = next((by_id[i] for i in m.get("source_annotation_ids", []) if i in by_id), None)
        return None if a is None else {"scenario_id": a.get("scenario_id"), "note": a.get("note")}

    return {"modes": [{**m, "examples": examples(m),
                       "negatives": [row(t, latest.get((t, True), {}), False)
                                     for t in m.get("close_negative_trace_ids", [])],
                       "trace_count": len(m.get("example_trace_ids", [])),
                       "conversation_count": len((m.get("scenarios") or {}).get("positive", [])),
                       "conflicts": conflicts(m),
                       "closest_example": closest(m)} for m in modes],
            "unassigned": unassigned_failures(annotations, modes),
            "gone": gone_examples(annotations, modes)}


def assign_to_mode(ident: str, mode_name: str) -> tuple[int, dict]:
    """Attach one live failure note to a mode. Returns (http status, body).

    `ident` is an annotation id. For back-compat it may be a trace id: the
    trace's latest live failure note is used, and a trace already in the mode is
    a 409 as before (which note to add would be a guess)."""
    with WRITE_LOCK:
        data = read_patterns()
        mode = _find_mode(data, mode_name)
        if mode is None:
            return 404, {"error": f"unknown mode: {mode_name}"}
        live = read_annotations()
        note = next((a for a in live if a.get("id") == ident), None)
        if note is None:
            fails = [a for a in live if a.get("trace_id") == ident and not a.get("no_failure")]
            if not fails:
                return 409, {"error": f"{ident} is not a live failure note or a trace with one"}
            if ident in mode.get("example_trace_ids", []):
                return 409, {"error": f"{ident} is already in {mode_name}"}
            note = fails[-1]
        if note.get("no_failure"):
            return 409, {"error": f"{ident} is a no-failure note; only a failure joins a mode"}
        if note["id"] in mode.get("source_annotation_ids", []):
            return 409, {"error": f"note {note['id']} is already in {mode_name}"}
        _add_note(mode, note)
        _write_patterns(data, live)
    return 200, {"ok": True, "mode": mode}


def remove_from_mode(ident: str, mode_name: str) -> tuple[int, dict]:
    """Detach a note (annotation id) or, for gone-note cleanup, a whole trace.

    By trace id: the trace leaves, with every source id on it or whose note
    appears nowhere in the log (so its trace cannot be known)."""
    with WRITE_LOCK:
        data = read_patterns()
        mode = _find_mode(data, mode_name)
        if mode is None:
            return 404, {"error": f"unknown mode: {mode_name}"}
        records = _records_by_id()
        if ident in mode.get("source_annotation_ids", []):
            _remove_note(mode, ident, records)
        elif ident in mode.get("example_trace_ids", []):
            mode["example_trace_ids"] = [t for t in mode["example_trace_ids"] if t != ident]
            mode["source_annotation_ids"] = [
                i for i in mode.get("source_annotation_ids", [])
                if i in records and records[i].get("trace_id") != ident]
        else:
            return 409, {"error": f"{ident} is not in {mode_name}"}
        _write_patterns(data, None, records)
    return 200, {"ok": True, "mode": mode}


def _follow_edit(old_id: str, live: list[dict]) -> None:
    """An edit that keeps a note a failure keeps it in its modes under its new id.
    Without this the edited note reads as unassigned and its mode holds a dead id.
    An edit to no-failure is left alone: that trace becomes a gone example."""
    new = next((r for r in live if r.get("supersedes") == old_id), None)
    data = read_patterns()
    hit = [m for m in data.get("modes", []) if old_id in m.get("source_annotation_ids", [])]
    if new is None or not hit:
        return
    for m in hit:
        m["source_annotation_ids"] = [new["id"] if i == old_id else i for i in m["source_annotation_ids"]]
    _write_patterns(data, live)


MANIFEST_NAME = "sample_manifest.json"


def _read_manifest() -> tuple[dict[str, dict], str | None]:
    """(traces map, None), or ({}, reason) when the sample cannot be read.

    Never raises: the manifest is read on every index request, and a missing
    file, a parse error or a reshaped file (`analysis.helpers.tools.select_traces`
    writes `picks`, not `traces`) must reach the page as a warning, not a 500."""
    path = _state.state_path(MANIFEST_NAME)
    try:
        manifest = _state.read_json(path)
    except (OSError, ValueError) as e:      # JSONDecodeError is a ValueError
        return {}, f"{path} is not readable JSON ({e})"
    if manifest is None:
        return {}, f"no sample manifest at {path}"
    traces = manifest.get("traces") if isinstance(manifest, dict) else None
    if not isinstance(traces, dict):
        return {}, f'{path} has no "traces" map, so no trace is in the sample'
    if not all(isinstance(m, dict) for m in traces.values()):
        return {}, f'{path} has a "traces" entry that is not an object'
    return traces, None


def sample_batches() -> dict[str, str]:
    """trace_id -> batch name, read from the manifest now, not at startup. An
    entry without a batch is left out, so it cannot mask its conversation's
    batched trace."""
    traces, _ = _read_manifest()
    return {t: m["batch"] for t, m in traces.items() if m.get("batch")}


def manifest_status() -> dict:
    """Whether the sample can be read; when not, the sample reads empty and the
    page warns instead of showing 0 of 0. Also each batch's hw4.md Part B batch
    (1-4) and depth-search mode, which the batch filter groups and labels by."""
    _, reason = _read_manifest()
    batches = []
    if reason is None:
        records = _state.read_json(_state.state_path(MANIFEST_NAME)).get("batches")
        batches = [{"batch": b.get("batch"), "part_b_batch": b.get("part_b_batch"), "mode": b.get("mode")}
                   for b in records if isinstance(b, dict)] if isinstance(records, list) else []
    return {"manifest_ok": reason is None, "reason": reason, "batches": batches}


# --------------------------------------------------------------------------- #
# Part E: one present/absent label per (trace, final mode)
# --------------------------------------------------------------------------- #

def final_modes() -> list[dict]:
    """The modes Part E labels against: every mode not dropped by status."""
    return [m for m in read_patterns().get("modes", []) if m.get("status") != "dropped"]


def _labels_file(mode: str) -> Path:
    return _state.state_path("labels", f"{mode}.jsonl")


def live_labels(mode: str) -> dict[str, dict]:
    """trace_id -> live label record. The file is append-only (`_state.append_jsonl`
    is its only write path): a flip appends a new line, and the last line per trace
    wins, the same rule `analysis.helpers.tools._load_labels` applies."""
    live: dict[str, dict] = {}
    for r in _state.read_jsonl(_labels_file(mode)):
        if not r.get("superseded_by"):
            live[r["trace_id"]] = r
    return live


def _label_chains(mode: str) -> dict[str, list[dict]]:
    """trace_id -> that pair's judgments, oldest first, after checking their `supersedes` links.

    A resync repeat (`resync_of`) is a Langfuse status change, not a judgment, so it is left
    out. Legacy records carry no `supersedes`; they still count, in file order. Like
    `_check_wellformed` for notes, a link no save could have written (two replacements of
    one record, or a link to another trace's or a missing record) raises instead of
    reading as a history it is not."""
    path = _labels_file(mode)
    rows = [r for r in _state.read_jsonl(path) if not r.get("resync_of")]
    trace_of = {r["label_id"]: r["trace_id"] for r in rows if r.get("label_id")}
    replaced: set[str] = set()
    for r in rows:
        target = r.get("supersedes")
        if not target:
            continue
        if target in replaced:
            raise ValueError(f"{path}: {target} superseded more than once. A label history is one "
                             "chain; two replacements of one record mean the file was edited by hand.")
        replaced.add(target)
        if trace_of.get(target) != r["trace_id"]:
            raise ValueError(f"{path}: {r.get('label_id')} supersedes {target}, which is not a record of "
                             f"trace {r['trace_id']} (it names another trace or no record).")
    chains: dict[str, list[dict]] = {}
    for r in rows:
        chains.setdefault(r["trace_id"], []).append(r)
    return chains


def _with_switched(chain: list[dict]) -> list[dict]:
    """The chain with `switched` derived per record: its label differs from the previous record's
    (for the first record: from the pair's blind answer, the first `blind_label`), differs from the
    blind answer, and equals the AI's answer at save time (this record's `ai`, else the first
    record's). Derived because a client flag misses a switch made by typing (support-0049, 0078).
    Never read from the current proposal file: it is re-run and would rewrite history. A record
    that resolves no AI label, or a chain with no blind answer, keeps its stored `switched`
    (legacy, e.g. support-0043)."""
    blind = next((r["blind_label"] for r in chain if r.get("blind_label") in (0, 1)), None)
    if blind is None:
        return chain
    first_ai = chain[0].get("ai")
    out, prev = [], blind
    for r in chain:
        ai = r.get("ai") or first_ai
        if ai:
            r = {**r, "switched": r["label"] != prev and r["label"] != blind and r["label"] == ai["label"]}
        out.append(r)
        prev = r["label"]
    return out


def label_history(mode: str, trace_id: str) -> list[dict]:
    """One pair's label records, oldest first: the blind answer, a switch, every later change."""
    return _with_switched(_label_chains(mode).get(trace_id, []))


# `note`: no longer written (the Note box is gone); kept so legacy records still show theirs.
HISTORY_KEYS = ("label", "evidence", "ts", "accepted_via", "ai", "blind_label", "blind_evidence",
                "switched", "note")


def _compact(record: dict) -> dict:
    """What the focus card shows of one history record; `ai` is null on a legacy record."""
    return {"ai": None, **{k: record[k] for k in HISTORY_KEYS if k in record}}


def label_proposals(mode: str, annotations: list[dict]) -> dict[str, dict]:
    """trace_id -> {label, source, evidence}: what the screen pre-fills, never a label.

    The taxonomy's own positives and close negatives are human judgments and win;
    `label_proposals/<mode>.json` (a list of {trace_id, label, evidence}) carries
    proposals for the rest, e.g. from a code check or an LLM pass."""
    out: dict[str, dict] = {}
    path = _state.state_path("label_proposals", f"{mode}.json")
    for p in _state.read_json(path, default=[]) if path.exists() else []:
        out[p["trace_id"]] = {"label": int(p["label"]), "source": p.get("source", "agent proposal"),
                              "evidence": p.get("evidence", "")}
    m = _find_mode(read_patterns(), mode) or {}
    by_id = {a["id"]: a for a in annotations}
    for t in m.get("close_negative_trace_ids", []):
        note = next((a.get("note") or "" for a in annotations if a.get("trace_id") == t and a.get("no_failure")), "")
        out[t] = {"label": 0, "source": "taxonomy: close negative",
                  "evidence": note or "your close negative for this mode (no note recorded)"}
    for t in m.get("example_trace_ids", []):
        notes = [by_id[i]["note"] for i in m.get("source_annotation_ids", [])
                 if i in by_id and by_id[i].get("trace_id") == t]
        out[t] = {"label": 1, "source": "taxonomy: positive", "evidence": " | ".join(notes)}
    return out


def labels_view(mode: str) -> tuple[int, dict]:
    """Every sampled trace with its live label and proposal, for one final mode."""
    m = next((x for x in final_modes() if x["name"] == mode), None)
    if m is None:
        return 404, {"error": f"{mode!r} is not a final mode", "modes": [x["name"] for x in final_modes()]}
    if not VIEWS:
        build_cache()
    turn_of = {t: (v["scenario_id"], i) for v in VIEWS for i, t in enumerate(v["trace_ids"], 1)}
    batches = sample_batches()
    annotations = read_annotations()
    labels, props = live_labels(mode), label_proposals(mode, annotations)
    rows = []
    for t, batch in batches.items():
        scen, turn = turn_of.get(t, (None, None))
        rows.append({"trace_id": t, "scenario_id": scen, "turn": turn, "batch": batch,
                     "label": labels.get(t), "proposal": props.get(t)})
    rows.sort(key=lambda r: (r["scenario_id"] or "", r["turn"] or 0))
    done = [r for r in rows if r["label"]]
    return 200, {"mode": mode, "definition": m.get("definition"), "boundary": m.get("boundary"),
                 "modes": [x["name"] for x in final_modes()], "rows": rows,
                 "counts": {"traces": len(rows), "labeled": len(done),
                            "present": sum(r["label"]["label"] == 1 for r in done),
                            "synced": sum(r["label"].get("langfuse") == "synced" for r in done)}}


def needs_reading(proposal: dict | None) -> bool:
    """A pair the reviewer reads in context rather than bulk-accepts: no proposal,
    or an agent proposal that says Present or flags itself UNSURE. Taxonomy rows are
    the reviewer's own judgments and rule rows are trigger-absent; both go in bulk."""
    if proposal is None:
        return True
    source = proposal.get("source") or ""
    if source.startswith(("taxonomy:", "rule:")):
        return False
    if source != "agent proposal":   # a later pass with its own tag is a second opinion to read, never a bulk accept
        return True
    return proposal.get("label") == 1 or (proposal.get("evidence") or "").startswith("UNSURE")


def label_queue(mode: str | None = None) -> dict:
    """The focus queue, ordered by conversation and turn so one trace's modes come together.

    Without `mode`: every unlabeled pair that needs reading, across all final modes.
    With `mode` (the careful pass): every pair of that mode except the rule rows: every
    agent proposal, confident or not, and the mode's taxonomy rows (`taxonomy: true`), so
    the reviewer's own HW4 judgments can be re-checked here.

    `balance`: per mode in the queue, Failure / No failure counts by live label and by
    conversation (scenario_id): Fail when any labeled turn is Failure, Pass when every
    sampled turn is labeled No failure, partial otherwise (unlabeled conversations are not
    counted). HW5 needs 30 of each from independent conversations. `unsynced`: live labels
    of those modes whose Langfuse score is not verified."""
    if not VIEWS:
        build_cache()
    turn_of = {t: (v["scenario_id"], i) for v in VIEWS for i, t in enumerate(v["trace_ids"], 1)}
    annotations = read_annotations()
    batches = sample_batches()
    items, unlabeled, balance, unsynced = [], 0, {}, 0
    for m in final_modes():
        if mode and m["name"] != mode:
            continue
        labels, props = live_labels(m["name"]), label_proposals(m["name"], annotations)
        chains = _label_chains(m["name"])
        convs: dict[str, list[int | None]] = {}
        for t, batch in batches.items():
            p = props.get(t)
            source = (p or {}).get("source") or ""
            careful = bool(mode) and not source.startswith("rule:")
            queued = careful or needs_reading(p)
            if t not in labels:
                unlabeled += 1
            scen, turn = turn_of.get(t, (None, None))
            convs.setdefault(scen or t, []).append(labels[t]["label"] if t in labels else None)
            if queued:   # a labeled pair stays in the queue, marked done, so the reviewer sees what they decided
                items.append({"trace_id": t, "scenario_id": scen, "turn": turn, "batch": batch, "mode": m["name"],
                              "definition": m.get("definition"), "boundary": m.get("boundary"),
                              "proposal": p, "label": labels.get(t), "taxonomy": source.startswith("taxonomy:"),
                              "history": [_compact(r) for r in _with_switched(chains.get(t, []))]})
        sampled = [r for t, r in labels.items() if t in batches]
        touched = [ls for ls in convs.values() if any(x is not None for x in ls)]
        balance[m["name"]] = {
            "labels": {"fail": sum(r["label"] == 1 for r in sampled), "pass": sum(r["label"] == 0 for r in sampled)},
            "conversations": {"fail": sum(1 in ls for ls in touched),
                              "pass": sum(all(x == 0 for x in ls) for ls in touched),
                              "partial": sum(1 not in ls and None in ls for ls in touched)}}
        unsynced += sum(r.get("langfuse") != "synced" for r in labels.values())
    items.sort(key=lambda x: (x["scenario_id"] or "", x["turn"] or 0, x["mode"]))
    todo = sum(1 for i in items if not i["label"])
    return {"items": items, "todo": todo, "unlabeled": unlabeled, "bulk": unlabeled - todo,
            "balance": balance, "unsynced": unsynced}


def _push_score(record: dict) -> str:
    """Send one label to Langfuse as a score named after the mode. The score id is
    stable per (trace, mode), so a flipped label updates the score in place.

    Returns 'sent', 'offline' (no LANGFUSE_* env) or 'failed: <reason>'. 'sent' is not
    'synced': the SDK queues the score and logs, never raises, an ingestion error
    (langfuse `create_score`), so only a read-back (`verify_score`) earns 'synced'."""
    from analysis.helpers import langfuse_io
    if not langfuse_io.is_configured():
        return "offline"
    try:
        langfuse_io.write_label_score(trace_id=record["trace_id"], mode=record["mode"],
                                      label=record["label"], comment=record.get("evidence") or None,
                                      score_id=f"{record['trace_id']}-{record['mode']}")
        return "sent"
    except Exception as exc:  # noqa: BLE001 -- any Langfuse error: the local label stands, the screen offers a retry
        return f"failed: {exc}"[:200]


def verify_score(record: dict) -> bool:
    """True when Langfuse holds a score for (trace, mode) with this label's value."""
    from analysis.helpers import langfuse_io
    if not langfuse_io.is_configured():
        return False
    try:
        scores = langfuse_io.read_label_scores(record["trace_id"], mode=record["mode"])
    except Exception:  # noqa: BLE001 -- unreadable counts as unverified, never as synced
        return False
    return any(s.get("value") is not None and int(s["value"]) == record["label"] for s in scores)


ACCEPTED_VIA = ("focus", "table", "bulk")


def save_label(mode: str, trace_id: str, label: Any, evidence: Any = "",
               accepted_via: Any = "focus", proposal_source: Any = None) -> tuple[int, dict]:
    """Record the reviewer's present (1) / absent (0) judgment for one sampled
    trace and final mode, then send it to Langfuse. `accepted_via` and
    `proposal_source` keep a read-in-context judgment apart from a bulk accept.

    History, like the notes log: nothing is deleted. Each record names the record it
    replaces (`supersedes`, computed here under the lock, never taken from the client, so
    a pair's history stays one chain) and carries `ai`, the proposal as it stood at save
    time. The pair's first focus save is its blind answer: this function writes
    `blind_label` / `blind_evidence` from it, never from the client (the focus card shows
    no AI answer before that save). `switched` is not written; `label_history` derives it.

    The local append is the source of truth and comes first, under WRITE_LOCK; the
    Langfuse push follows outside the lock and is best-effort. The file records
    'offline' or 'pending' (Verify / retry reads 'pending' back); the response's
    `label.langfuse` carries the push result ('sent' or 'failed: <reason>')."""
    if mode not in {m["name"] for m in final_modes()}:
        return 404, {"error": f"{mode!r} is not a final mode"}
    if label not in (0, 1) or isinstance(label, bool):
        return 400, {"error": "label must be 1 (failure present) or 0 (absent)"}
    traces, _ = _read_manifest()
    if trace_id not in traces or not traces[trace_id].get("batch"):
        return 409, {"error": f"{trace_id} is not in the review sample"}
    if accepted_via not in ACCEPTED_VIA:
        return 400, {"error": f"accepted_via must be one of {ACCEPTED_VIA}"}
    from analysis.helpers import langfuse_io
    evidence = evidence.strip() if isinstance(evidence, str) else ""
    if not evidence:   # a label nobody can audit later is not a judgment; every path must say why
        return 400, {"error": "evidence is required: one line saying why (the reply span and the fact "
                              "it conflicts with, or why there is no failure)"}
    with WRITE_LOCK:
        path = _labels_file(mode)
        n = sum(r["trace_id"] == trace_id for r in _state.read_jsonl(path))
        prior = live_labels(mode).get(trace_id)
        if prior and prior["label"] == label and prior.get("evidence", "") == evidence:
            return 200, {"ok": True, "label": prior, "unchanged": True}
        if prior and prior["label"] != label and prior.get("evidence", "") == evidence:
            return 400, {"error": "a changed answer needs new evidence: the saved evidence argues "
                                  f"{'Failure' if prior['label'] == 1 else 'No failure'}"}
        ai = label_proposals(mode, read_annotations()).get(trace_id)
        if ai and ai["source"].startswith("taxonomy:"):   # the reviewer's own HW4 judgment, not an AI answer
            ai = None
        chain = _label_chains(mode).get(trace_id)
        record = {"trace_id": trace_id, "scenario_id": traces[trace_id].get("scenario_id"),
                  "mode": mode, "label": label, "evidence": evidence,
                  "source": "human", "accepted_via": accepted_via,
                  "proposal_source": proposal_source if isinstance(proposal_source, str) else None,
                  **({"blind_label": label, "blind_evidence": evidence}
                     if not chain and accepted_via == "focus" else {}),
                  "ai": {k: ai[k] for k in ("label", "evidence", "source")} if ai else None,
                  **({"supersedes": chain[-1]["label_id"]} if chain else {}),
                  "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                  "label_id": f"{trace_id}#{n}"}
        record["langfuse"] = "pending" if langfuse_io.is_configured() else "offline"
        _state.append_jsonl(path, record)
    if record["langfuse"] == "pending":
        pushed = _push_score(record)
        if pushed.startswith("failed"):
            print(f"label {record['label_id']} ({mode}) saved; Langfuse push {pushed}", file=sys.stderr, flush=True)
        record = {**record, "langfuse": pushed}
    return 200, {"ok": True, "label": record}


def sync_labels(mode: str) -> tuple[int, dict]:
    """Verify every live label of `mode` not yet 'synced' by reading its score back
    from Langfuse; a label whose score is missing or differs is sent again. Each
    status change appends the same judgment (`resync_of` names the record it
    repeats); the judgment itself never changes. Offline: nothing is written."""
    from analysis.helpers import langfuse_io
    if mode not in {m["name"] for m in final_modes()}:
        return 404, {"error": f"{mode!r} is not a final mode"}
    if not langfuse_io.is_configured():
        live = live_labels(mode).values()
        return 200, {"ok": True, "tried": 0, "not_synced": sum(r.get("langfuse") != "synced" for r in live),
                     "offline": True}
    tried = left = 0
    with WRITE_LOCK:
        path = _labels_file(mode)
        rows = _state.read_jsonl(path)
        for t, r in live_labels(mode).items():
            if r.get("langfuse") == "synced":
                continue
            status = "synced" if verify_score(r) else _push_score(r)
            again = {**r, "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(), "resync_of": r["label_id"],
                     "label_id": f"{t}#{sum(x['trace_id'] == t for x in rows)}", "langfuse": status}
            _state.append_jsonl(path, again)
            rows.append(again)
            tried += 1
            left += status != "synced"
    return 200, {"ok": True, "tried": tried, "not_synced": left}


# --------------------------------------------------------------------------- #
# HW5 Part C: judge verdicts beside the human labels (hw5.md:203)
# --------------------------------------------------------------------------- #

def judge_list() -> list[dict]:
    """Every registered judge of every final mode, oldest version first."""
    out = []
    for mode in final_modes():
        hist = _state.read_json(_state.state_path("judges", f"_history_{mode['name']}.json"),
                                default={"versions": []})
        for v in hist["versions"]:
            judge = _state.read_json(_state.state_path("judges", f"{v['judge_id']}.json"), default=None)
            if judge is not None:
                out.append({"judge_id": judge["judge_id"], "mode": judge["mode"], "model": judge["model"],
                            "status": judge.get("status", "draft"), "input": judge.get("input", "")})
    return out


def judge_view(judge_id: str, split: str) -> tuple[int, dict]:
    """One row per trace of `split`: the live human label, the judge's cached verdict and critique.

    Verdicts read the judge record's cache; nothing is run or logged. Test rows only for a frozen
    judge: the handout forbids seeing test predictions before the freeze. Labels and verdicts are
    "Pass"/"Fail" here (the helpers hold failure flags, 1 = failure present).
    """
    from analysis.helpers import guards, tools

    if split not in ("train", "dev", "test"):
        return 400, {"error": f"split must be train, dev or test, not {split!r}"}
    try:
        judge = tools._load_judge(judge_id)
    except FileNotFoundError as exc:
        return 404, {"error": str(exc)}
    if split == "test" and not guards.is_frozen(judge):
        return 403, {"error": f"{judge_id} is not frozen; test verdicts stay hidden until it is"}
    mode = judge["mode"]
    ids = _state.read_json(_state.state_path("splits.json"), default={}).get(mode, {}).get(split, [])
    labels = {r["trace_id"]: r for r in tools._load_labels(mode)}
    preds = tools._cached_preds(judge)
    critiques = judge.get("critiques", {}).get(judge["prompt_hash"], {})
    word = {0: "Pass", 1: "Fail"}
    rows = [{"trace_id": t,
             "scenario_id": labels.get(t, {}).get("scenario_id"),
             "label": word[int(labels[t]["label"])] if t in labels else None,
             "verdict": word[int(preds[t])] if t in preds else None,
             "critique": critiques.get(t, ""),
             "evidence": labels.get(t, {}).get("evidence", "")} for t in ids]
    for r in rows:
        r["agree"] = None if r["label"] is None or r["verdict"] is None else r["label"] == r["verdict"]
    # Metrics use the HW5 export (as judge_alignment and the report do); a flip saved in the Labels tab
    # reaches it only on re-export, so count the split's traces whose live label differs.
    live = live_labels(mode)
    # The judge read each conversation up to its labeled turn: the user turns in its prepared input.
    inputs = {r["trace_id"]: r["trace"] for r in _state.read_json(_state.state_path("hw5_trace_inputs.json"), default=[])}
    for r in rows:
        t = r["trace_id"]
        r["label_changed"] = t in live and t in labels and int(live[t]["label"]) != int(labels[t]["label"])
        r["turn"] = sum(m.get("role") == "user" for m in inputs.get(t, [])) or None
    changed = sum(r["label_changed"] for r in rows)
    return 200, {"judge_id": judge_id, "model": judge["model"], "status": judge.get("status", "draft"),
                 "split": split, "metrics": _judge_metrics(rows), "labels_changed_since_export": changed,
                 "rows": rows}


def _judge_metrics(rows: list[dict]) -> dict:
    """hw5.md "Confusion matrix" and "Compute TPR, TNR": Pass is the positive class.

    Same counts and 95% Wilson intervals as `judge_alignment`, without its side effect
    (a dev call on a draft judge appends an iteration row). Rows without a label or a
    verdict are left out and counted in `unjudged`, split into `no_label` (judge_alignment
    refuses such a split) and `no_verdict`.
    """
    from analysis.helpers import tools

    def n(label, verdict):
        return sum(r["label"] == label and r["verdict"] == verdict for r in rows)

    tp, fn, tn, fp = n("Pass", "Pass"), n("Pass", "Fail"), n("Fail", "Fail"), n("Fail", "Pass")
    judged = tp + fn + tn + fp
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "human_pass": tp + fn, "human_fail": tn + fp, "unjudged": len(rows) - judged,
            "no_label": sum(r["label"] is None for r in rows),
            "no_verdict": sum(r["label"] is not None and r["verdict"] is None for r in rows),
            "tpr": round(tp / (tp + fn), 4) if tp + fn else None,
            "tnr": round(tn / (tn + fp), 4) if tn + fp else None,
            "tpr_interval": tools._wilson_interval(tp, tp + fn),
            "tnr_interval": tools._wilson_interval(tn, tn + fp),
            "agreement": round((tp + tn) / judged, 4) if judged else None}


def index_rows(annotations: list[dict]) -> list[dict]:
    """The lightweight list the page navigates: the whole pool, each row with its
    batch (None outside the sample). Full views are fetched per item."""
    noted = {a.get("trace_id") for a in annotations}
    batches = sample_batches()
    rows = []
    for v in VIEWS:
        done = [t in noted for t in v["trace_ids"]]
        batch = next((batches[t] for t in v["trace_ids"] if t in batches), None)
        rows.append({
            "session_id": v["session_id"],
            "scenario_id": v["scenario_id"],
            "user_role": v["user_role"],
            "turns": v["totals"]["turns"],
            "tool_calls": v["totals"]["tool_calls"],
            "latency_s": v["totals"]["latency_s"],
            "trace_ids": v["trace_ids"],
            "batch": batch,
            "in_sample": batch is not None,
            # a 3-turn conversation is not reviewed because one turn has a note;
            # `started` drives the resume cursor, `reviewed` drives the count.
            "started": any(done),
            "reviewed": all(done),
            "turns_annotated": sum(done),
        })
    return rows


def contradictory_verdicts(annotations: list[dict]) -> list[dict]:
    """Live notes that disagree: one says failure, another says no failure.
    Run before Part E labeling -- the accept path now withdraws its own
    no-failure verdict, but other paths (a second verdict button, an edit) are
    not guarded. Returns one row per contradiction:
    {"trace_id", "turn", "step", "failure_ids", "no_failure_ids"}."""
    # Per step, not per turn: a failure on one step beside "no failure" on
    # another is a step-level reclassification (support-0007), not a conflict.
    groups: dict[tuple, list[dict]] = {}
    for a in annotations:
        groups.setdefault((a.get("trace_id"), a.get("turn"), a.get("step")), []).append(a)
    rows = []
    for (trace_id, turn, step), notes in groups.items():
        fail = [a.get("id") for a in notes if not a.get("no_failure")]
        ok = [a.get("id") for a in notes if a.get("no_failure")]
        if fail and ok:
            rows.append({"trace_id": trace_id, "turn": turn, "step": step,
                         "failure_ids": fail, "no_failure_ids": ok})
    return rows


def progress(annotations: list[dict]) -> dict:
    """Counts over the sample only. The log also holds notes on pool traces
    outside it, so counting it whole showed a fresh batch as part-done."""
    rows = [r for r in index_rows(annotations) if r["in_sample"]]
    served = {t for r in rows for t in r["trace_ids"]}
    gone = [g for g in gone_examples(annotations, read_patterns().get("modes", []))
            if g["trace_id"] in served]
    annotations = [a for a in annotations if a.get("trace_id") in served]
    traces = len(served)
    return {
        "conversations_total": len(rows),
        "conversations_reviewed": sum(1 for r in rows if r["reviewed"]),
        "conversations_started": sum(1 for r in rows if r["started"]),
        "conversations_incomplete": sum(1 for r in rows if r["started"] and not r["reviewed"]),
        "traces_total": traces,
        "traces_annotated": len({a["trace_id"] for a in annotations}),
        "annotations": len(annotations),
        "no_failure_observed": sum(1 for a in annotations if a.get("no_failure")),
        "contradictory_verdicts": contradictory_verdicts(annotations),
        "examples_with_note_gone": len(gone),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than default
        pass

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            self._json({"error": f"not found: {path.name}"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            self._file(UI_DIR / "index.html", "text/html; charset=utf-8")
            return

        # Every view-backed route needs the cache, not just whichever one the page
        # happens to call first. main() builds it before serving, so this only
        # matters to a direct API caller -- but the asymmetry was a trap.
        if path == "/api/index" or path == "/api/progress" or path.startswith("/api/session/"):
            if not VIEWS:
                build_cache()

        if path == "/api/index":
            self._json(index_rows(read_annotations()))
        elif path == "/api/manifest":
            self._json(manifest_status())
        elif path == "/api/annotations":
            self._json(read_annotations())
        elif path == "/api/withdrawn":
            self._json(withdrawn_notes(history()))
        elif path == "/api/history":
            self._json({"records": history(), "superseded_counts": superseded_counts()})
        elif path == "/api/progress":
            self._json(progress(read_annotations()))
        elif path == "/api/suggestions":
            self._json(read_suggestions())
        elif path == "/api/patterns":
            self._json(read_patterns())
        elif path == "/api/taxonomy":
            self._json(taxonomy(read_annotations()))
        elif path == "/api/labels/queue":
            from urllib.parse import parse_qs, urlsplit
            mode = (parse_qs(urlsplit(self.path).query).get("mode") or [""])[0]
            self._json(label_queue(mode or None))
        elif path == "/api/labels":
            from urllib.parse import parse_qs, urlsplit
            mode = (parse_qs(urlsplit(self.path).query).get("mode") or [""])[0]
            if not mode:
                self._json({"modes": [m["name"] for m in final_modes()]})
            else:
                status, body = labels_view(mode)
                self._json(body, status)
        elif path == "/api/judges":
            self._json({"judges": judge_list()})
        elif path.startswith("/api/judge/"):
            from urllib.parse import parse_qs, urlsplit
            split = (parse_qs(urlsplit(self.path).query).get("split") or ["dev"])[0]
            status, body = judge_view(path[len("/api/judge/"):], split)
            self._json(body, status)
        elif path.startswith("/api/session/"):
            key = path[len("/api/session/"):]
            view = BY_KEY.get(key)
            if view is None:
                self._json({"error": f"unknown session or scenario: {key}"}, 404)
            else:
                self._json(view)
        else:
            self._json({"error": f"unknown path: {path}"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/api/annotations", "/api/suggestions/decision", "/api/patterns/assign",
                        "/api/patterns/unassign", "/api/labels", "/api/labels/sync"):
            self._json({"error": f"cannot POST to {path}"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            record = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "expected a JSON body with a numeric Content-Length"}, 400)
            return
        if path in ("/api/patterns/assign", "/api/patterns/unassign"):
            if not isinstance(record, dict):
                self._json({"error": "expected a JSON object"}, 400)
                return
            ident = str(record.get("annotation_id") or record.get("trace_id") or "")
            fn = assign_to_mode if path.endswith("/assign") else remove_from_mode
            status, body = fn(ident, str(record.get("mode", "")))
            self._json(body, status)
            return
        if path in ("/api/labels", "/api/labels/sync"):
            if not isinstance(record, dict):
                self._json({"error": "expected a JSON object"}, 400)
                return
            if path == "/api/labels":
                status, body = save_label(str(record.get("mode", "")), str(record.get("trace_id", "")),
                                          record.get("label"), record.get("evidence", ""),
                                          record.get("accepted_via", "focus"), record.get("proposal_source"))
            else:
                status, body = sync_labels(str(record.get("mode", "")))
            self._json(body, status)
            return
        if path == "/api/suggestions/decision":
            if not isinstance(record, dict):
                self._json({"error": "expected a JSON object"}, 400)
                return
            status, body = decide_suggestion(str(record.get("id", "")), record.get("decision", ""),
                                             record.get("reason", ""), record.get("verdict"),
                                             record.get("annotation_id"))
            self._json(body, status)
            return
        if not isinstance(record, dict) or not record.get("trace_id"):
            self._json({"error": "an annotation needs a trace_id"}, 400)
            return
        if record.get("retracted") and not record.get("supersedes"):
            self._json({"error": "a retraction must name the id it supersedes"}, 400)
            return
        if record.get("retracted") and "reason" in record:
            if not isinstance(record["reason"], str):
                self._json({"error": "a retraction reason must be a string"}, 400)
                return
            record["reason"] = record["reason"].strip()
        with WRITE_LOCK:   # the live-set check and the append must be one step
            if record.get("supersedes"):
                # Against the LIVE set, not the whole log. Superseding a dead record
                # would let a stale tab double-supersede one id (leaving two live
                # notes), silently no-op a retraction, or resurrect a tombstone.
                live = {r["id"]: r for r in read_annotations()}
                if record["supersedes"] not in live:
                    self._json(
                        {"error": f"{record['supersedes']} is not a live annotation; "
                                  "it may already have been edited or retracted"}, 409)
                    return
                # An edit keeps its lineage: a note born from an accepted
                # suggestion stays linked to it, so un-accepting still finds it.
                target = live[record["supersedes"]]
                if target.get("suggestion_id") and record.get("no_failure") is True \
                        and not record.get("retracted"):
                    # Flipping it to no-failure would leave the suggestion saying
                    # "accepted" while its own note says the opposite.
                    self._json({"error": "this note came from an accepted suggestion; "
                                         "reject the suggestion instead"}, 409)
                    return
                for key in ("suggestion_id", "source"):
                    if target.get(key) is not None and record.get(key) is None:
                        record[key] = target[key]
            everything = _append_unlocked(record)
            if record.get("supersedes") and not record.get("retracted") and not record.get("no_failure"):
                _follow_edit(record["supersedes"], everything)
        self._json({"ok": True, "annotations": len(everything)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8030)
    # Accepted so existing launch commands keep working; the whole pool is served
    # and the page filters by batch, read from the manifest per request.
    ap.add_argument("--only", default="", help="ignored: the page lists the whole pool")
    ap.add_argument("--only-from", default="",
                    help="ignored: the page lists the whole pool; filter by batch there")
    args = ap.parse_args()
    if args.only or args.only_from:
        print("  --only/--only-from are ignored: the whole pool is served, "
              "filter by batch in the page", flush=True)

    print("building conversation views...", flush=True)
    pool = loader.load_pool()          # parsed once, not once per caller
    build_cache(pool)
    c = loader.census(pool)
    print(f"  {c['conversations']} conversations, {c['traces']} traces, "
          f"{c['excluded']} excluded")
    print(f"  annotations so far: {derive_graded_json()} in {log_path()}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"review app on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
