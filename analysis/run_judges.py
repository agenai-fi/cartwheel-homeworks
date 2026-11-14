"""HW5 Part B: judge inputs and the label split for one failure mode.

`prepare_inputs()` writes `analysis/state/hw5_trace_inputs.json`, the frozen judge
input: one record per eligible labeled conversation, `{"trace_id", "trace"}`, where
`trace` holds only what the agent saw and said: the system prompt rendered for the
session, then user and assistant text, tool calls and tool results (opened policy
passages arrive as `get_policy` results). It first re-exports the review app's
label log to the HW5 label file (`export_labels`, 1 = Pass). Labels decide
which turn is evaluated and which conversations are eligible, but no label,
evidence, scenario metadata or review note is written into a record.

Conversations come from the review app's loader (`analysis/review_app/loader.py`),
which groups traces by `cartwheel.session_id` and orders observations by
`startTime`; this module does not re-parse the export.

Independence rule (hw5.md Part B, "keep one record from each group of duplicate
runs or close scenario variants"): see `variant_key`. The kept member of a group is
the lowest scenario id, chosen without reading labels.

`split_data(mode)` is the handout helper call. It persists to
`analysis/state/splits.json`; run it once, after the grouping is approved.
"""

from __future__ import annotations

import collections
import itertools
import json
import re
import sqlite3
import statistics
from pathlib import Path

from analysis.review_app import loader

MODE = "unsupported_claim"
REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "analysis" / "state"
LABEL_LOG = STATE / "labels" / f"{MODE}.jsonl"
LABELS = STATE / "hw5_labels" / f"{MODE}.jsonl"
SCENARIOS = REPO / "scenarios" / "support_scenarios.jsonl"
INPUTS = STATE / "hw5_trace_inputs.json"
REPORT = REPO / "analysis" / "report" / "hw5-part-b-inputs.md"
# The database the HW3 run served from; only its users table is read (role and
# store per user, as server/app.py builds the session context). Explicit rather
# than CARTWHEEL_DB, which tests point at a freshly seeded world.
USERS_DB = REPO / "data" / "cartwheel.db"

PASS, FAIL = 1, 0  # HW5 convention


# --------------------------------------------------------------------------- #
# labels and scenarios
# --------------------------------------------------------------------------- #

def export_labels(mode: str = MODE, log: Path | None = None, out: Path | None = None) -> list[dict]:
    """Review-app label log -> HW5 label export, Pass = 1.

    The log `analysis/state/labels/<mode>.jsonl` is append-only and records
    1 = failure present; the latest record per trace_id (file order) is the live
    label. The export `analysis/state/hw5_labels/<mode>.jsonl` holds one row per
    trace with `label = 1 - log label` and `source_label_id` = the log record's
    `label_id`, sorted by (scenario_id, trace_id).
    """
    log = log or STATE / "labels" / f"{mode}.jsonl"
    out = out or STATE / "hw5_labels" / f"{mode}.jsonl"
    latest: dict[str, dict] = {}
    for line in log.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["trace_id"]] = row
    rows = []
    for row in sorted(latest.values(), key=lambda r: (r["scenario_id"], r["trace_id"])):
        if row.get("label") not in (0, 1):
            raise ValueError(f"log label must be 0 or 1: {row}")
        rows.append({
            "trace_id": row["trace_id"], "scenario_id": row["scenario_id"], "mode": row["mode"],
            "label": 1 - row["label"], "evidence": row["evidence"],
            "source_label_id": row["label_id"], "accepted_via": row["accepted_via"], "ts": row["ts"],
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def read_labels(path: Path = LABELS) -> dict[str, dict]:
    """trace_id -> live label record (superseded records dropped, last write wins)."""
    live: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("superseded_by"):
            continue
        if row.get("label") not in (PASS, FAIL):
            raise ValueError(f"label must be 0 or 1: {row}")
        live[row["trace_id"]] = row
    return live


def read_scenarios(path: Path = SCENARIOS) -> dict[str, dict]:
    return {s["id"]: s for s in map(json.loads, path.read_text().splitlines()) if s}


# --------------------------------------------------------------------------- #
# independence
# --------------------------------------------------------------------------- #

def variant_key(scenario: dict) -> tuple:
    """Equal keys = close variants: the same seeded order (`tuple.order_id`) or
    the same product (`tuple.product_id`), AND the same `tuple.intent`.

    Justified by construction, not by text similarity:

    * same record and same intent: the scenarios read the same tool data to
      answer the same question; they differ in who asks and how (role, user,
      user style), sometimes also in turn count, difficulty or applicable policy.
      The report lists the differing tuple fields of every group.
    * same record, different intent: the five challenge scenarios per
      data-quality case share one record but ask different questions on purpose
      (scenarios/skill/SKILL.md, "Five challenge scenarios target each of the six
      data quality cases"), so they are independent tests of the agent.
    * no order or product: the scenario stands alone. This includes the
      policy-page coverage scenarios: their `data_quality_case_id` is null and
      each was generated independently, so a shared policy page does not make
      variants.
    """
    t = scenario["tuple"]
    if t.get("order_id") is not None:
        return ("order", t["order_id"], t["intent"])
    if t.get("product_id") is not None:
        return ("product", t["product_id"], t["intent"])
    return ("scenario", scenario["id"])


def group_variants(scenario_ids: list[str], scenarios: dict[str, dict]) -> dict[tuple, list[str]]:
    """variant key -> sorted scenario ids. Label-blind: takes no labels."""
    groups: dict[tuple, list[str]] = collections.defaultdict(list)
    for sid in sorted(set(scenario_ids)):
        groups[variant_key(scenarios[sid])].append(sid)
    return dict(groups)


def kept_scenarios(groups: dict[tuple, list[str]]) -> set[str]:
    """One per group, the lowest scenario id."""
    return {min(ids) for ids in groups.values()}


# --------------------------------------------------------------------------- #
# turn choice and messages
# --------------------------------------------------------------------------- #

def choose_turn(turn_labels: dict[int, int]) -> int:
    """The first Failing labeled turn, else the last labeled turn."""
    fails = sorted(n for n, label in turn_labels.items() if label == FAIL)
    return fails[0] if fails else max(turn_labels)


def turn_messages(turn: dict) -> list[dict]:
    """One loader turn as judge messages, in observation order."""
    out = [{"role": "user", "text": turn["user"]}]
    for step in turn["steps"]:
        if step["kind"] == "agent":
            out.append({"role": "assistant", "text": step["text"]})
        else:
            out.append({"role": "tool_call", "name": step["tool"], "arguments": step["in"]})
            out.append({"role": "tool_result", "name": step["tool"], "content": step["out"]})
    out.append({"role": "assistant", "text": turn["reply"]})
    return out


def _conversations(export: Path) -> dict[str, list[dict]]:
    """scenario_id -> its session's loader turns, one session per scenario enforced.

    Each turn also carries its trace's `attrs` (the recorded `cartwheel.*`
    session attributes), read only to render the system prompt.
    """
    pool = loader.load_pool(export)
    loader.check_one_session_per_scenario(pool)
    return {
        loader.scenario_of(group[0]): [{**loader.build_turn(i + 1, t), "attrs": loader.attrs(t)}
                                       for i, t in enumerate(group)]
        for group in loader.sessions(pool).values()
    }


# --------------------------------------------------------------------------- #
# system prompt
# --------------------------------------------------------------------------- #

def read_users(db: Path = USERS_DB) -> dict[int, tuple[str, int | None]]:
    """user id -> (role, store_id) from the users table."""
    conn = sqlite3.connect(db)
    try:
        return {uid: (role, store) for uid, role, store
                in conn.execute("SELECT id, role, store_id FROM users")}
    finally:
        conn.close()


def system_message(attrs: dict, users: dict[int, tuple[str, int | None]]) -> dict:
    """The agent's system prompt as the session saw it.

    The template is chosen by the session's recorded `cartwheel.prompt_version`;
    only the current `SYSTEM_PROMPT_TEMPLATE` is known, so any other version
    raises rather than rendering a prompt the agent never received. Role and
    user come from the recorded attributes, the store from the users table,
    as server/app.py fills the session context.
    """
    from agent.agent import SYSTEM_PROMPT_TEMPLATE, prompt_version, render_system_prompt
    from agent.auth import AuthContext

    templates = {prompt_version(SYSTEM_PROMPT_TEMPLATE): SYSTEM_PROMPT_TEMPLATE}
    version = attrs.get("cartwheel.prompt_version")
    if version not in templates:
        raise ValueError(f"recorded prompt_version {version!r} has no known template "
                         f"(known: {sorted(templates)})")
    user_id, role = int(attrs["cartwheel.user_id"]), attrs["cartwheel.user_role"]
    db_role, store_id = users[user_id]
    if db_role != role:
        raise ValueError(f"user {user_id}: recorded role {role!r}, users table {db_role!r}")
    ctx = AuthContext(user_id=user_id, role=role, store_id=store_id)
    return {"role": "system", "text": render_system_prompt(ctx, templates[version])}


def session_system_message(turns: list[dict], users: dict) -> dict:
    """One system message per session; every turn must record the same context."""
    keys = ("cartwheel.prompt_version", "cartwheel.user_id", "cartwheel.user_role")
    contexts = {tuple(t["attrs"].get(k) for k in keys) for t in turns}
    if len(contexts) != 1:
        raise ValueError(f"session {turns[0]['trace_id']} changes context across turns: {contexts}")
    return system_message(turns[0]["attrs"], users)


# --------------------------------------------------------------------------- #
# prepare_inputs
# --------------------------------------------------------------------------- #

def prepare_inputs(
    labels_path: Path = LABELS,
    export: Path = loader.EXPORT,
    scenarios_path: Path = SCENARIOS,
    out: Path = INPUTS,
    report: Path | None = REPORT,
    label_log: Path | None = LABEL_LOG,
    users_db: Path = USERS_DB,
) -> list[dict]:
    """Write one `{"trace_id", "trace"}` record per eligible labeled conversation.

    First re-exports `label_log` to `labels_path` (`export_labels`), unless
    `label_log` is None. Each record's `trace` starts with the session's system
    prompt, then every turn up to the chosen one.
    """
    if label_log is not None:
        export_labels(MODE, log=label_log, out=labels_path)
    labels = read_labels(labels_path)
    users = read_users(users_db)
    scenarios = read_scenarios(scenarios_path)
    convs = _conversations(export)
    where = {t["trace_id"]: (sid, t["turn"]) for sid, turns in convs.items() for t in turns}
    missing = sorted(set(labels) - set(where))
    if missing:
        raise ValueError(f"labeled traces absent from {export}: {missing[:5]}")

    turn_labels: dict[str, dict[int, int]] = collections.defaultdict(dict)
    for tid, row in labels.items():
        sid, n = where[tid]
        turn_labels[sid][n] = row["label"]

    groups = group_variants(list(turn_labels), scenarios)
    kept = kept_scenarios(groups)

    records = []
    for sid in sorted(kept):
        n = choose_turn(turn_labels[sid])
        turns = convs[sid][:n]
        records.append({
            "trace_id": turns[-1]["trace_id"],
            "trace": [session_system_message(convs[sid], users)]
                     + [m for t in turns for m in turn_messages(t)],
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n")
    check = check_eligibility(records, labels, where, turn_labels, kept)
    print(check["summary"])
    if report is not None:
        write_report(report, records, labels, where, turn_labels, groups, convs, scenarios, check)
    return records


def conversation_label(turn_labels: dict[int, int]) -> int:
    return FAIL if FAIL in turn_labels.values() else PASS


def check_eligibility(records, labels, where, turn_labels, kept) -> dict:
    """Every record has a label; every eligible conversation has exactly one record."""
    unlabeled = [r["trace_id"] for r in records if r["trace_id"] not in labels]
    per_conv = collections.Counter(where[r["trace_id"]][0] for r in records if r["trace_id"] in where)
    not_one = sorted(sid for sid in kept if per_conv[sid] != 1)
    extra = sorted(set(per_conv) - kept)
    mismatch = [r["trace_id"] for r in records if r["trace_id"] in labels
                and labels[r["trace_id"]]["label"] != conversation_label(turn_labels[where[r["trace_id"]][0]])]
    ok = not (unlabeled or not_one or extra or mismatch)
    summary = (
        f"eligibility: {'OK' if ok else 'FAILED'} -- {len(records)} records, "
        f"{len(kept)} eligible conversations of {len(turn_labels)} labeled; "
        f"unlabeled records {len(unlabeled)}, conversations without exactly one record "
        f"{len(not_one)}, records outside the eligible set {len(extra)}, "
        f"record label != conversation label {len(mismatch)}"
    )
    if not ok:
        raise ValueError(f"{summary}: {unlabeled[:3]} {not_one[:3]} {extra[:3]} {mismatch[:3]}")
    return {"ok": ok, "summary": summary}


# --------------------------------------------------------------------------- #
# evidence for the variant rule and the report
# --------------------------------------------------------------------------- #

def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def variant_evidence(scenario_ids, scenarios, convs) -> list[tuple[str, int, float, float, float]]:
    """Median word-Jaccard of user request, replies and tool calls, per pair category.

    Measured over all pairs of labeled conversations, so a same-record pair can be
    compared with pairs that only share an intent and with all other pairs.
    """
    req = {s: _words(scenarios[s]["opening_message"]) for s in scenario_ids}
    rep = {s: _words(" ".join(t["reply"] for t in convs[s])) for s in scenario_ids}
    tools = {s: {(st["tool"], json.dumps(st["in"], sort_keys=True))
                 for t in convs[s] for st in t["steps"] if st["kind"] == "tool"}
             for s in scenario_ids}
    cats: dict[str, list[tuple[float, float, float]]] = collections.defaultdict(list)
    for a, b in itertools.combinations(sorted(scenario_ids), 2):
        if variant_key(scenarios[a]) == variant_key(scenarios[b]):
            cat = "same variant key"
        elif scenarios[a]["tuple"]["intent"] == scenarios[b]["tuple"]["intent"]:
            cat = "different key, same intent"
        else:
            cat = "different key, different intent"
        cats[cat].append((_jaccard(req[a], req[b]), _jaccard(rep[a], rep[b]),
                          _jaccard(tools[a], tools[b])))
    return [(c, len(v), *(round(statistics.median(x[i] for x in v), 2) for i in range(3)))
            for c, v in sorted(cats.items(), key=lambda kv: kv[0] != "same variant key")]


def write_report(path, records, labels, where, turn_labels, groups, convs, scenarios, check) -> None:
    def lab(sid):
        return "Fail" if conversation_label(turn_labels[sid]) == FAIL else "Pass"

    by_label = collections.Counter(
        "Pass" if labels[r["trace_id"]]["label"] == PASS else "Fail" for r in records)
    before = collections.Counter(lab(s) for s in turn_labels)
    lines = [
        "# HW5 Part B: judge inputs",
        "",
        "Generated by `analysis.run_judges.prepare_inputs()`; regenerate rather than edit.",
        f"Inputs: `{INPUTS.relative_to(REPO)}`. Labels: `{LABELS.relative_to(REPO)}` (1 = Pass),",
        f"exported by `export_labels` from the review app's log `{LABEL_LOG.relative_to(REPO)}`.",
        "",
        "## Record contents",
        "",
        "`trace` = the system prompt, then every turn up to the chosen one (user text,",
        "assistant text, tool calls with arguments, tool results). The system prompt is",
        "`agent/agent.py` `SYSTEM_PROMPT_TEMPLATE` rendered with the session's recorded",
        "role and user and the user's store from the users table; `system_message` raises",
        "when a session's recorded `prompt_version` is not that template's hash.",
        "No label, evidence, scenario id or review note enters a record.",
        "The judge text (`normalize_trace`) names the tool on every tool line,",
        "e.g. `tool_call get_policy: {\"policy_id\": \"cw-returns\"}`.",
        "",
        "## Independence rule",
        "",
        "Two labeled conversations are close variants when their scenarios share the same",
        "seeded order (`tuple.order_id`) or the same product (`tuple.product_id`), AND the",
        "same `tuple.intent`. Scenarios with no order or product (policy-page questions,",
        "out-of-scope and account requests) stand alone.",
        "One record is kept per group: the lowest scenario id, chosen without labels.",
        "Code: `variant_key`, `group_variants`, `kept_scenarios` in `analysis/run_judges.py`.",
        "",
        "Justification (by construction, not by a similarity threshold):",
        "",
        "- Same record and same intent: same tool data, same question; the group table",
        "  lists the tuple fields in which the members differ.",
        "- Same record, different intent: `scenarios/skill/SKILL.md` writes five challenge",
        "  scenarios per data-quality case on one record, asking different questions on",
        "  purpose; they stay separate.",
        "- Policy-page coverage scenarios have `data_quality_case_id` null and were generated",
        "  independently; a shared policy page does not make them variants.",
        "- Duplicate runs: the loader enforces one session per scenario",
        "  (`loader.check_one_session_per_scenario`); the only re-run, support-0144 attempt 1,",
        "  is in `loader.EXCLUDED` and is not labeled.",
        "",
        "Similarity over all pairs of labeled conversations, for reference (median",
        "word-Jaccard; `variant_evidence`):",
        "",
        "| Pair category | Pairs | User request | Replies | Tool calls (name + arguments) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cat, n, a, b, c in variant_evidence(list(turn_labels), scenarios, convs):
        lines.append(f"| {cat} | {n} | {a} | {b} | {c} |")
    lines += [
        "",
        "## Groups",
        "",
        "| Key | Scenarios (conversation label) | Differs in | Kept |",
        "| --- | --- | --- | --- |",
    ]
    for key, ids in sorted(groups.items(), key=lambda kv: kv[1][0]):
        if len(ids) > 1:
            members = ", ".join(f"{s} ({lab(s)})" for s in ids)
            tuples = [scenarios[s]["tuple"] for s in ids]
            differs = ", ".join(sorted(k for k in set().union(*tuples)
                                       if len({json.dumps(t.get(k)) for t in tuples}) > 1))
            lines.append(f"| {' / '.join(map(str, key))} | {members} | {differs} | {min(ids)} |")
    dropped = [s for ids in groups.values() for s in ids if s != min(ids)]
    lines += [
        "",
        (f"Dropped as variants: {len(dropped)} conversations "
         f"({sum(lab(s) == 'Fail' for s in dropped)} Fail, {sum(lab(s) == 'Pass' for s in dropped)} Pass)."),
        "",
        "## Conversations with more than one labeled turn",
        "",
        "Rule: the first Failing labeled turn, else the last labeled turn. The record holds",
        "every turn up to and including the chosen one.",
        "",
        "| Scenario | Turns in session | Labeled turns (label) | Chosen turn | Kept |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    kept = {where[r["trace_id"]][0] for r in records}
    for sid in sorted(s for s, t in turn_labels.items() if len(t) > 1):
        tl = turn_labels[sid]
        shown = ", ".join(f"{n} ({'Pass' if v == PASS else 'Fail'})" for n, v in sorted(tl.items()))
        lines.append(f"| {sid} | {len(convs[sid])} | {shown} | {choose_turn(tl)} | "
                     f"{'yes' if sid in kept else 'no'} |")
    lines += [
        "",
        "## Counts",
        "",
        f"Labeled conversations: {sum(before.values())} ({before['Pass']} Pass, {before['Fail']} Fail).",
        f"Input records after dedupe: {len(records)} ({by_label['Pass']} Pass, {by_label['Fail']} Fail),",
        "joined to the labels file after the inputs were written.",
        "hw5.md asks for at least 30 Pass and 30 Fail from independent conversations: "
        + ("met." if min(by_label["Pass"], by_label["Fail"]) >= 30 else
           f"NOT met ({min(by_label.values())} in the smaller class)."),
        "",
        f"Eligibility check ({check['summary']}).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# split (run once, after the grouping is approved)
# --------------------------------------------------------------------------- #

def split_data(mode: str = MODE) -> dict[str, list[str]]:
    """The handout's `split_labels` call; persists to analysis/state/splits.json."""
    from analysis.helpers import split_labels

    export_labels(mode)
    records = json.loads(Path("analysis/state/hw5_trace_inputs.json").read_text())
    splits = split_labels(
        mode,
        fractions=(0.20, 0.40, 0.40),
        seed=7,
        min_per_class=10,
        eligible_trace_ids=[record["trace_id"] for record in records],
    )
    return splits


# --------------------------------------------------------------------------- #
# Part C and D: development runs, freeze and test (hw5.md:178, hw5.md:211)
# --------------------------------------------------------------------------- #

def _save_metrics(split: str, metrics: dict) -> Path:
    path = REPO / "analysis" / "report" / f"{split}-{metrics['judge_id']}.json"
    path.write_text(json.dumps(metrics, indent=1) + "\n")
    return path


def run_development(mode: str, prompt_path: str | Path, judge_model: str = "gpt-4o-mini") -> dict:
    """Register the prompt for `judge_model`, run it on dev, save and return the metrics."""
    from analysis.helpers import judge_alignment, register_judge, run_judge

    record = register_judge(mode=mode, prompt_text=Path(prompt_path).read_text(), judge_model=judge_model)
    run_judge(record["judge_id"], split="dev", batch_size=10)
    metrics = judge_alignment(record["judge_id"], split="dev")
    _save_metrics("dev", metrics)
    return metrics


def run_test(judge_id: str) -> dict:
    """Freeze `judge_id` (once; a frozen version cannot be refrozen), run it on test, save the metrics."""
    from analysis.helpers import freeze_judge, judge_alignment, run_judge

    freeze_judge(judge_id)
    run_judge(judge_id, split="test", batch_size=10)
    metrics = judge_alignment(judge_id, split="test")
    _save_metrics("test", metrics)
    return metrics


if __name__ == "__main__":
    prepare_inputs()
