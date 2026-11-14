"""Depth search: the coding agent's retrieval of candidate instances for one mode.

hw4.md Part D: "Treat a similarity score, model prediction, or deterministic
filter as a retrieval signal rather than a label. Review every returned trace
yourself. Record whether you accepted or rejected each suggestion. Repeat the
search after revising the mode definition. Recheck earlier traces when a mode
discovered late in the process may apply to them."

So a hit is written as a *suggestion* (status pending) for the review app, never
as a label. Two ways to run it:

* default -- search conversations nobody has read; each hit's conversation joins
  the sample manifest as a depth batch (Part B batch 3).
* --recheck -- search conversations already in the manifest; suggestions only,
  no manifest change, because one trace counts toward one batch.

`--round` goes into every suggestion id, and the batch record keeps the rule and
the mode definition it ran with, so a search repeated after a definition
revision is a new, distinguishable round. Run with the review server stopped:
this is a separate process writing the same state files.

    uv run python analysis/review_app/depth_search.py policy_claim_without_opening \\
        --batch depth-1 --round 1 --weak 6 --seed 11
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.helpers import _state  # noqa: E402
from analysis.review_app import loader  # noqa: E402

_RULE = re.compile(r"takes precedence|store policy wins|overrid|store-specific|store's own", re.I)


# A sentence ends at ". " / "! " / "? " or at any newline; a newline is a hard
# stop so a markdown bullet is quoted alone, never spliced with the next line.
_END = re.compile(r"(?<=[.!?]) |\n")
_BULLET = re.compile(r"^(?:[-*•]|\d+\.)\s+")
_IDS = re.compile("|".join(map(re.escape, sorted(loader.KNOWN_POLICY_IDS, key=len, reverse=True))))


def _quote(text: str, rx: re.Pattern, cap: int = 300) -> str:
    """The whole sentence holding the match; over `cap` chars, a word-aligned
    window around the match with each cut marked "…"."""
    m = rx.search(text)
    lo, hi = (m.start(), m.end()) if m else (0, 0)
    a = max((e.end() for e in _END.finditer(text, 0, lo)), default=0)
    b = next((e.start() for e in _END.finditer(text, hi)), len(text))
    # A citation trails the claims it sources ("... for its orders. [cw-returns]"),
    # so a sentence of nothing but policy ids quotes its whole line.
    if not re.search(r"\w", _IDS.sub("", text[a:b])):
        a = text.rfind("\n", 0, a) + 1
    sent = text[a:b]
    lead = len(sent) - len(sent.lstrip())
    lead += (bm.end() if (bm := _BULLET.match(sent[lead:])) else 0)
    a, b = a + lead, a + len(sent.rstrip())
    if b - a <= cap:
        return text[a:b]
    pad = max(0, (cap - (hi - lo)) // 2)
    # slack one side cannot use goes to the other, so a trailing match keeps its claim
    x, y = max(a, min(lo - pad, b - cap)), min(b, max(hi + pad, a + cap))
    if x > a:   # snap each cut to a space outside the match, never mid-word
        x = text.find(" ", x, lo) + 1 or x
    if y < b and (cut := text.rfind(" ", hi, y)) >= 0:
        y = cut
    return ("…" if x > a else "") + text[x:y].strip() + ("…" if y < b else "")


def _hit(turn: dict, reason: str, quote: str) -> tuple[dict, dict]:
    # `evidence` is the fused string earlier records carry; kept for back-compat.
    return turn, {"reason": reason, "quote": quote, "evidence": f"{reason}: {quote}"}


def policy_claim_without_opening(view: dict) -> tuple[str, list[tuple[dict, dict]]] | None:
    """(strength, [(turn, {reason, quote, evidence}), ...]) for a hit -- one entry
    per matching turn.

    strong: a reply cites a policy id the conversation never opened (the rail's
            `cited_not_opened`, a deterministic filter).
    weak:   a reply states a store-override rule while `cw-store-overrides` was
            never opened. `cw-returns` and a store's own page both mention
            overrides too, so most weak hits are near misses (close negatives).
    """
    unopened = view["policy"]["cited_not_opened"]
    strong = [_hit(t, f"cites {pid}, never opened with get_policy", _quote(t["reply"] or "", loader._CITE_RE[pid]))
              for t in view["turns"] for pid in unopened
              if loader._CITE_RE[pid].search(t["reply"] or "")]
    if strong:
        return "strong", strong
    if "cw-store-overrides" in view["policy"]["opened"]:
        return None
    opened = ", ".join(view["policy"]["opened"]) or "none"
    weak = [_hit(t, f"states a store-override rule; cw-store-overrides not opened (opened: {opened}) "
                    "-- may be a near miss", _quote(t["reply"] or "", _RULE))
            for t in view["turns"] if _RULE.search(t["reply"] or "")]
    return ("weak", weak) if weak else None


_SCENARIOS: dict[str, dict] = {}


def _scenario(view: dict) -> dict:
    """The scenario's designed tuple (intent, record_state) -- metadata about what was
    asked, never the expected outcome, so a filter on it is retrieval, not a label."""
    if not _SCENARIOS:
        import json
        for line in (REPO / "scenarios" / "support_scenarios.jsonl").read_text().splitlines():
            row = json.loads(line)
            _SCENARIOS[row["id"]] = row["tuple"]
    return _SCENARIOS.get(view["scenario_id"], {})


def _tools(turn: dict) -> list[str]:
    return [s.get("tool") for s in turn["steps"] if s["kind"] == "tool"]


_DISPUTE_WORD = re.compile(r"\bdisput|\bchargeback", re.I)


def missed_mandatory_escalation(view: dict) -> tuple[str, list[tuple[dict, dict]]] | None:
    """strong: a dispute scenario (designed intent) whose conversation never calls
    escalate_to_human (ESC-3: disputes always go to a human). A text match on
    'dispute' was tried first and rejected: it missed support-0103 and hit four
    merchant policy questions that use the word in another sense."""
    if _scenario(view).get("intent") != "dispute" or any("escalate_to_human" in _tools(t) for t in view["turns"]):
        return None
    # Anchor on the first turn whose user message raises the dispute: that reply is
    # the first one where escalation was required (support-0093 raises it in turn 1).
    # Anchoring on the last turn put the suggestion on a turn the reviewer was not
    # judging for this failure.
    first = next((t for t in view["turns"] if _DISPUTE_WORD.search(t["user"] or "")), view["turns"][0])
    return "strong", [_hit(first, "dispute request; no escalate_to_human in the conversation",
                           _quote(first["reply"] or "", re.compile(r".")))]


_GUIDANCE = re.compile(r"\busually\b|look for|point you to|send this|^\s*\d+\.\s", re.I | re.M)


_OFFER = re.compile(r"\bif you (tell|share|send|let) me\b|\bI can (help you|point|walk|draft|look)\b", re.I)


def answered_out_of_scope_question(view: dict) -> tuple[str, list[tuple[dict, dict]]] | None:
    """Round 2 (definition revised 2026-10-06: more than a refusal and a redirect).
    strong: an out-of-scope scenario (designed intent) whose reply goes past a refusal
    with how-to guidance or an offer to do the task.
    weak:   an out-of-scope scenario whose reply has neither -- a plain refusal, the
            close-negative candidate."""
    if _scenario(view).get("intent") != "out_of_scope":
        return None
    hits = [_hit(t, "out-of-scope request; reply gives guidance or offers the task beyond a refusal",
                 _quote(t["reply"] or "", _GUIDANCE if _GUIDANCE.search(t["reply"] or "") else _OFFER))
            for t in view["turns"]
            if _GUIDANCE.search(t["reply"] or "") or _OFFER.search(t["reply"] or "")]
    if hits:
        return "strong", hits
    first = view["turns"][0]
    return "weak", [_hit(first, "out-of-scope request; reply looks like a plain refusal -- expect a close negative",
                         _quote(first["reply"] or "", re.compile(r".")))]


_ORDER_NO = re.compile(r"#\s?\d{2,5}\b|\bor?der\s*(?:number|no\.?|id)?\s*#?\s*\d{2,5}\b", re.I)
_ORDER_INTENTS = {"find_order", "order_status", "refund_request", "cancel_request", "return_deadline", "dispute"}
_LOOKUPS = {"find_order", "list_my_orders", "get_order"}


def skipped_order_search(view: dict) -> tuple[str, list[tuple[dict, dict]]] | None:
    """An order question with no order number whose first turn runs no order lookup.
    strong: the user names a product and a store (a find_order query exists).
    weak:   no product named -- asking for the order id may be right (expect close negatives)."""
    tup, first = _scenario(view), view["turns"][0]
    if tup.get("intent") not in _ORDER_INTENTS or _ORDER_NO.search(first["user"] or "") \
            or _LOOKUPS & set(_tools(first)):
        return None
    named = bool(re.search(r"\bfrom [A-Z]", first["user"] or ""))
    reason = ("order described by product and store, no order number; no find_order call" if named
              else "order question with no order number and no product named; no lookup -- may be a near miss")
    return ("strong" if named else "weak"), [_hit(first, reason, _quote(first["reply"] or "", re.compile(r".")))]


SEARCHES = {"policy_claim_without_opening": (policy_claim_without_opening, _RULE.pattern),
            "missed_mandatory_escalation": (missed_mandatory_escalation, "intent == dispute and no escalate_to_human"),
            "answered_out_of_scope_question": (answered_out_of_scope_question,
                                               f"guidance: {_GUIDANCE.pattern} | offer: {_OFFER.pattern}"),
            "skipped_order_search": (skipped_order_search, _ORDER_NO.pattern)}
SIGNALS = {
    "policy_claim_without_opening": {
        "strong": "deterministic: cites a policy it never opened (rail cited_not_opened)",
        "weak": "text match: states a store-override rule; cw-store-overrides not opened, "
                "but cw-returns or the store's page may still support it -- expect near misses"},
    "missed_mandatory_escalation": {
        "strong": "deterministic: scenario intent dispute; no escalate_to_human call in the conversation"},
    "answered_out_of_scope_question": {
        "strong": "scenario intent out_of_scope + text match: reply contains how-to guidance "
                  "(usually / look for / point you to / send this / numbered steps) or an offer "
                  "to do the task (if you tell me / I can help you)",
        "weak": "scenario intent out_of_scope, no guidance or offer matched: a plain refusal -- "
                "expect close negatives"},
    "skipped_order_search": {
        "strong": "deterministic: order intent, no order number, product and store named, no order lookup",
        "weak": "deterministic: order intent, no order number, no product named, no lookup -- expect near misses"},
}


_STRONG_PID = re.compile(r"cites (\S+) \(never opened\)")


def backfill_reason_quote(pool: list[dict] | None = None) -> tuple[int, list[str]]:
    """Add `reason` and `quote` to records written before those keys existed.

    Recomputed from the trace views, matched on scenario, turn and (strong) the
    cited policy id in the old `evidence`. Only those two keys are written: the
    record's decisions and its original `evidence` are the reviewer's history.
    Returns (records filled, ids left untouched because they could not be matched).
    """
    path = _state.state_path("suggestions.json")
    suggestions = _state.read_json(path, default=[])
    views = {v["scenario_id"]: v for v in loader.all_views(pool)}
    filled, missed = 0, []
    for sg in suggestions:
        search, view = SEARCHES.get(sg.get("mode"), (None,))[0], views.get(sg.get("scenario_id"))
        h = search(view) if search and view else None
        pid = _STRONG_PID.match(sg.get("evidence", ""))
        want = f"cites {pid[1]}," if pid else "states "
        fields = next((f for t, f in (h[1] if h and h[0] == sg.get("strength") else [])
                       if t["turn"] == sg.get("turn") and f["reason"].startswith(want)), None)
        if fields is None:
            missed.append(sg.get("id"))
            continue
        sg["reason"], sg["quote"] = fields["reason"], fields["quote"]
        filled += 1
    _state.write_json(path, suggestions)
    return filled, missed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=sorted(SEARCHES))
    ap.add_argument("--batch", help="batch name, e.g. depth-1 (or recheck-1 with --recheck); required for a search")
    ap.add_argument("--round", type=int, default=1, help="search round; bump after revising the definition")
    ap.add_argument("--weak", type=int, default=6,
                    help="weak-hit conversations to sample (all strong kept; --recheck keeps all)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--recheck", action="store_true",
                    help="search already-read conversations; write suggestions only")
    ap.add_argument("--backfill", action="store_true",
                    help="one-off: add reason/quote to existing suggestions; changes no other key")
    args = ap.parse_args()
    if args.backfill:
        filled, missed = backfill_reason_quote()
        print(f"reason/quote added to {filled} suggestions; {len(missed)} left untouched")
        for sid in missed:
            print(f"  unmatched: {sid}")
        return 0
    if not (args.mode and args.batch):
        ap.error("a search needs a mode and --batch")

    manifest_path = _state.state_path("sample_manifest.json")
    manifest = _state.read_json(manifest_path)
    if any(b["batch"] == args.batch for b in manifest["batches"]):
        raise SystemExit(f"batch {args.batch!r} already exists")
    if any(b.get("mode") == args.mode and b.get("round") == args.round for b in manifest["batches"]):
        raise SystemExit(f"{args.mode} round {args.round} already ran; a repeat search needs a new --round")
    read_before: set[str] = set()
    for b in manifest["batches"]:
        read_before |= set(b.get("excluded_as_previously_read", []))
    in_manifest = {m["scenario_id"] for m in manifest["traces"].values()}
    views = loader.all_views()
    pool = ([v for v in views if v["scenario_id"] in in_manifest] if args.recheck
            else [v for v in views if v["scenario_id"] not in in_manifest | read_before])

    search, rule = SEARCHES[args.mode]
    hits = {"strong": [], "weak": []}
    for v in sorted(pool, key=lambda v: v["scenario_id"]):
        if (h := search(v)):
            hits[h[0]].append((v, h[1]))
    n_weak = len(hits["weak"]) if args.recheck else min(args.weak, len(hits["weak"]))
    weak = sorted(random.Random(args.seed).sample(hits["weak"], n_weak), key=lambda h: h[0]["scenario_id"])
    chosen = [("strong", *h) for h in hits["strong"]] + [("weak", *h) for h in weak]

    definition = next((m.get("definition") for m in (_state.read_json(
        _state.state_path("patterns.json"), default={"modes": []})["modes"]) if m["name"] == args.mode), None)
    record = {"batch": args.batch, "method": "depth search recheck" if args.recheck else "depth search",
              "mode": args.mode, "round": args.round, "rule": rule, "definition": definition,
              "seed": args.seed, "hits_in_pool": {k: len(v) for k, v in hits.items()},
              "taken": {"strong": len(hits["strong"]), "weak": len(weak)},
              "scenarios": sorted(v["scenario_id"] for _, v, _ in chosen), "eligible_at_draw": len(pool)}
    if args.recheck:
        record["counts_toward_batch"] = False   # these traces already belong to earlier batches
    else:
        record["part_b_batch"] = 3   # hw4.md Part B batch 3: depth searches
        record["excluded_as_previously_read"] = sorted(read_before)
        for _, v, _ in chosen:
            for tid in v["trace_ids"]:
                manifest["traces"][tid] = {
                    "scenario_id": v["scenario_id"], "session_id": v["session_id"], "batch": args.batch,
                    "selection": "depth search", "reason": f"{args.mode} round {args.round}"}
    manifest["batches"].append(record)

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    sugg_path = _state.state_path("suggestions.json")
    suggestions = _state.read_json(sugg_path, default=[])
    have = {s["id"] for s in suggestions}
    # A trace pending or accepted for this mode is not suggested again (accepting a
    # second time would put two failure notes on one step). A rejected one may be,
    # in a later round after the definition changed.
    pending = {s["trace_id"] for s in suggestions
               if s.get("mode") == args.mode and s.get("status", "pending") in ("pending", "accepted")}
    new = []
    for strength, v, turns in chosen:
        for turn, fields in turns:
            sid = f"{args.mode}:r{args.round}:{v['scenario_id']}:t{turn['turn']}"
            if sid in have or turn["trace_id"] in pending:
                continue
            have.add(sid)
            new.append({
                "id": sid, "trace_id": turn["trace_id"], "turn": turn["turn"], "step": "reply",
                "session_id": v["session_id"], "scenario_id": v["scenario_id"], "mode": args.mode,
                "round": args.round, "signal": SIGNALS[args.mode][strength], "strength": strength,
                **fields, "author": "agent", "status": "pending", "created_at": now,
                "search_batch": args.batch})
    # Suggestions first: ids are deterministic and skipped when present, so a crash
    # before the manifest write is recovered by re-running with the same --batch.
    record["suggested_scenarios"] = sorted({s["scenario_id"] for s in new})
    _state.write_json(sugg_path, suggestions + new)
    _state.write_json(manifest_path, manifest)

    print(f"{len(pool)} {'already-read' if args.recheck else 'unread'} conversations searched for "
          f"{args.mode} (round {args.round}): {len(hits['strong'])} strong, {len(hits['weak'])} weak; "
          f"took all strong + {len(weak)} weak (seed {args.seed}) as {args.batch!r}; {len(new)} suggestions")
    for s in new:
        print(f"  {s['strength']:<6} {s['scenario_id']}  turn {s['turn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
