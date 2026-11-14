"""Part E pre-fill: which (trace, final mode) pairs can contain the mode at all.

hw4.md requires a present/absent judgment for every trace and final mode; the
reader (Module 2, "point of first failure" note) asks for exhaustive labeling where
it matters. Most pairs cannot contain the failure because the mode's trigger never
occurs in the trace -- an out-of-scope answer needs an out-of-scope request. This
module states each trigger as a code check, so:

- trigger absent  -> proposal "Absent", source `rule: trigger absent`, evidence = the rule;
- trigger present -> needs a reading (taxonomy judgment, or an agent proposal).

A trigger is retrieval, never a label: the reviewer accepts or flips every row in
the Labels view. Safety check: every known positive must trip its mode's trigger
(recall 1.0 on the taxonomy), or nothing is written.

    uv run python analysis/review_app/label_rules.py            # check + write rule proposals
    uv run python analysis/review_app/label_rules.py --dry-run  # check + counts only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.helpers import _state  # noqa: E402
from analysis.review_app import loader  # noqa: E402
from analysis.review_app.depth_search import _scenario, _tools  # noqa: E402

_DISPUTE = re.compile(r"\bdisput|\bchargeback", re.I)
_REVIEW_OFFER = re.compile(r"\bexception|\breview|\bescalat|\bticket\b|human (support|agent)|support team", re.I)
_POLICY_TALK = re.compile(r"\bcw-[a-z-]+|store-[a-z-]+-policy|\bpolicy\b|\bwindow\b|\b\d+[- ]days?\b|restocking"
                          r"|\boverrid|precedence|business days|\$\s?100\b|\bdisput|\beligib"
                          r"|\bcan(?:no|’|')t (?:start|issue|process|refund|cancel|return|open)", re.I)
# record states the scenario generator built as inconsistent (the dq cases) or refunded
_RECORD_STATES = {"order_missing_delivery_date", "order_reversed_dates", "order_store_mismatch",
                  "product_duplicate_title", "product_invalid_price", "product_missing_title", "order_refunded"}

# mode -> (rule text shown as evidence, predicate(view, turn_index) -> bool)
TRIGGERS = {
    "answered_out_of_scope_question": (
        "scenario intent is not out_of_scope (the designed request is a Cartwheel task)",
        lambda v, i: _scenario(v).get("intent") == "out_of_scope"),
    "missed_mandatory_escalation": (
        "no dispute, account change or issue_refund (ESC-1 to ESC-3) up to this turn",
        lambda v, i: _scenario(v).get("intent") in ("dispute", "account_change")
        or any(_DISPUTE.search(t["user"] or "") for t in v["turns"][: i + 1])
        or "issue_refund" in _tools(v["turns"][i])),
    "unnecessary_escalation": (
        "no escalate_to_human call and no review/exception/ticket offer in the reply",
        lambda v, i: "escalate_to_human" in _tools(v["turns"][i])
        or bool(_REVIEW_OFFER.search(v["turns"][i]["reply"] or ""))),
    "acted_on_inconsistent_record": (
        "record state is neither a data-quality case nor refunded, so no record inconsistency to act on",
        lambda v, i: _scenario(v).get("record_state") in _RECORD_STATES),
    "policy_claim_without_opening": (
        "the reply cites no policy and states no policy-like rule (window, days, fee, override, threshold, "
        "dispute, eligibility, what cannot be done)",
        lambda v, i: bool(_POLICY_TALK.search(v["turns"][i]["reply"] or ""))),
    "unsupported_claim": (
        "(no trigger: any reply can assert an unsupported fact)",
        lambda v, i: True),
}


def sample_traces() -> list[tuple[dict, int, str]]:
    """(view, turn index, trace id) for every sampled trace."""
    man = _state.read_json(_state.state_path("sample_manifest.json"))["traces"]
    views = {v["scenario_id"]: v for v in loader.all_views()}
    out = []
    for sid in sorted({m["scenario_id"] for m in man.values()}):
        v = views[sid]
        out += [(v, i, t) for i, t in enumerate(v["trace_ids"]) if t in man]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="check recall and print counts; write nothing")
    args = ap.parse_args()
    modes = [m for m in _state.read_json(_state.state_path("patterns.json"))["modes"] if m.get("status") != "dropped"]
    traces = sample_traces()
    ok = True
    worklist, rule_rows = {}, {}
    sampled = {t for _, _, t in traces}
    for m in modes:
        rule, trig = TRIGGERS[m["name"]]
        hit = {t for v, i, t in traces if trig(v, i)}
        missed = [t for t in m.get("example_trace_ids", []) if t in sampled and t not in hit]
        decided = set(m.get("example_trace_ids", [])) | set(m.get("close_negative_trace_ids", []))
        worklist[m["name"]] = sorted(t for t in hit if t not in decided)
        rule_rows[m["name"]] = [{"trace_id": t, "label": 0, "source": "rule: trigger absent",
                                 "evidence": f"trigger absent: {rule}"}
                                for v, i, t in traces if t not in hit and t not in decided]
        print(f"{m['name']:34} trigger present {len(hit):3} / {len(traces)} | taxonomy-decided {len(decided):3}"
              f" | to read {len(worklist[m['name']]):3} | positives missed by trigger {len(missed)}")
        if missed:
            ok = False
            print(f"   RECALL FAIL: {missed}")
    if not ok:
        print("a known positive does not trip its trigger: widen the rule; nothing was written")
        return 1
    if args.dry_run:
        return 0
    for name, rows in rule_rows.items():
        # The file also holds proposals from other sources (an agent pass); only rule rows are replaced.
        path = _state.state_path("label_proposals", f"{name}.json")
        kept = [p for p in _state.read_json(path, default=[]) if not p.get("source", "").startswith("rule:")]
        ruled = {p["trace_id"] for p in rows}
        _state.write_json(path, rows + [p for p in kept if p["trace_id"] not in ruled])
    _state.write_json(_state.state_path("label_proposals", "_to_read.json"), worklist)
    print(f"rule proposals written (other proposals kept); trigger-present pairs to read -> "
          f"{_state.state_path('label_proposals', '_to_read.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
