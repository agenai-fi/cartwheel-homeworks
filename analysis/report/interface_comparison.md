# Interface comparison: `analysis/review_app/` vs the reference interface

## Retained from the reference

**Labels live in Langfuse as scores, with a local mirror under `analysis/state/`.** The reference syncs labeled annotations through `langfuse_io.write_label_score` (`analysis/server.py`, `_sync_annotation_scores`). The review app keeps that design: each present/absent judgment is appended to `analysis/state/labels/<mode>.jsonl` and written as a score named after the mode. One addition: a stable score id per (trace, mode), so a flipped label updates its score instead of adding a second one, and a read-back check before a label counts as synced (the SDK queues scores and only logs ingestion errors).

## Changed after inspecting the traces

**The conversation is shown with its evidence, and the evidence is computed.** A dry run of ten hand-read traces found that every missed failure sat between a claim in the reply and the evidence above it (a tool output or a policy). So the app groups a session's traces in order, shows every tool call and result inline next to the reply, and adds a rail of computed facts: order fields with anomalies flagged, and policy provenance split into opened / seen only in a search snippet / cited. The split matters: "retrieved" in the first version conflated opened with surfaced, and the `policy_claim_without_opening` failures were invisible until it was separated. For Part E the same idea drives the labeling queue: the turn being labeled is shown in full, earlier turns collapse to one line per tool call, and the reviewer answers before seeing the agent's proposal.

## Remaining limitation

**Most labels were accepted in bulk from proposals.** Of 684 labels, 28 were read in context and 656 were bulk-accepted: the reviewer's own taxonomy judgments, a recall-checked trigger rule, or a confident agent "absent". A failure the agent missed and did not flag as unsure would go unreviewed. Each record carries `accepted_via` and `proposal_source`, so the bulk set can be audited or re-read later, but the interface does not itself sample those rows for spot checks. The layout also has no phone-width design.
