# HW4 review summary

## Reviewed sample

114 distinct traces from 100 conversations (Cartwheel creates one trace per user turn). No trace counts toward more than one batch; the record is `analysis/state/sample_manifest.json`.

| Part B batch | Method | Conversations | Traces |
|---|---|---|---|
| 1 | uniform random (seeds 4, 5) | 15 | 18 |
| 1 | k-means cluster representatives (k = 15, seed 7) | 15 | 16 |
| 2 | stratified by user role, chosen before outcomes (10 shopper, 10 merchant, 10 support) | 30 | 32 |
| 3 | depth searches (`depth-1` .. `depth-5`) for candidate modes and close negatives | 25 | 31 |
| 4 | uniform random after the draft taxonomy | 15 | 17 |

Every trace has an open-coding verdict (a failure note on its first failure, or "no failure observed").

## New modes in the final 15

**0.** Batch 4 produced 4 failures (support-0011, 0060, 0099, 0122), all in existing modes: three `unnecessary_escalation`, one `unsupported_claim`. The taxonomy is treated as stable; no extra batch was reviewed.

## Final taxonomy and sample fractions

Every trace was labeled against every final mode: 114 traces x 6 modes = 684 present/absent labels in `analysis/state/labels/<mode>.jsonl`, each also written to Langfuse as a score named after the mode (1 = failure present). These are **sample fractions**, not prevalence: the depth searches deliberately over-sample likely failures (7 of the 11 `policy_claim_without_opening` positives came from the depth search for that mode).

| Mode | Definition (short) | Evaluator | Requirement source | Failures | Sample fraction |
|---|---|---|---|---|---|
| `unsupported_claim` | reply asserts a platform, order or capability fact that no tool output, policy or user message supports | LLM judge | RESP-3 | 16 | 14.0% |
| `policy_claim_without_opening` | reply states or cites a policy rule whose document was never opened with `get_policy` (a snippet is not opened) | code | RESP-1 + recorded revision (below) | 11 | 9.6% |
| `unnecessary_escalation` | escalates a case the policy and record already resolve, or offers an exception path no policy describes | LLM judge | ESC-3, cw-escalations | 9 | 7.9% |
| `missed_mandatory_escalation` | a case SPEC always sends to a human (e.g. a dispute) ends without `escalate_to_human` | LLM judge | ESC-3, cw-disputes | 7 | 6.1% |
| `answered_out_of_scope_question` | for a request SCOPE-2 says to refuse, the reply holds more than a refusal and a redirect | LLM judge | SCOPE-2 | 5 | 4.4% |
| `acted_on_inconsistent_record` | relies on a record field that contradicts another field or a policy the agent had in context, without saying so | LLM judge | RESP-3 | 4 | 3.5% |

Positives, close negatives, boundaries and originating annotations per mode are in `analysis/state/patterns.json`. Four candidate modes were dropped by status with fewer than three positives: `return_deadline_miscalculated` (support-0072), `treated_capped_results_as_complete` (support-0162), `skipped_order_search` (support-0207, 0243), `reply_contradicts_retrieved_data` (merged away).

**How the labels were made.** Pairs a mode's trigger cannot reach (e.g. an out-of-scope answer on a return-deadline question) were proposed "absent" by a recall-checked rule (`analysis/review_app/label_rules.py`; every known positive trips its trigger). The taxonomy's own positives and close negatives are the reviewer's judgments. The remaining pairs carry an agent proposal with evidence. The reviewer read in context every pair the agent proposed as a failure or marked unsure (part of them blind: own answer first, then the agent's), plus a careful pass over `policy_claim_without_opening`; the rest were accepted in bulk. Each record says which: `accepted_via` = focus (read, 28 labels) or bulk (656), with `proposal_source`.

## Taxonomy revision: `acted_on_inconsistent_record`, judged by what the agent saw

On 2026-10-05 the mode was widened to "fields that break the platform contract" after the seed defect below surfaced in support-0213, 0113 and 0123. On 2026-10-06 it was narrowed back: `get_order` returns no refund rows and the system prompt gives no threshold value, so "refunded" with only a queued refund is invisible unless the agent had the \$100 rule in context. support-0213 (opened cw-refunds, \$281.25 order called "already complete") and support-0123 (saw the cw-refunds snippet) stay positive; support-0083, 0093, 0103, 0113 and 0133, the same defect with no rule in context, became close negatives. The defect itself is reported as a data finding, not an agent failure.

## Specification revisions

| Revision | Where | Motivating annotations |
|---|---|---|
| **RESP-6** (added to `SPEC.md`): a return is on time if started by delivery date + window days, inclusive | SPEC.md | support-0158, support-0179 |
| Citing or stating a policy rule requires `get_policy`; a search snippet does not count, because snippets are cut at 300 characters with no marker | recorded here (requirement source of `policy_claim_without_opening`) | support-0177 vs 0161; support-0248 (snippet actually cut) |
| ESC-3 applies to actions or decisions the agent cannot resolve; for missing information, stating the gap (RESP-3) is enough and escalation is allowed, not required | recorded here | support-0193, 0206 |
| ESC-2 "account changes" means Cartwheel accounts; SSO / Zendesk credential requests fall under SCOPE-2 | recorded here | support-0241 |
| Candidate: refund amounts must be grounded in what was paid and checked against amounts already refunded | proposed, not applied | support-0045 (see data findings) |

## Data findings (not agent failures)

- The seed marks orders `refunded` while their refund is only `queued_for_approval` (418 of 579 refunded orders; `seed/generate.py`). This contradicts TOOL-7.
- `issue_refund` checks only that the amount is positive and at most the order total; any auto-approved amount marks the whole order refunded, and `refund_eligible` is never cleared afterwards (order 8438: status refunded, still eligible after refund 578). A second refund on the same order would pass every check.

## Search for additional instances

Depth searches (`analysis/review_app/depth_search.py`) produced 27 suggestions: 16 accepted, 11 rejected, recorded in `analysis/state/suggestions.json`. The out-of-scope search was repeated after its definition was revised (`depth-5`, round 2). Example rejection: support-0223 matched the offer pattern "I can help you", but the sentence redirects to Cartwheel products ("I can help you compare products that are available through Cartwheel"), a refusal plus redirect, which the boundary puts on the negative side.

## Comparison with AgentDebug

AgentDebug's AgentErrorTaxonomy (Zhu et al., arXiv 2509.25370) has five modules: memory, reflection, planning, action, system.

| This taxonomy | Nearest AgentDebug type | Note |
|---|---|---|
| `policy_claim_without_opening` | Memory: Retrieval Failure | narrower: the document existed and was not opened before it was cited |
| `unsupported_claim` | Memory / Reflection: Hallucination | same idea, stated against this product's sources |
| `acted_on_inconsistent_record` | Reflection: Outcome Misinterpretation | the revision above matches AgentDebug's split: a broken environment is a System "Environment Error", not the agent's fault |
| `unnecessary_escalation`, `missed_mandatory_escalation`, `answered_out_of_scope_question` | none | product-policy compliance has no AgentDebug counterpart; these come from SPEC |

Possible omission flagged by AgentDebug: **Action: Parameter Error** (an unreasonable tool argument). The refund-amount gap above makes it plausible here, but no reviewed trace showed a wrong amount (support-0045's \$54.25 is exactly one of two items), so no mode was added.
