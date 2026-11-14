# HW5 report: an LLM judge for `unsupported_claim`

**Submitted judge: `unsupported_claim-v4`**, prompt `analysis/prompts/unsupported_claim-v4.txt`, model
`gpt-4o-mini`. A second judge, `unsupported_claim-v5` (same prompt, `gpt-5.5`), is reported as a secondary result.

## Failure mode

The last assistant reply states something specific about Cartwheel, the order, the agent's abilities, or what a
human will do, that no tool output, opened policy, user message or system prompt supports (SPEC RESP-3).
Pass and Fail rules: `analysis/state/patterns.json`; boundary decisions:
`analysis/report/hw5-unsupported_claim-boundary-review.md`.

## Labels and split

One label per independent conversation (1 = Pass, 0 = Fail): `analysis/state/hw5_labels/unsupported_claim.jsonl`.
Judge inputs: `analysis/state/hw5_trace_inputs.json`. Split once, seed 7, 20/40/40: `analysis/state/splits.json`.

| Split | Conversations | Pass | Fail |
|---|---|---|---|
| train | 28 | 22 | 6 |
| dev | 56 | 31 | 25 |
| test | 57 | 28 | 29 |

**Why the labels changed after v2.** v0-v2 disagreed with the labels in a shared way. Sorting their common errors
showed that many close calls turned on rules the boundary did not state. Four rules were added: what a human or
the escalation will do, statements that depend on today's date, undefined policy terms, and conclusions drawn from
record fields. Six reviewers then re-read all 113 dev and test labels blind (without seeing the labels) under the
new rules, and 19 changes were approved: 18 Pass to Fail, 1 Fail to Pass. Counting all re-read rounds, 13 dev and
16 test labels differ from those v0 was first scored on. Inputs and splits did not change. v0-v2 are scored against
the new labels below; v3 was written for them.

**No labels changed after seeing v4 or v5 output.** The last label change was made at 11:38 UTC on 2026-11-13;
v4 and v5 were registered at 23:49 and frozen at 23:52, before any test call.

## Development

Pass is the positive class. J = TPR + TNR − 1. Files: `analysis/report/dev-unsupported_claim-v<N>.json`.

| Version | Model | Dev TPR | Dev TNR | J | Change |
|---|---|---|---|---|---|
| v0 | gpt-4o-mini | 0.87 | 0.20 | 0.07 | first draft |
| v1 | gpt-4o-mini | 0.52 | 0.68 | 0.20 | one claim-and-source test |
| v2 | gpt-4o-mini | 0.42 | 0.52 | −0.06 | revision |
| v3 | gpt-4o-mini | 0.23 | 0.96 | 0.19 | rewritten after the boundary and 29 labels changed |
| **v4** | gpt-4o-mini | 0.65 | 0.60 | 0.25 | one question per rule family |
| v5 | gpt-5.5 | 0.77 | 1.00 | 0.77 | v4's prompt on a stronger model |

v0-v3 ran on judge input without the agent's system prompt, because `CARTWHEEL_JUDGE_TRACE_SOURCE`
(`hw5.md:107`) was not set. v4 and v5 use the saved inputs (`scale.judge_rows` now reads them for every split run).
v0-v2 ran at the API default temperature, v3 and v4 at 0, v5 at 1 (the only value `gpt-5.5` accepts). Temperature 0
halved verdict flips between identical `gpt-4o-mini` runs (17 to 9 of 56) without changing J
(`analysis/report/spike-v1-dev-noise.json`). So v3 to v4
changed the input as well as the prompt; adding the system prompt alone did not change `gpt-4o-mini`'s verdicts on
14 train conversations, which points to the prompt structure, but that check was on train only.

**One disagreement.** support-0038, labeled Pass: "Your last day to return it is July 5, 2026 … delivered May 21",
with a 45-day store policy opened. v3 judged Fail ("the specific date is not confirmed"). The judge was wrong:
May 21 + 45 days = July 5. Most of v3's 24 false Fails were the same kind: a computed date, a refusal backed by the
system prompt, a "not found" report, or advice. Wording changes did not fix it; restructuring the prompt into one
question per rule family (v4) did, for this trace.

**Why revising stopped.** On `gpt-4o-mini`, no change moved dev J above 0.33: six wording variants and three
prompt structures (`analysis/spikes/2026-11-13/README.md`). Rewrites moved which side it erred on (v0 lenient,
TNR 0.20; v3 strict, TPR 0.23) without separating the classes. v4's remaining errors show why: all 11 clean replies it
failed were flagged by its first question, which the model uses as a catch-all, and 4 of its 10 missed failures
depend on today's date, a question it never answered yes on any dev trace. The model follows each question's
headline and ignores its qualifying clauses. This is beyond the two revisions `hw5.md:205` allows: v3 was needed
because the ground truth changed after v2, and the 11 unregistered candidate prompts scored on train or dev during
the diagnosis count as iterations too. v4 was chosen on dev as the best `gpt-4o-mini` version, not on test.

## Test results for v4 (`analysis/report/test-unsupported_claim-v4.json`)

| TP | FN | TN | FP | TPR [95% CI] | TNR [95% CI] | J |
|---|---|---|---|---|---|---|
| 20 | 8 | 19 | 10 | 0.71 [0.53, 0.85] | 0.66 [0.47, 0.80] | 0.37 |

Recompute from saved predictions:
`uv run python -c "from analysis.helpers import judge_alignment as a; print(a('unsupported_claim-v4', 'test'))"`

## Would I use the judge?

**No, not to measure or gate the agent.** v4 misses about a third of failures (TNR 0.66, interval down to 0.47) and
fails about 3 in 10 clean replies. With J 0.37 and intervals about 0.3 wide, a failure rate it reports cannot be
trusted. At most it could order traces for human review.

## Secondary result: the same prompt on gpt-5.5

v5 was frozen with v4, before any test call, so the two differ only in the model
(`analysis/report/test-unsupported_claim-v5.json`).

| TP | FN | TN | FP | TPR [95% CI] | TNR [95% CI] | J |
|---|---|---|---|---|---|---|
| 22 | 6 | 26 | 3 | 0.79 [0.60, 0.90] | 0.90 [0.74, 0.96] | 0.68 |

The model, not the prompt, limits v4: on dev, the same prompt scored J 0.25-0.33 on `gpt-4o-mini` and 0.77 on
`gpt-5.5`. Not separated: the models that worked (`gpt-5-mini`, `gpt-5.5`) are reasoning models run at
temperature 1; those that did not (`gpt-4o-mini`, `gpt-4o`, `gpt-4.1`) are non-reasoning models run at 0. Prompt structure still mattered for the mid-tier `gpt-4.1` (dev J 0.22 with v3, 0.42 with v4's prompt), not for `gpt-5.5` (0.81 and 0.77). v5 would be usable as a screen with human review of its Fails. It is not the submission because it
uses a different model than `hw5.md:176` requires.

## Limits

- 57 test conversations: each rate is uncertain by about ±0.15.
- One prompt rule (a last day stated in the user's terms passes) was motivated partly by test conversations 0054
  and 0203. Without them, v4's test TPR is 0.69.
- Repeat runs vary: 2 of 56 dev verdicts differed between two v4 runs.
- Train labels were not re-read under the final boundary rules.
