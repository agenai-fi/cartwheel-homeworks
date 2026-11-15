# AI Evals cohort 6 — Glossary

Shared vocabulary for the Cartwheel homeworks (HW1-HW9). Sources of truth: the handouts in
`ai-evals/cohort-6/cartwheel-homeworks/homework/` and `eval_cases/README.md`.

## Failure analysis (HW3-HW5)

**Trace**:
One recorded Cartwheel conversation: user messages, agent replies, tool calls and tool results.
_Avoid_: log, transcript (when meaning the reviewed unit)

**Failure mode**:
A named, recurring way the agent fails, defined in `analysis/state/patterns.json` (e.g. `unsupported_claim`).
_Avoid_: bug, error type, category

**Label**:
A reviewer's Pass or Fail verdict for one trace under one failure mode. HW4 stores 1 = failure present; HW5 exports 1 = Pass.
_Avoid_: annotation, tag

**Judge**:
An LLM prompt plus model that returns Pass or Fail for one failure mode on one trace.
_Avoid_: grader, evaluator (when meaning the LLM one)

**Accepted judge**:
A judge frozen in HW5 after its test-set TPR/TNR were measured; the only judges HW6 may use.
_Avoid_: approved judge, final judge

## Evaluation cases and CI (HW6)

**Case**:
One record in `eval_cases/cases.jsonl`: an input, a starting world, and the expected checks and judges for one failure mode.
_Avoid_: test, example, scenario (a scenario is the HW3 generator input a case may be based on)

**Check**:
A deterministic code test on a run's tool calls, final reply or end-state database (e.g. `tool_called`).
_Avoid_: assertion, rule

**Assertion**:
Plain-language statement of the expected behaviour in a case, for human readers; it produces no reward.
_Avoid_: check, expectation

**Task**:
The Harbor directory the exporter generates from one case (`task.toml`, environment, verifier tests).
_Avoid_: case (the task is derived; the case is the source)

**Trial**:
One execution of the agent on one task in a fresh Docker container, plus verification, ending in a reward. Harbor's `--n-attempts` sets how many trials each task gets.
_Avoid_: attempt (except in the flag name), run, rollout

**Reward**:
The verifier's score for one trial: 1 when every check passes and every judge verdict equals its expected value, else 0.
_Avoid_: score, grade

**Baseline**:
The five valid trials of an unclassified case, used once to set its kind.
_Avoid_: calibration run

**Regression case**:
A case that passed 5 of 5 baseline trials; any later failed trial blocks CI.
_Avoid_: must-pass test, guard

**Capability case**:
A case that failed at least one baseline trial; it reports results and never blocks CI. Records `baseline_pass_rate`.
_Avoid_: stretch case, flaky case

**n (observed trials)**:
The number of trials actually run for a case; c is how many of them passed.
_Avoid_: k, attempts

**k (product attempts)**:
The number of tries the product would get in use; the k in pass@k and pass^k. Independent of n, but k <= n.
_Avoid_: n, --n-attempts

**pass@k**:
Estimated probability that at least one of k attempts succeeds, from n trials with c passes: 1 - C(n-c,k)/C(n,k). Measures capability.
_Avoid_: success rate, pass rate

**pass^k**:
Estimated probability that all k attempts succeed: C(c,k)/C(n,k). Measures reliability.
_Avoid_: pass-all, consistency score
