# HW5 Part C spikes, 2026-11-13: why the `unsupported_claim` judge scored near chance

Diagnostic runs made after v3 scored dev J 0.19. None of these prompts was registered except E3, which became
`unsupported_claim-v4` (gpt-4o-mini) and `-v5` (gpt-5.5). Conclusions and the decisions they fed:
`analysis/report/hw5-report.md`. All runs: dev or train splits only; no test traces.

Invariants:
- Labels: `analysis/state/hw5_labels/unsupported_claim.jsonl` as of 2026-11-13 (1 = Pass, 0 = Fail).
- Every result file holds per-trace `label`, `pred` (1 = Pass) and the judge `critique`.
- "Old input" = Langfuse trace text without the agent system prompt (the runner's behaviour before
  `scale.judge_rows`, added the same day). "Fixed input" = `analysis/state/hw5_trace_inputs.json`.

## Runs

| Question | Prompt(s) | Model / path | Input | Split | Result files |
|---|---|---|---|---|---|
| Does v3 work at all on a few cases? | v3 (three edit rounds) | gpt-4o-mini, DocETL | old | 10 train | `results/smoke_v3_run{1,2,3}.json` |
| Do targeted wording changes help? | `prompts/v4-cand{A,B,C,D}.txt` vs v3 | gpt-4o-mini, DocETL | old | 14 train | `results/train14-*.json` |
| Is it the DocETL system prompt, the missing system prompt, or the model? | v3 | gpt-4o-mini / gpt-4.1 / gpt-4o, direct LiteLLM | old (r2, r4: system prompt prepended) | 14 train | `results/spike-r{1..5}*.json` |
| Does a stronger model fix it? | v3 | gpt-4.1, direct | old | dev | `results/spike-dev-gpt41.json` |
| Does one question per rule family help? | `prompts/v4-candE.txt` | gpt-4o-mini, DocETL / direct | old | 14 train, dev x2 | `results/train14-v4-candE.json`, `results/dev-v4-candE{,-rep}.json` |
| Do sharper Q1/Q2 wordings help? | `prompts/v4-candE3.txt` (E2 = E + Q1 edit) | gpt-4o-mini, DocETL / direct | fixed | dev x2 | `results/dev-v4-candE3-{A,B}.json` |
| Do separate single-rule detectors help? | `prompts/detector-*.txt`, OR-combined | gpt-4o-mini, DocETL | fixed | 14 train | `results/train-det-*.json` |
| Model x prompt matrix | v3, E3 | gpt-4.1, gpt-5-mini, gpt-5.5, direct | fixed | dev | `results/mx-*.json`, `results/mx2-gpt-5.5-v3.json` (replicate) |

The 14 train traces: Pass 0004 0045 0088 0095 0133 0171 0179 0193 0226; Fail 0028 0119 0142 0196 0213
(the three prompt examples and four suspected-stale labels excluded).

## Findings (each verified by the runs above)

- Wording variants A-D: 0-2 of 9 train Pass correct, 5/5 Fail; no variant changed behaviour.
- DocETL's own system message ("as precisely and exhaustively (i.e., high recall) as possible",
  `docetl/operations/utils/api.py`) and the missing agent system prompt: no effect on gpt-4o-mini verdicts.
- One question per rule family (E): mean dev J 0.28 vs v1 0.20; in E3 only Q1 ever fired (Q2, Q3, Q5: 0 of 56).
- Separate detectors: combined train J -0.38; each detector re-labelled computed dates and advice as its own
  failure type.
- Model matrix (dev J): gpt-4.1 v3 0.22 / E3 0.42; gpt-5-mini 0.58 / 0.64; gpt-5.5 0.81 (replicate 0.77) / 0.77.
  gpt-5.5 caught 25/25 dev Fails; gpt-5.5 accepts only temperature 1; replicate flipped 1 of 56 verdicts.

## Re-running

From the repository root, with `.env` loaded (`set -a && . ./.env && set +a`):

```
uv run python analysis/spikes/2026-11-13/scripts/spike_direct.py <prompt> <model> <sys:0|1> <out.json> DEV
uv run python analysis/spikes/2026-11-13/scripts/smoke_any.py <prompt> <out.json> DEV|<train scenario numbers...>
python3 analysis/spikes/2026-11-13/scripts/analyze_mx.py
```

`spike_direct.py` and `smoke_any.py` now read the fixed input, so a re-run of an "old input" row will differ.
Each call costs model usage; gpt-5.5 runs are the expensive ones.
