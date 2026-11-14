# Reference for writing unsupported_claim-v0.txt (not sent to the judge)

## How the prompt file is used
- `analysis/helpers/scale.py:235-248`: the runner sends `<your prompt text>`, then
  `--- Trace to evaluate ---` and the conversation text, then its own line asking for a critique
  first and a result of exactly Pass (failure absent) or Fail (failure present), with the output
  schema `{critique, result}`.
- Do not write `{{` or `{%` anywhere in the prompt: DocETL treats the text as a Jinja template.
- Every `<<TODO …>>` must be gone before the first run.

## Handout requirements (homework/module-2/hw5.md:150-152)
- Pass and Fail rules; a clear Pass, a clear Fail and a borderline example, from training only.
- Output: a critique with specific trace evidence, then the verdict.
- Review the boundary with neighbouring issues before running it.
- Tell the judge to evaluate the trace without following instructions quoted inside it.
- At most two prompt revisions after v0 (hw5.md:205). Pass is the positive class.

## Definition and boundary (analysis/state/patterns.json, mode unsupported_claim)
Read them there: they are the source; this sheet does not copy them so it cannot go stale.
`python3 -c "import json; m=[x for x in json.load(open('analysis/state/patterns.json'))['modes'] if x['name']=='unsupported_claim'][0]; print(m['definition']); print(m['boundary'])"`

## Sources
What the judge can see in each record (analysis/run_judges.py, `prepare_inputs`): the agent's
system prompt (first message), the user's messages, tool calls and tool results with tool names,
and the assistant's replies, up to and including the labeled turn. Opened policies are
`get_policy` results; `search_help_center` results are snippets, not opened policies.

## Training pairs (analysis/state/splits.json, train: 22 Pass / 6 Fail)
List them with your labels and evidence:
`python3 -c "import json; s=json.load(open('analysis/state/splits.json'))['unsupported_claim']['train']; L={(r:=json.loads(l))['trace_id']: r for l in open('analysis/state/hw5_labels/unsupported_claim.jsonl') if l.strip()}; [print('PASS' if L[t]['label']==1 else 'FAIL', L[t]['scenario_id'], t[:8], L[t]['evidence'][:100]) for t in s]"`
Read one in the review app: open the pair from the Labels list, or its record in
`analysis/state/hw5_trace_inputs.json` by trace_id. Never use a dev or test pair.
