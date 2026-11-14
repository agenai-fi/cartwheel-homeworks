"""Direct LiteLLM judge spike (no DocETL system prompt).

usage: spike_direct.py <prompt> <model> <sys:0|1> <out.json> <ids...>
Run from the cartwheel worktree root with .env loaded.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm

sys.path.insert(0, str(Path.cwd()))
from analysis.helpers import scale  # noqa: E402

prompt_path, model, add_sys, out_path, picks = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4], sys.argv[5:]
prompt = Path(prompt_path).read_text()
store = {t["trace_id"]: t for t in scale.load_store_traces()}
recs = json.load(open("analysis/state/hw5_trace_inputs.json"))
recs = recs if isinstance(recs, list) else recs["records"]
prepared = {r["trace_id"]: r for r in recs}
train = json.load(open("analysis/state/splits.json"))["unsupported_claim"]["train"]
labs = {}
for line in open("analysis/state/hw5_labels/unsupported_claim.jsonl"):
    r = json.loads(line)
    labs[r["trace_id"]] = r
by_sid = {labs[t]["scenario_id"][-4:]: t for t in train}
if picks == ["DEV"]:
    dev = json.load(open("analysis/state/splits.json"))["unsupported_claim"]["dev"]
    by_sid = {t: t for t in dev}
    picks = list(dev)

SUFFIX = ("First write a critique of the trace against the criterion. "
          "Use specific evidence from the provided trace. Then return result "
          "as exactly Pass when the named failure is absent, or Fail when present.")
SCHEMA = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["critique", "result"],
    "properties": {"critique": {"type": "string"}, "result": {"type": "string", "enum": ["Pass", "Fail"]}}}}}


def judge_text(tid):
    text = scale.judge_rows([tid])[0]["content"]
    if add_sys:
        msgs = prepared[tid]["trace"]
        msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
        sys_msg = next(m for m in msgs if m["role"] == "system")
        text = "system: " + (sys_msg.get("text") or sys_msg.get("content")) + "\n" + text
    return text


def call(sid):
    tid = by_sid[sid]
    content = f"{prompt}\n\n--- Trace to evaluate ---\n{judge_text(tid)}\n\n{SUFFIX}"
    kw = {"temperature": 0} if not model.startswith(("o", "gpt-5")) else {}
    resp = litellm.completion(model=model, messages=[{"role": "user", "content": content}],
                              response_format=SCHEMA, **kw)
    out = json.loads(resp.choices[0].message.content)
    return {"scenario": sid, "label": int(labs[tid]["label"]), "pred": 1 if out["result"] == "Pass" else 0,
            "critique": out["critique"]}


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(call, picks))
Path(out_path).write_text(json.dumps({"prompt": prompt_path, "model": model, "sys": add_sys, "rows": rows}, indent=1))
p = [r for r in rows if r["label"] == 1]
f = [r for r in rows if r["label"] == 0]
print(f"{Path(prompt_path).stem} {model} sys={int(add_sys)}  Pass ok {sum(r['pred'] == 1 for r in p)}/{len(p)}  Fail ok {sum(r['pred'] == 0 for r in f)}/{len(f)}  ",
      " ".join(f"{r['scenario']}:{'ok' if r['label'] == r['pred'] else 'XX'}" for r in rows))
