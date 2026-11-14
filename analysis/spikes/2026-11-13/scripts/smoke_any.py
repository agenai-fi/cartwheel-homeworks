"""Unregistered smoke run: smoke_any.py <prompt> <out.json> <ids...>. Run from the cartwheel worktree root."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from analysis.helpers import scale  # noqa: E402
prompt_path, out_path, picks = sys.argv[1], sys.argv[2], sys.argv[3:]
train = json.load(open("analysis/state/splits.json"))["unsupported_claim"]["train"]
labs = {}
for line in open("analysis/state/hw5_labels/unsupported_claim.jsonl"):
    r = json.loads(line); labs[r["trace_id"]] = r
by_sid = {labs[t]["scenario_id"][-4:]: t for t in train}
if picks == ["DEV"]:
    picks = json.load(open("analysis/state/splits.json"))["unsupported_claim"]["dev"]
    by_sid = {x: x for x in picks}
ids = [by_sid[s] for s in picks]
t0 = time.time()
out = scale.classify_store(Path(prompt_path).read_text(), "gpt-4o-mini", ids)
rows = [{"scenario": s, "label": int(labs[t]["label"]), "pred": out[t], "critique": out.critiques[t]} for s, t in zip(picks, ids)]
Path(out_path).write_text(json.dumps({"prompt": prompt_path, "seconds": round(time.time() - t0, 1), "rows": rows}, indent=1))
print(Path(prompt_path).name, " ".join(f"{r['scenario']}:{'ok' if r['label'] == r['pred'] else 'XX'}" for r in rows))
