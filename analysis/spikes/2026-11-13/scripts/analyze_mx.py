"""Metrics for the model x variant matrix on dev, with catches by rule family."""
import glob
import json
import math

T = "analysis/spikes/2026-11-13/results"
W = "."
labs = {}
for line in open(f"{W}/analysis/state/hw5_labels/unsupported_claim.jsonl"):
    r = json.loads(line)
    labs[r["trace_id"]] = r

# rule family of each dev Fail, from its label evidence (read 2026-11-13)
FAM = {"0040": "human", "0048": "human", "0084": "human", "0249": "human", "0011": "process", "0052": "process",
       "0129": "process", "0231": "process", "0005": "process", "0032": "process", "0215": "process", "0243": "process",
       "0157": "date", "0178": "date", "0195": "date", "0159": "date", "0053": "date", "0078": "date",
       "0083": "record", "0123": "record", "0188": "record", "0242": "record", "0240": "record", "0046": "record",
       "0087": "other"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return f"[{(c - h) / d:.2f},{(c + h) / d:.2f}]"


print(f"{'model':12} {'prompt':6} {'TPR':>12} {'TNR':>12}   J      TPR CI      TNR CI     Fail catches by family")
for f in sorted(glob.glob(f"{T}/mx-*.json")):
    d = json.load(open(f))
    rows = d["rows"]
    P = [r for r in rows if r["label"] == 1]
    F = [r for r in rows if r["label"] == 0]
    tp = sum(r["pred"] == 1 for r in P)
    tn = sum(r["pred"] == 0 for r in F)
    fam = {}
    for r in F:
        k = FAM.get(labs[r["scenario"]]["scenario_id"][-4:], "?")
        a, b = fam.get(k, (0, 0))
        fam[k] = (a + (r["pred"] == 0), b + 1)
    name = f.rsplit("mx-", 1)[1][:-5]
    model, prompt = name.rsplit("-", 1)
    fams = " ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(fam.items()))
    print(f"{model:12} {prompt:6} {tp:>3}/{len(P)}={tp/len(P):.2f} {tn:>3}/{len(F)}={tn/len(F):.2f}  {tp/len(P)+tn/len(F)-1:+.2f}  "
          f"{wilson(tp, len(P))} {wilson(tn, len(F))}  {fams}")
