"""combine.py <prefix> : OR-combine detector runs <prefix>-d*.json; print per-detector and combined metrics."""
import glob, json, sys
runs = {f.rsplit("-", 2)[-2] + "-" + f.rsplit("-", 1)[-1][:-5]: {r["scenario"]: r for r in json.load(open(f))["rows"]}
        for f in sorted(glob.glob(sys.argv[1] + "-d*.json"))}
keys = next(iter(runs.values())).keys()
P = [k for k in keys if next(iter(runs.values()))[k]["label"] == 1]
F = [k for k in keys if next(iter(runs.values()))[k]["label"] == 0]
for name, rows in runs.items():
    print(f"{name:12} false alarms on Pass {sum(rows[k]['pred'] == 0 for k in P)}/{len(P)}   catches on Fail {sum(rows[k]['pred'] == 0 for k in F)}/{len(F)}")
comb = {k: 0 if any(rows[k]["pred"] == 0 for rows in runs.values()) else 1 for k in keys}
tp = sum(comb[k] == 1 for k in P); tn = sum(comb[k] == 0 for k in F)
print(f"COMBINED (OR)  TPR {tp}/{len(P)}={tp/len(P):.2f}  TNR {tn}/{len(F)}={tn/len(F):.2f}  J {tp/len(P)+tn/len(F)-1:+.2f}")
print("missed Fails:", [k for k in F if comb[k] == 1], " Pass judged Fail:", [k for k in P if comb[k] == 0])
