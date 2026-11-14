"""A0: does the frozen HW5 judge give the same verdicts on Harbor's judge input?

Runs the judge file that harbor_adapter.export generates (exact code Harbor's
verifier runs) on the HW5 test traces, re-rendered into the runtime transcript
shape that replay.rollout.judge_trace_text formats. Compares with the judge's
HW5 test predictions (HW5 input: system prompt first, named tool lines,
temperature 0) and with the HW5 labels.

Usage (repo root):  uv run python analysis/spikes/2026-11-14-hw6-judge-input/run.py [N]
N limits the number of traces (smoke test); omit for all 57.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from harbor_adapter.export import export_tasks  # noqa: E402
from replay.rollout import judge_trace_text, load_frozen_judge  # noqa: E402

OUT = Path(__file__).resolve().parent
STATE = REPO / "analysis" / "state"
MODE = "unsupported_claim"


def _maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def to_transcript(messages: list[dict]) -> dict:
    """HW5 normalized messages -> runtime transcript (turns of user, tool_calls, reply).

    Mirrors replay.rollout._extract_turn: one turn per user message; assistant
    text of a turn joined into one reply; tool calls in order with their result.
    The system message is dropped, as the runtime transcript has none.
    """
    turns: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            turns.append({"user": m.get("text", ""), "tool_calls": [], "parts": []})
            continue
        if not turns:
            raise ValueError("message before the first user message")
        turn = turns[-1]
        if role == "assistant":
            turn["parts"].append(m.get("text", ""))
        elif role == "tool_call":
            turn["tool_calls"].append(
                {"name": m["name"], "args": _maybe_json(m.get("arguments")), "result": None}
            )
        elif role == "tool_result":
            pending = [c for c in turn["tool_calls"] if c["name"] == m["name"] and c["result"] is None]
            if not pending:
                raise ValueError(f"tool_result {m['name']} without a call")
            pending[0]["result"] = _maybe_json(m.get("content"))
        else:
            raise ValueError(f"unknown role {role}")
    return {
        "turns": [
            {"user": t["user"], "reply": "\n".join(p for p in t["parts"] if p), "tool_calls": t["tool_calls"]}
            for t in turns
        ]
    }


def load_generated_judge(tmp: Path) -> dict:
    """Export one throwaway case and exec its generated judge file."""
    case = {
        "id": "a0-judge",
        "mode": MODE,
        "input": {"role": "shopper", "user_id": 1, "message": "x"},
        "initial_state": {"world": "reseed", "fixture": None, "assumes": "A0 harness only."},
        "expected": {"assertions": ["A0 harness only."], "checks": [], "judges": {MODE: "pass"}},
    }
    cases = tmp / "cases.jsonl"
    cases.write_text(json.dumps(case) + "\n")
    export_tasks(cases, tmp / "tasks", baseline=True, require_suite=False)
    source = (tmp / "tasks" / "a0-judge" / "tests" / f"judge_{MODE}.py").read_text()
    sys.modules["rewardkit"] = types.SimpleNamespace(criterion=lambda f: f)
    ns: dict = {"__name__": "generated_judge"}
    exec(compile(source, "judge_generated.py", "exec"), ns)
    return ns


def latest_labels() -> dict[str, int]:
    latest: dict[str, dict] = {}
    for line in (STATE / "hw5_labels" / f"{MODE}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["trace_id"] not in latest or r["ts"] >= latest[r["trace_id"]]["ts"]:
                latest[r["trace_id"]] = r
    return {k: v["label"] for k, v in latest.items()}  # 1 = Pass


ARMS = {
    # name: (input, temperature 0?)
    "harbor": ("harbor", False),  # exactly as Harbor's verifier runs it
    "harbor_t0": ("harbor", True),  # A0b arm A: Harbor input, temperature 0
    "hw5_t0": ("hw5", True),  # A0b arm B: HW5 frozen input (scale.judge_rows), temperature 0
    # A0c: the fixed adapter as Harbor now runs it (ordered events, system prompt, named
    # tools, temperature from the generated file). Needs the runtime transcript shape.
    "fixed": ("events", False),
}


def to_runtime_transcript(messages: list[dict]) -> dict:
    """HW5 normalized messages -> the fixed runtime transcript (system_prompt + turn events)."""
    transcript: dict = {"turns": []}
    for m in messages:
        if m["role"] == "system":
            transcript["system_prompt"] = m["text"]
        elif m["role"] == "user":
            transcript["turns"].append({"user": m["text"], "events": []})
        else:
            transcript["turns"][-1]["events"].append(m)
    return transcript


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--arm=")]
    arm = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--arm=")), "harbor")
    input_kind, temp0 = ARMS[arm]
    limit = int(args[0]) if args else None
    judge = load_frozen_judge(MODE)
    hw5_pred = judge["predictions"][judge["prompt_hash"]]  # 1 = Pass
    labels = latest_labels()
    test_ids = json.loads((STATE / "splits.json").read_text())[MODE]["test"]
    traces = {r["trace_id"]: r["trace"] for r in json.loads((STATE / "hw5_trace_inputs.json").read_text())}
    ids = test_ids[:limit] if limit else test_ids

    with tempfile.TemporaryDirectory(prefix="a0-") as d:
        ns = load_generated_judge(Path(d))
        print(f"arm {arm} judge {ns['JUDGE_ID']} model {ns['MODEL']} traces {len(ids)}")
        decode = ns["_decode"]
        if temp0:  # same setting as analysis/helpers/scale.py for non-reasoning models
            map_op = ns["MapOp"]
            ns["MapOp"] = lambda **kw: map_op(**kw, litellm_completion_kwargs={"temperature": 0})
        if input_kind == "hw5":
            from analysis.helpers.scale import judge_rows

            hw5_content = {r["trace_id"]: r["content"] for r in judge_rows(list(ids))}

        def one(tid: str) -> dict:
            rows: list[dict] = []
            ws = Path(tempfile.mkdtemp(prefix="ws-", dir=d))
            convert = to_runtime_transcript if input_kind == "events" else to_transcript
            transcript = convert(traces[tid])
            (ws / "cartwheel-result.json").write_text(json.dumps({"transcript": transcript}))
            ns["_decode"] = lambda row: (rows.append(row), decode(row))[1]
            if input_kind == "hw5":
                ns["judge_trace_text"] = lambda _transcript: hw5_content[tid]
            passed = ns["cartwheel_judge"](ws)  # EXPECTED = "pass": True means verdict Pass
            return {
                "trace_id": tid,
                "label": labels.get(tid),
                "hw5_verdict": hw5_pred.get(tid),
                "harbor_verdict": int(bool(passed)),
                "critique": rows[-1].get("critique") if rows else None,
                "content_head": (hw5_content[tid] if input_kind == "hw5" else judge_trace_text(transcript))[:400],
            }

        # Sequential: _decode is swapped per call to capture the critique.
        results = [one(t) for t in ids]

    name = f"{'smoke' if limit else 'results'}{'' if arm == 'harbor' else '-' + arm}.json"
    (OUT / name).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    def rates(key: str) -> str:
        tp = sum(1 for r in results if r["label"] == 0 and r[key] == 0)
        fn = sum(1 for r in results if r["label"] == 0 and r[key] == 1)
        tn = sum(1 for r in results if r["label"] == 1 and r[key] == 1)
        fp = sum(1 for r in results if r["label"] == 1 and r[key] == 0)
        tpr = tp / (tp + fn) if tp + fn else float("nan")
        tnr = tn / (tn + fp) if tn + fp else float("nan")
        return f"TPR(Fail caught) {tp}/{tp + fn} = {tpr:.2f}  TNR(Pass kept) {tn}/{tn + fp} = {tnr:.2f}"

    flips = [r for r in results if r["hw5_verdict"] != r["harbor_verdict"]]
    print(f"HW5 input   : {rates('hw5_verdict')}")
    print(f"Harbor input: {rates('harbor_verdict')}")
    print(f"verdict flips vs HW5: {len(flips)}/{len(results)}")
    for r in flips:
        print(f"  {r['trace_id']} label={r['label']} hw5={r['hw5_verdict']} harbor={r['harbor_verdict']}")
    print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
