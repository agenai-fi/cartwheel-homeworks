"""Re-run a case's generated judge on the saved transcripts of a finished Harbor job,
storing each verdict and critique in <trial>/verifier/judge_<mode>.json (as the verifier
now does live). For trials that ran before the judge kept its evidence.

Usage (repo root): uv run python analysis/spikes/2026-11-14-hw6-judge-input/rejudge.py <job_dir> <case_id>
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from harbor_adapter.export import export_tasks  # noqa: E402
from harbor_adapter.summary import load_trial_results  # noqa: E402


def main() -> None:
    job_dir, case_id = Path(sys.argv[1]), sys.argv[2]
    cases = [json.loads(line) for line in (REPO / "eval_cases" / "cases.jsonl").read_text().splitlines() if line.strip()]
    case = next(c for c in cases if c["id"] == case_id)
    with tempfile.TemporaryDirectory(prefix="rejudge-") as d:
        cases_path = Path(d) / "cases.jsonl"
        cases_path.write_text(json.dumps(case) + "\n")
        export_tasks(cases_path, Path(d) / "tasks", require_suite=False)
        sys.modules["rewardkit"] = types.SimpleNamespace(criterion=lambda f: f)
        trials, _ = load_trial_results(job_dir)
        for mode in case["expected"].get("judges", {}):
            source = (Path(d) / "tasks" / case_id / "tests" / f"judge_{mode}.py").read_text()
            ns: dict = {"__name__": "generated_judge"}
            exec(compile(source, f"judge_{mode}.py", "exec"), ns)
            for trial in trials:
                trial_dir = job_dir / trial["trial_name"]
                details = json.loads((trial_dir / "verifier" / "reward-details.json").read_text())
                original = next(
                    c["detail"]["score"] for c in details["reward"]["components"] if c["name"] == f"judge_{mode}"
                )
                os.environ["CARTWHEEL_VERIFIER_LOG_DIR"] = str(trial_dir / "verifier")
                passed = ns["cartwheel_judge"](trial_dir / "agent")
                same = "same" if float(passed) == original else "DIFFERENT"
                print(f"RESULT {trial['trial_name']} {mode}: now={'pass' if passed else 'not pass'} "
                      f"original={'pass' if original else 'not pass'} ({same})")


if __name__ == "__main__":
    main()
