from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from harbor_adapter.analysis import analyze_capability_job
from harbor_adapter.export import _provider_key, export_tasks
from harbor_adapter.summary import summarize_job


def _write_cases(path: Path, cases: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(case) for case in cases) + "\n")


def test_provider_key_follows_the_selected_model() -> None:
    assert _provider_key("gpt-example") == "OPENAI_API_KEY"
    assert _provider_key("o4-mini") == "OPENAI_API_KEY"
    assert _provider_key("anthropic/example") == "ANTHROPIC_API_KEY"
    assert _provider_key("openrouter/vendor-model") == "OPENROUTER_API_KEY"


def test_export_preserves_cartwheel_formats_and_builds_harbor_task(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    judges = state / "judges"
    judges.mkdir(parents=True)
    (judges / "response-quality-v1.json").write_text(
        json.dumps(
            {
                "judge_id": "response-quality-v1",
                "mode": "response-quality",
                "version": 1,
                "prompt_text": "The reply gives the information needed for the next step.",
                "model": "provider/frozen-judge-model",
                "status": "frozen",
            }
        )
    )
    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(state))
    cases_path = tmp_path / "cases.jsonl"
    case = {
        "id": "e-101",
        "mode": "response-quality",
        "kind": "capability",
        "baseline_pass_rate": 0.6,
        "input": {
            "role": "shopper",
            "user_id": 1,
            "message": "Can I return this order?",
        },
        "initial_state": {"world": "reseed", "fixture": None},
        "expected": {
            "assertions": ["The reply explains the next step."],
            "checks": [{"check": "tool_called", "name": "get_order"}],
            "judges": {"response-quality": "pass"},
        },
    }
    _write_cases(cases_path, [case])

    output = tmp_path / "tasks"
    tasks = export_tasks(cases_path, output, require_suite=False)

    assert tasks == [output / "e-101"]
    exported = json.loads(
        (output / "e-101" / "environment" / "cartwheel" / "case.json").read_text()
    )
    assert exported == case
    task = tomllib.loads((output / "e-101" / "task.toml").read_text())
    assert task["metadata"] == {
        "case_id": "e-101",
        "kind": "capability",
        "mode": "response-quality",
    }
    rubric = (output / "e-101" / "tests" / "judge_response-quality.py").read_text()
    compile(rubric, "judge_response-quality.py", "exec")
    assert "The reply gives the information needed for the next step." in rubric
    assert "MODEL = 'provider/frozen-judge-model'" in rubric
    assert "JUDGE_ID = 'response-quality-v1'" in rubric
    assert "COMPLETION_KWARGS = {'temperature': 0}" in rubric
    assert 'sys.path.insert(0, "/app")' in rubric
    assert "--- Trace to evaluate ---" in rubric
    assert '"critique": "string", "result": "string"' in rubric
    assert "judge_trace_text" in rubric

    # The generated judge keeps its evidence in the verifier log dir (no model call needed).
    import sys
    import types

    log_dir = tmp_path / "verifier-logs"
    log_dir.mkdir()
    monkeypatch.setenv("CARTWHEEL_VERIFIER_LOG_DIR", str(log_dir))
    monkeypatch.setitem(sys.modules, "rewardkit", types.SimpleNamespace(criterion=lambda f: f))
    namespace: dict = {"__name__": "generated_judge"}
    exec(compile(rubric, "judge_response-quality.py", "exec"), namespace)
    namespace["_record"]("user: hi", "fail", "Q1: yes")
    record = json.loads((log_dir / "judge_response-quality.json").read_text())
    assert record == {
        "judge_id": "response-quality-v1",
        "model": "provider/frozen-judge-model",
        "expected": "pass",
        "verdict": "fail",
        "critique": "Q1: yes",
        "input": "user: hi",
    }
    assert (output / "e-101" / "tests" / "test.sh").stat().st_mode & 0o111
    assert "harbor-rewardkit==0.2.1" in (
        output / "e-101" / "tests" / "test.sh"
    ).read_text()
    assert "docetl==0.3.0" in (
        output / "e-101" / "tests" / "test.sh"
    ).read_text()
    assert not (output / "e-101" / "environment" / "cartwheel" / ".env").exists()


def test_baseline_export_accepts_only_unclassified_cases(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    case = {
        "id": "e-102",
        "mode": "response_quality",
        "input": {"role": "shopper", "user_id": 1, "message": "hello"},
        "initial_state": {"world": "reseed", "fixture": None},
        "expected": {
            "assertions": ["The agent asks what the shopper needs."],
            "checks": [{"check": "reply_asks_question"}],
        },
    }
    _write_cases(cases_path, [case])
    output = tmp_path / "tasks"

    export_tasks(cases_path, output, baseline=True, case_ids={"e-102"})
    task = tomllib.loads((output / "e-102" / "task.toml").read_text())
    assert task["metadata"]["kind"] == "unclassified"

    with pytest.raises(ValueError, match="kind must be regression or capability"):
        export_tasks(cases_path, output, require_suite=False)


def test_summary_blocks_regressions_but_reports_capabilities(
    tmp_path: Path, monkeypatch
) -> None:
    import tests.eval.passk as passk

    def fake_case_passes(kind: str, passes: int, k: int, baseline_pass_rate=None):
        decision = "block" if kind == "regression" and passes < k else "pass"
        return {"decision": decision, "reason": "test"}

    monkeypatch.setattr(passk, "case_passes", fake_case_passes)
    monkeypatch.setattr(passk, "pass_at_k", lambda n, c, k: c / n)
    monkeypatch.setattr(passk, "pass_hat_k", lambda n, c, k: c / n)
    cases_path = tmp_path / "cases.jsonl"
    base = {
        "mode": "response_quality",
        "input": {"role": "shopper", "user_id": 1, "message": "hello"},
        "initial_state": {"world": "reseed", "fixture": None},
        "expected": {
            "assertions": ["The reply asks a question."],
            "checks": [{"check": "reply_asks_question"}],
        },
    }
    regression = {**base, "id": "e-201", "kind": "regression"}
    capability = {
        **base,
        "id": "e-202",
        "kind": "capability",
        "baseline_pass_rate": 0.4,
    }
    _write_cases(cases_path, [regression, capability])

    def trial(case_id: str, reward: float) -> dict:
        return {
            "task_name": f"cartwheel/evals__{case_id}",
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": None,
        }

    job = tmp_path / "job"
    job.mkdir()
    trials = [trial("e-201", value) for value in [1, 1, 1, 1, 0]]
    trials += [trial("e-202", value) for value in [0, 0, 1, 0, 1]]
    (job / "result.json").write_text(json.dumps({"trial_results": trials}))

    markdown, passed = summarize_job(
        job, cases_path=cases_path, expected_attempts=5
    )

    assert passed is False
    assert "| `e-201` | regression | 4 | 5 | 0.800" in markdown
    assert "| `e-202` | capability | 2 | 5 | 0.400" in markdown


def test_baseline_summary_reports_the_observed_classification(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    base = {
        "mode": "response_quality",
        "input": {"role": "shopper", "user_id": 1, "message": "hello"},
        "initial_state": {"world": "reseed", "fixture": None},
        "expected": {"checks": [{"check": "reply_asks_question"}]},
    }
    first = {**base, "id": "e-301"}
    second = {**base, "id": "e-302"}
    _write_cases(cases_path, [first, second])

    def trial(case_id: str, reward: float, attempt: int) -> dict:
        return {
            "task_name": f"cartwheel/evals__{case_id}",
            "trial_name": f"trial-{attempt}",
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": None,
        }

    job = tmp_path / "job"
    job.mkdir()
    trials = [trial("e-301", 1, i) for i in range(5)]
    trials += [trial("e-302", value, i) for i, value in enumerate([1, 1, 1, 0, 0])]
    (job / "result.json").write_text(json.dumps({"trial_results": trials}))

    markdown, passed = summarize_job(
        job,
        cases_path=cases_path,
        expected_attempts=5,
        classify=True,
    )

    assert passed is True
    assert '`kind: "regression"`' in markdown
    assert '`kind: "capability"`, `baseline_pass_rate: 0.6`' in markdown


def test_baseline_summary_does_not_classify_infrastructure_errors(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    case = {
        "id": "e-303",
        "mode": "response_quality",
        "input": {"role": "shopper", "user_id": 1, "message": "hello"},
        "initial_state": {"world": "reseed", "fixture": None},
        "expected": {"checks": [{"check": "reply_asks_question"}]},
    }
    _write_cases(cases_path, [case])
    job = tmp_path / "job"
    job.mkdir()
    trials = [
        {
            "task_name": "cartwheel/evals__e-303",
            "verifier_result": {"rewards": {"reward": 1}},
            "exception_info": None,
        }
        for _ in range(4)
    ]
    trials.append(
        {
            "task_name": "cartwheel/evals__e-303",
            "verifier_result": None,
            "exception_info": {"message": "sandbox failed"},
        }
    )
    (job / "result.json").write_text(json.dumps({"trial_results": trials}))

    markdown, passed = summarize_job(
        job,
        cases_path=cases_path,
        expected_attempts=5,
        classify=True,
    )

    assert passed is False
    assert "do not classify; complete five valid trials" in markdown


def test_final_export_requires_the_homework_suite(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    base = {
        "mode": "first_mode",
        "kind": "regression",
        "input": {"role": "shopper", "user_id": 1, "message": "hello"},
        "initial_state": {"world": "reseed", "fixture": None},
        "expected": {
            "assertions": ["The reply asks a question."],
            "checks": [{"check": "reply_asks_question"}],
        },
    }
    cases = [{**base, "id": f"e-{index:03d}"} for index in range(10)]
    _write_cases(cases_path, cases)

    with pytest.raises(ValueError, match="at least two failure modes"):
        export_tasks(cases_path, tmp_path / "tasks")

    cases[-1] = {
        **cases[-1],
        "mode": "second_mode",
        "kind": "capability",
        "baseline_pass_rate": 0.8,
    }
    _write_cases(cases_path, cases)

    tasks = export_tasks(cases_path, tmp_path / "tasks")

    assert len(tasks) == 10


def test_capability_analysis_uses_5_10_and_15_observed_runs(
    tmp_path: Path, monkeypatch
) -> None:
    import tests.eval.passk as passk

    monkeypatch.setattr(passk, "pass_at_k", lambda n, c, k: (n + c + k) / 100)
    monkeypatch.setattr(
        "harbor_adapter.analysis.pass_at_k",
        lambda n, c, k: (n + c + k) / 100,
    )
    job = tmp_path / "job"
    job.mkdir()
    trials = []
    for attempt in range(14, -1, -1):
        trials.append(
            {
                "task_name": "cartwheel/evals__e-401",
                "trial_name": f"trial-{attempt:02d}",
                "verifier_result": {
                    "rewards": {"reward": 1 if attempt % 2 == 0 else 0}
                },
                "agent_info": {
                    "model_info": {
                        "provider": "student",
                        "name": "provider-model",
                    }
                },
                "exception_info": None,
            }
        )
    (job / "result.json").write_text(json.dumps({"trial_results": trials}))

    result = analyze_capability_job(job, "e-401")

    assert result["n"] == 15
    assert result["rewards"][:3] == [1, 0, 1]
    assert result["model"] == "student/provider-model"
    assert result["trials"][0]["trial_name"] == "trial-14"
    assert result["trial_order"] == "result.json trial_results order"
    assert [row["n"] for row in result["comparisons"]] == [5, 10, 15]
    assert set(result["comparisons"][-1]["pass_at_k"]) == {
        "1",
        "3",
        "5",
        "10",
        "15",
    }


def _write_judges(state: Path, statuses: dict[int, str]) -> None:
    judges = state / "judges"
    judges.mkdir(parents=True)
    for version, status in statuses.items():
        (judges / f"quality-v{version}.json").write_text(
            json.dumps(
                {
                    "judge_id": f"quality-v{version}",
                    "mode": "quality",
                    "version": version,
                    "prompt_text": f"prompt v{version}",
                    "model": f"model-v{version}",
                    "status": status,
                }
            )
        )


def _point_loader_at(tmp_path: Path, monkeypatch, pins: dict | None) -> None:
    from replay import rollout

    monkeypatch.setenv("CARTWHEEL_ANALYSIS_STATE", str(tmp_path / "state"))
    pins_path = tmp_path / "judges.json"
    if pins is not None:
        pins_path.write_text(json.dumps(pins))
    monkeypatch.setattr(rollout, "JUDGE_PINS_PATH", pins_path)


def test_two_frozen_judges_without_a_pin_fail_loudly(tmp_path: Path, monkeypatch) -> None:
    from replay.rollout import load_frozen_judge

    _write_judges(tmp_path / "state", {1: "frozen", 2: "frozen"})
    _point_loader_at(tmp_path, monkeypatch, pins=None)
    with pytest.raises(ValueError, match="quality-v1, quality-v2"):
        load_frozen_judge("quality")


def test_pin_selects_the_accepted_frozen_judge(tmp_path: Path, monkeypatch) -> None:
    from replay.rollout import load_frozen_judge

    _write_judges(tmp_path / "state", {1: "frozen", 2: "frozen"})
    _point_loader_at(tmp_path, monkeypatch, pins={"quality": "quality-v1"})
    assert load_frozen_judge("quality")["model"] == "model-v1"


def test_pin_to_a_judge_that_is_not_frozen_is_rejected(tmp_path: Path, monkeypatch) -> None:
    from replay.rollout import load_frozen_judge

    _write_judges(tmp_path / "state", {1: "draft", 2: "frozen"})
    _point_loader_at(tmp_path, monkeypatch, pins={"quality": "quality-v1"})
    with pytest.raises(ValueError, match="not a frozen judge"):
        load_frozen_judge("quality")


def test_single_frozen_judge_needs_no_pin(tmp_path: Path, monkeypatch) -> None:
    from replay.rollout import load_frozen_judge

    _write_judges(tmp_path / "state", {1: "draft", 2: "frozen"})
    _point_loader_at(tmp_path, monkeypatch, pins=None)
    assert load_frozen_judge("quality")["judge_id"] == "quality-v2"


def _harbor_023_job(job: Path, trials: list[tuple[str, str, float]]) -> None:
    """The layout Harbor 0.23.0 writes: a job result.json without trial_results,
    and one <trial>/result.json per trial (observed in .harbor/jobs, 2026-11-14)."""
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"id": "j", "n_total_trials": len(trials), "stats": {}}))
    for trial_name, started_at, reward in trials:
        (job / trial_name).mkdir()
        (job / trial_name / "result.json").write_text(
            json.dumps(
                {
                    "task_name": f"cartwheel/evals__{trial_name.split('__')[0]}",
                    "trial_name": trial_name,
                    "started_at": started_at,
                    "verifier_result": {"rewards": {"reward": reward}},
                    "exception_info": None,
                    "agent_info": {"model_info": {"name": "m", "provider": None}},
                }
            )
        )


def test_baseline_summary_reads_harbor_023_per_trial_results(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(
        cases_path,
        [
            {
                "id": "e-301",
                "mode": "m",
                "input": {"role": "shopper", "user_id": 1, "message": "x"},
                "initial_state": {"world": "reseed", "fixture": None, "assumes": "x"},
                "expected": {"assertions": ["x"], "checks": [{"check": "reply_asks_question"}]},
            }
        ],
    )
    job = tmp_path / "job"
    _harbor_023_job(
        job, [(f"e-301__t{i}", f"2026-11-14T15:48:1{i}Z", r) for i, r in enumerate([1, 0, 1, 1, 0])]
    )

    markdown, passed = summarize_job(job, cases_path=cases_path, expected_attempts=5, classify=True)

    assert passed
    assert "| `e-301` | 3 | 5 |" in markdown
    assert "baseline_pass_rate: 0.6" in markdown


def test_capability_analysis_orders_harbor_023_trials_by_start_time(tmp_path: Path, monkeypatch) -> None:
    import tests.eval.passk as passk

    monkeypatch.setattr(passk, "pass_at_k", lambda n, c, k: 0.0)
    import harbor_adapter.analysis as analysis

    monkeypatch.setattr(analysis, "pass_at_k", lambda n, c, k: 0.0)
    job = tmp_path / "job"
    # Directory names sort differently from start times on purpose.
    starts = [f"2026-11-14T15:{m:02d}:00Z" for m in range(15)]
    names = [f"e-401__{chr(ord('z') - i)}" for i in range(15)]
    rewards = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    _harbor_023_job(job, list(zip(names, starts, rewards)))

    result = analyze_capability_job(job, "e-401")

    assert [t["trial_name"] for t in result["trials"]] == names
    assert result["rewards"] == rewards
    assert "started_at" in result["trial_order"]
