"""Monitor one failure mode on one period or on the last N hours of traffic.

    uv run python -m monitoring.run --period before [--dry-run]
    uv run python -m monitoring.run --last-hours 24 [--dry-run]

A period (``--period``) is one complete run of the 50 monitoring scenarios in a
UTC window from ``monitoring/config.json``; it is rejected unless every
scenario id is present, each in one session, on the configured model. The
scheduled path (``--last-hours``) groups whatever traces the window holds by
session and has no completeness rule.

Conversation text is built by the Homework 5 renderer (system prompt first,
turns in time order, named tool lines), the input the frozen judge was
validated on. Only the random sample estimates the failure rate; risk groups
are judged for inspection.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

from analysis.helpers.normalization import normalize_trace
from analysis.review_app import loader
from analysis.run_judges import read_users, session_system_message, turn_messages
from monitoring.sample import DEFAULT_RISK_GROUPS, select_traces

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "monitoring" / "config.json"
OUTPUT = REPO / "monitoring" / "output"
SCENARIOS = REPO / "scenarios" / "monitoring_scenarios.jsonl"
HISTORY = REPO / "monitoring" / "history.jsonl"
CHART = REPO / "monitoring" / "prevalence.svg"


class PeriodRejected(ValueError):
    """The window does not hold exactly one complete run on the configured model."""


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def fetch_raw(start: dt.datetime, end: dt.datetime, client: Any | None = None) -> list[dict]:
    """Every trace with start <= timestamp < end, as plain dicts shaped like the
    HW3 export (Langfuse's to_timestamp is exclusive)."""
    from analysis.helpers.langfuse_io import _client

    lf = client or _client()
    summaries, page = [], 1
    while True:
        resp = lf.api.trace.list(from_timestamp=start, to_timestamp=end, page=page, limit=100)
        summaries.extend(resp.data or [])
        if len(resp.data or []) < 100:
            break
        page += 1
    full = [lf.api.trace.get(s.id) for s in summaries]
    return [json.loads(json.dumps(t.dict(by_alias=True), default=str)) for t in full]


def generation_models(trace: dict) -> set[str]:
    return {
        o.get("model")
        for o in trace.get("observations") or []
        if o.get("type") == "GENERATION" and o.get("model")
    }


def build_record(group: list[dict], users: dict) -> dict[str, Any]:
    """One conversation record from one session's traces in time order.

    The id is the final trace id, so the score lands on the last turn. The text
    is HW5's judge input (analysis/run_judges.py prepare_inputs + scale.judge_rows).
    """
    turns = [{**loader.build_turn(i + 1, t), "attrs": loader.attrs(t)} for i, t in enumerate(group)]
    messages = [session_system_message(turns, users)] + [m for t in turns for m in turn_messages(t)]
    last = group[-1]["id"]
    return {
        "id": last,
        "text": normalize_trace({"trace_id": last, "trace": messages})["text"],
        "scenario_id": loader.scenario_of(group[0]),
        "session_id": loader.session_of(group[0]),
        "trace_ids": [t["id"] for t in group],
        "tools": sorted({s["tool"] for t in turns for s in t["steps"] if s["kind"] == "tool"}),
        "turn_count": len(group),  # one trace per user message
        "models": sorted(set().union(*(generation_models(t) for t in group))),
    }


def period_records(raw: list[dict], model: str, scenario_ids: set[str]) -> list[dict]:
    """The 50 conversation records of one period, or PeriodRejected."""
    pool = [t for t in raw if loader.scenario_of(t) in scenario_ids]
    print(f"window: {len(raw)} traces, {len(pool)} on the {len(scenario_ids)} monitoring "
          f"scenarios, {len(raw) - len(pool)} other traces ignored")
    missing = scenario_ids - {loader.scenario_of(t) for t in pool}
    if missing:
        raise PeriodRejected(f"{len(missing)} scenario id(s) missing: {sorted(missing)[:5]}")
    retries = loader.scenarios_with_multiple_sessions(pool)
    if retries:
        raise PeriodRejected(f"scenario(s) with more than one session (retries): {sorted(retries)}")
    users = read_users()
    records = [build_record(g, users) for g in loader.sessions(pool).values()]
    other = {r["scenario_id"]: r["models"] for r in records if set(r["models"]) - {model}}
    if other:
        raise PeriodRejected(f"traces on a model other than {model}: {other}")
    if len(records) != len(scenario_ids):
        raise PeriodRejected(f"{len(records)} conversation records, expected {len(scenario_ids)}")
    return sorted(records, key=lambda r: r["scenario_id"])


def session_records(raw: list[dict]) -> list[dict]:
    """Scheduled path: every session in the window, no completeness rule."""
    pool = [t for t in raw if loader.session_of(t)]
    if len(pool) < len(raw):
        print(f"{len(raw) - len(pool)} trace(s) without cartwheel.session_id ignored")
    users = read_users()
    records = [build_record(g, users) for g in loader.sessions(pool).values()]
    models = collections.Counter(m for r in records for m in r["models"])
    print(f"window: {len(pool)} traces, {len(records)} conversations, models {dict(models)}")
    return sorted(records, key=lambda r: r["session_id"])


def write_output(label: str, payload: dict, records: list[dict]) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{label}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    (OUTPUT / f"{label}.records.json").write_text(json.dumps(records, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--period", help="period label from the config")
    which.add_argument("--last-hours", type=float, help="monitor the previous N hours")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="stop before the judge call")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    cfg = json.loads(args.config.read_text())

    if args.period:
        period = next((p for p in cfg["periods"] if p["label"] == args.period), None)
        if period is None:
            parser.error(f"no period {args.period!r} in {args.config}")
        label, start, end = period["label"], parse_utc(period["from"]), parse_utc(period["to"])
        scenario_ids = {json.loads(line)["id"] for line in SCENARIOS.read_text().splitlines() if line}
        raw = fetch_raw(start, end)
        records = period_records(raw, cfg["model"], scenario_ids)
    else:
        end = dt.datetime.now(dt.UTC)
        start = end - dt.timedelta(hours=args.last_hours)
        label = f"last-{args.last_hours:g}h-{end:%Y%m%dT%H%M%SZ}"
        raw = fetch_raw(start, end)
        records = session_records(raw)

    n_traces = sum(len(r["trace_ids"]) for r in records)
    base = {
        "label": label, "judge_id": cfg["judge_id"], "model": cfg["model"],
        "from": start.isoformat(), "to": end.isoformat(),
        "n_traces": n_traces, "n_conversations": len(records),
    }
    if not records:
        path = write_output(label, {**base, "n_random": 0, "n_risk": 0, "n_judged": 0}, records)
        print(f"0 eligible conversations; no judge call. Wrote {path}")
        return 0

    groups = {name: DEFAULT_RISK_GROUPS[name] for name in cfg["risk_groups"]}
    plan = select_traces(records, cfg["random_rate"], groups)
    risk_ids = {r["id"] for members in plan["risk_groups"].values() for r in members}
    plan_ids = {
        "random": [r["id"] for r in plan["random"]],
        "risk_groups": {k: [r["id"] for r in v] for k, v in plan["risk_groups"].items()},
    }
    counts = {"n_random": len(plan["random"]), "n_risk": len(risk_ids), "n_judged": len(plan["to_judge"])}
    print(f"{label}: {n_traces} traces, {len(records)} conversations, cartwheel model {cfg['model']}")
    print(f"random {counts['n_random']}; risk " +
          ", ".join(f"{k} {len(v)}" for k, v in plan["risk_groups"].items()) +
          f" (union {counts['n_risk']})")
    print(f"judge calls: {counts['n_judged']} with {cfg['judge_id']}")
    if args.dry_run:
        path = write_output(label, {**base, **counts, **plan_ids}, records)
        print(f"dry run: no judge call. Wrote {path}")
        return 0

    from monitoring.correct import corrected_mode_prevalence
    from monitoring.run_judges import judge_sample, judge_test_data
    from monitoring.write_scores import build_score_records, post_scores

    verdicts = judge_sample(cfg["judge_id"], plan["to_judge"])
    random_verdicts = {i: verdicts[i] for i in plan_ids["random"]}
    risk_verdicts = {i: verdicts[i] for i in sorted(risk_ids)}
    print(f"random flags {sum(random_verdicts.values())}/{len(random_verdicts)}; "
          f"risk flags {sum(risk_verdicts.values())}/{len(risk_verdicts)}")

    # Only the random sample estimates the rate; risk verdicts are for inspection.
    estimate = corrected_mode_prevalence(
        list(random_verdicts.values()), *judge_test_data(cfg["judge_id"])
    )
    print(f"raw {estimate['raw']}, corrected {estimate['corrected']} "
          f"[{estimate['ci_low']}, {estimate['ci_high']}], threshold {cfg['threshold']}")
    scores = build_score_records(cfg["judge_mode"], random_verdicts, risk_verdicts, estimate, label)
    # Langfuse (3.225) rejects a score with no trace, so the period's prevalence
    # score goes on one stable "monitor period" trace per label.
    scores[-1]["trace_id"] = period_trace(cfg["judge_mode"], label, {**base, **counts}, estimate)
    print(f"scores sent to Langfuse: {post_scores(scores)}")

    path = write_output(label, {**base, **counts, **plan_ids, "random_verdicts": random_verdicts,
                                "risk_verdicts": risk_verdicts, "estimate": estimate}, records)
    upsert_history({**base, **counts, **estimate})
    draw_chart(cfg["threshold"], cfg["judge_mode"])
    print(f"Wrote {path}, {HISTORY.name}, {CHART.name}")
    return 0


def period_trace(mode: str, label: str, run: dict, estimate: dict) -> str:
    """The stable trace carrying one period's prevalence score (same id on rerun)."""
    from langfuse import Langfuse, get_client

    trace_id = Langfuse.create_trace_id(seed=f"{mode}|period|{label}")
    span = get_client().start_span(
        trace_context={"trace_id": trace_id}, name=f"monitor.{mode}",
        input=run, output=estimate, metadata={"monitor_label": label},
    )
    span.update_trace(name=f"monitor {mode} {label}", tags=["monitor"])
    span.end()
    return trace_id


def upsert_history(line: dict) -> None:
    """One line per label; a rerun of a period replaces its line."""
    rows = [json.loads(x) for x in HISTORY.read_text().splitlines()] if HISTORY.exists() else []
    rows = [r for r in rows if r["label"] != line["label"]] + [line]
    HISTORY.write_text("".join(json.dumps(r) + "\n" for r in rows))


def draw_chart(threshold: float, mode: str) -> None:
    from monitoring.chart import prevalence_chart

    rows = [json.loads(x) for x in HISTORY.read_text().splitlines()]
    CHART.write_text(prevalence_chart(rows, threshold, mode))


if __name__ == "__main__":
    raise SystemExit(main())
