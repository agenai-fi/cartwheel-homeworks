"""The one session view for HW4: grouping, ordering, ground truth, provenance.

Why this exists rather than reusing `analysis/helpers/normalization.py`: that
module serves clustering and Module 3, and three of its choices are wrong for
human review.

* It groups on `scenario_id`. Verified on `support-0144`, whose stalled first
  attempt and retry are two sessions: scenario grouping splices them into one
  three-turn conversation in which the shopper repeats himself. That
  conversation never happened.
* `_merge_multi_turn` keeps one `trace_id` per merged record and discards the
  rest. HW4 Part E writes one Langfuse score per trace, so the discarded ids
  are unscoreable.
* It recomputes `turn_count` and `tool_call_count` on merge but leaves
  `distinct_tools`, `has_retrieval` and `tokens` at the first turn's values.
  Measured on `support-0075`: reports `distinct_tools` 1 where four tools ran.

Three observed facts this module is built around, each measured over the live
project rather than assumed:

* `observations[]` is NOT chronological -- 10 of 11 inspected traces came back
  out of order. Sorting by `startTime` is a correctness requirement: unsorted, a
  tool result renders before the call that produced it.
* Langfuse's native `sessionId` is null on every trace. Session identity lives
  only in `metadata.attributes["cartwheel.session_id"]`.
Observation-level attributes are stringified by OpenTelemetry, so a boolean
arrives as `"false"`, a truthy string. This module does not read them -- the
tool result already carries `ok`/`error` -- but anything that starts to must
compare to `"true"` rather than testing truthiness.
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from seed.eligibility import effective_return_window_days, is_refund_eligible

REPO = Path(__file__).resolve().parent.parent.parent
EXPORT = REPO / "traces" / "support_traces.json"
POLICY_DIR = REPO / "data" / "policies"
FACTS = REPO / "facts.yaml"


def _platform_window_days() -> int:
    """`return_window_days` from facts.yaml, the stated source of truth.

    Read with a regex rather than `seed.generate.load_facts` so this module stays
    importable without pyyaml -- the CLI front end runs on bare python3. A
    hardcoded 30 was the same defect as the hardcoded WORLD_ASOF this module
    removed: it agrees with facts.yaml today and would diverge in silence.
    """
    m = re.search(r"^return_window_days:\s*(\d+)", FACTS.read_text(), re.M)
    if not m:
        raise RuntimeError(f"return_window_days not found in {FACTS}")
    return int(m.group(1))


PLATFORM_WINDOW_DAYS = _platform_window_days()

# The 18 real policy ids, read once rather than re-globbed per conversation.
KNOWN_POLICY_IDS = frozenset(p.stem for p in POLICY_DIR.glob("*.md"))

# A citation is the id as a whole word. The agent's house style is
# markdown-bold -- `**cw-returns**`, `**[cw-returns] [cw-store-overrides]**` --
# and the first version of this matched whitespace-delimited tokens with a strip
# set that had no `*`, so it found citations in 32 of 250 conversations where a
# boundary-anchored match finds 121. Every miss read as "cited: none".
_CITE_RE = {
    pid: re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(pid)}(?![A-Za-z0-9_-])")
    for pid in KNOWN_POLICY_IDS
}

# trace id -> why it is out of scope. The reason is kept because the review
# summary must report sample composition, and an unexplained gap between 279 and
# 280 is worse than no gap.
EXCLUDED: dict[str, str] = {
    "1c828dff7e934ab059048deb2644d592":
        "support-0144 attempt 1 (session ef006276): stalled 381.8s, abandoned, "
        "re-run as session c76cae64. Harness failure, not agent behaviour.",
}


class InvariantViolation(RuntimeError):
    """A scenario maps to more than one session, so scenario != conversation."""


def db_path() -> Path:
    """The database to read ground truth from; CARTWHEEL_DB wins, as in tests."""
    env = os.environ.get("CARTWHEEL_DB")
    return Path(env) if env else REPO / "data" / "cartwheel.db"


def _connect() -> sqlite3.Connection:
    path = db_path()
    if not path.exists():
        raise FileNotFoundError(
            f"no database at {path}.\n"
            "Point CARTWHEEL_DB at the post-HW3-run database, or at the snapshot in\n"
            "  0-learn/notes/ai-evals-cohort-6/hw-4/artifacts/cartwheel-hw3-postrun.db\n"
            "Do NOT run `python -m seed.generate` to recover it: that unlinks\n"
            "data/cartwheel.db and reseeds the PRE-run world. Refund rows 575-583 were\n"
            "written by the HW3 run itself, so a fresh seed disagrees with the traces "
            "(order 8438 reads `delivered` there, while its own get_order shows `refunded`)."
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def world_asof(conn: sqlite3.Connection) -> datetime.date:
    """The simulated today, read from the database rather than hardcoded.

    Every eligibility answer and every `days_since_delivery` depends on it, so a
    reseeded or wrong-anchor database would silently shift the whole evidence
    rail. Reading `meta.world_asof` makes that impossible to get wrong quietly.
    """
    row = conn.execute("SELECT value FROM meta WHERE key='world_asof'").fetchone()
    if row is None:
        raise RuntimeError("meta.world_asof missing; is this a Cartwheel database?")
    return datetime.date.fromisoformat(row["value"])


# --------------------------------------------------------------------------- #
# pool
# --------------------------------------------------------------------------- #

def attrs(trace: dict) -> dict:
    a = (trace.get("metadata") or {}).get("attributes") or {}
    return json.loads(a) if isinstance(a, str) else a


def session_of(trace: dict) -> str | None:
    return attrs(trace).get("cartwheel.session_id")


def scenario_of(trace: dict) -> str | None:
    return trace.get("cartwheel_scenario_id") or attrs(trace).get("cartwheel.scenario_id")


def load_pool(export: str | Path = EXPORT) -> list[dict]:
    """Every in-scope trace. Accepts the export wrapper, one trace, or a list."""
    data = json.loads(Path(export).read_text())
    if isinstance(data, dict) and "traces" in data:
        data = data["traces"]
    elif isinstance(data, dict) and "observations" in data:
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"not a trace export, a trace, or a list: {export}")
    pool = [t for t in data if t.get("id") not in EXCLUDED]
    missing = [t.get("id") for t in pool if not session_of(t)]
    if missing:
        raise ValueError(
            f"{len(missing)} trace(s) carry no cartwheel.session_id (e.g. {missing[0]}); "
            "without it every such trace would fuse into one phantom conversation"
        )
    return pool


def sessions(pool: list[dict]) -> dict[str, list[dict]]:
    """session_id -> its traces in timestamp order."""
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for t in pool:
        key = session_of(t)
        if not key:
            raise ValueError(
                f"trace {t.get('id')} has no cartwheel.session_id; without it every "
                "such trace would fuse into one phantom conversation"
            )
        out[key].append(t)
    return {s: sorted(g, key=lambda t: t["timestamp"]) for s, g in out.items()}


def scenarios_with_multiple_sessions(pool: list[dict]) -> dict[str, set[str]]:
    by: dict[str, set[str]] = collections.defaultdict(set)
    for t in pool:
        by[scenario_of(t)].add(session_of(t))
    return {k: v for k, v in by.items() if len(v) > 1}


def check_one_session_per_scenario(pool: list[dict]) -> None:
    """Raise unless every scenario ran exactly once."""
    offenders = scenarios_with_multiple_sessions(pool)
    if not offenders:
        return
    detail = "; ".join(f"{s} -> {sorted(v)}" for s, v in sorted(offenders.items()))
    raise InvariantViolation(
        f"{len(offenders)} scenario(s) map to more than one session: {detail}. "
        "Add the out-of-scope trace id to EXCLUDED with a reason."
    )


def census(pool: list[dict]) -> dict:
    convs = sessions(pool)
    return {
        "traces": len(pool),
        "conversations": len(convs),
        "scenarios": len({scenario_of(t) for t in pool}),
        "excluded": len(EXCLUDED),
        "turns_per_conversation": dict(sorted(collections.Counter(
            len(g) for g in convs.values()).items())),
    }


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #

def _parts_text(messages: Any, role: str | None = None) -> str:
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        if role is not None and m.get("role") != role:
            continue
        for p in m.get("parts") or []:
            if isinstance(p, dict) and p.get("content"):
                out.append(str(p["content"]))
    return "\n".join(out)


def _generation_texts(output: Any) -> list[str]:
    """Assistant prose from one model call. Tool-call parts are not prose."""
    out = []
    for m in output or []:
        if not isinstance(m, dict) or m.get("finish_reason") == "tool_call":
            continue
        for p in m.get("parts") or []:
            if isinstance(p, dict) and p.get("type") == "text" and p.get("content"):
                out.append(p["content"])
    return out


def build_turn(n: int, trace: dict) -> dict:
    """One turn: the user message, the ordered steps, and the reply.

    Observations are sorted by `startTime` because the API does not return them
    in order.
    """
    steps: list[dict] = []
    for o in sorted(trace.get("observations") or [], key=lambda x: x.get("startTime") or ""):
        if o.get("type") == "GENERATION":
            for txt in _generation_texts(o.get("output")):
                steps.append({"kind": "agent", "at": o["startTime"], "text": txt})
        elif o.get("type") == "TOOL":
            steps.append({"kind": "tool", "at": o["startTime"], "tool": o.get("name"),
                          "in": o.get("input"), "out": o.get("output")})
    steps.sort(key=lambda s: s["at"])
    reply = _parts_text(trace.get("output"))
    # the final model call restates the reply; keep it once, as `reply`
    while steps and steps[-1]["kind"] == "agent" and steps[-1]["text"].strip() == reply.strip():
        steps.pop()
    return {
        "turn": n,
        "trace_id": trace["id"],
        "timestamp": trace.get("timestamp"),
        "latency_s": round(trace.get("latency") or 0, 2),
        "user": _parts_text(trace.get("input"), role="user"),
        "steps": steps,
        "reply": reply,
    }


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #

def order_facts(order_id: int, conn: sqlite3.Connection,
                asof: datetime.date | None = None) -> dict:
    """Ground truth for one order, with every derived value flagged if it is odd.

    Eligibility is delegated to `seed.eligibility.is_refund_eligible`, the
    repo's designated oracle -- the same function the seed used to stamp
    `orders.refund_eligible`. The first version of this reimplemented the rule
    inline with a hardcoded 30-day window, which ignored the four stores that
    override it (Juniper 14, Saltbox 7, Meridian 21, Northwind 45) and so raised
    a red "flag disagrees with function" anomaly on consistent data.

    Refund money is split by status. `refunds.status` is `auto_approved` or
    `queued_for_approval`; a queued refund has NOT moved money. Summing both
    commits exactly the conflation this homework's worked example is about --
    "the reply reports a completed refund even though the tool only queued it".
    """
    rows = [dict(r) for r in conn.execute("SELECT * FROM orders WHERE id=?", (order_id,))]
    if not rows:
        return {"order_id": order_id, "MISSING_FROM_DB": True, "flags": ["MISSING_FROM_DB"]}
    o = rows[0]
    asof = asof or world_asof(conn)
    refunds = [dict(r) for r in conn.execute(
        "SELECT amount_cents,status,created_at,reason FROM refunds WHERE order_id=?", (order_id,))]
    store = conn.execute(
        "SELECT name,return_window_days_override FROM stores WHERE id=?", (o["store_id"],)
    ).fetchone()
    product = conn.execute(
        "SELECT title,store_id FROM products WHERE id=?", (o["product_id"],)).fetchone()

    override = store["return_window_days_override"] if store else None
    window = effective_return_window_days(PLATFORM_WINDOW_DAYS, override)
    delivered = o["delivered_at"]
    age = (asof - datetime.date.fromisoformat(delivered)).days if delivered else None
    by_function = is_refund_eligible(
        status=o["status"],
        delivered_at=datetime.date.fromisoformat(delivered) if delivered else None,
        as_of=asof,
        return_window_days=window,
    )
    executed = sum(r["amount_cents"] for r in refunds if r["status"] == "auto_approved") / 100
    queued = sum(r["amount_cents"] for r in refunds if r["status"] == "queued_for_approval") / 100
    total = o["total_cents"] / 100

    facts = {
        "order_id": order_id,
        "owner_user_id": o["user_id"],
        "store": store["name"] if store else None,
        "store_id": o["store_id"],
        "product_id": o["product_id"],
        "product": product["title"] if product else None,
        "product_store_id": product["store_id"] if product else None,
        "quantity": o["quantity"],
        "total_usd": total,
        "status": o["status"],
        "ordered_at": o["ordered_at"],
        "shipped_at": o["shipped_at"],
        "delivered_at": delivered,
        "days_since_delivery_vs_asof": age,
        "return_window_days": window,
        "store_window_override_days": override,
        "refund_eligible_flag": bool(o["refund_eligible"]),
        "refund_eligible_by_function": by_function,
        "refunds_on_record": [
            {"usd": r["amount_cents"] / 100, "status": r["status"],
             "at": r["created_at"], "reason": r["reason"]} for r in refunds],
        "refunded_executed_usd": executed,
        "refunded_queued_usd": queued,
    }

    flags: list[str] = []
    if facts["refund_eligible_flag"] != by_function:
        # Two different causes, and only the second is a data defect.
        #
        # `agent/db.py` sets `status='refunded'` when a refund executes and never
        # clears `refund_eligible`, so any order this run refunded now carries a
        # stale stamp. That is bookkeeping drift, not an inconsistency the agent
        # should have caught -- and painting it as an anomaly on the very
        # conversation that issued the refund misleads the reviewer.
        #
        # A disagreement on a non-refunded order is the seeded kind: order 8001
        # is the only such row in a freshly seeded world.
        if o["status"] == "refunded":
            flags.append("ELIGIBILITY_STAMP_STALE_AFTER_REFUND")
        else:
            # On an order whose store_id is itself corrupt (the seeded
            # store-mismatch case) the window resolves against the wrong store,
            # so this can fire for a mismatch reason. ORDER_STORE_NE_PRODUCT_STORE
            # shows on the same card; read them together.
            flags.append("ELIGIBILITY_FLAG_DISAGREES_WITH_ORACLE")
    if 0 < executed < total:
        flags.append("PARTIALLY_REFUNDED")
    elif executed == total > 0:
        flags.append("FULLY_REFUNDED")
    elif executed > total > 0:
        flags.append("OVER_REFUNDED")
    if queued:
        flags.append("REFUND_QUEUED_NOT_EXECUTED")
    # The seed sets status='refunded' while writing the refund as
    # queued_for_approval whenever the total exceeds the threshold, so the order
    # record itself says refunded with no money moved. That is precisely the
    # conflation hw4.md's worked example is about, and it needs saying out loud.
    if o["status"] == "refunded" and executed == 0:
        flags.append("STATUS_REFUNDED_BUT_NOTHING_EXECUTED")
    # the three seeded order-side data-quality defects, each otherwise invisible
    if o["status"] == "delivered" and not delivered:
        flags.append("DELIVERED_WITHOUT_DELIVERY_DATE")
    if o["shipped_at"] and delivered and o["shipped_at"] > delivered:
        flags.append("SHIPPED_AFTER_DELIVERED")
    if product and product["store_id"] != o["store_id"]:
        flags.append("ORDER_STORE_NE_PRODUCT_STORE")
    facts["flags"] = flags
    return facts


def product_facts(product_id: int, conn: sqlite3.Connection) -> dict:
    rows = [dict(r) for r in conn.execute("SELECT * FROM products WHERE id=?", (product_id,))]
    if not rows:
        return {"product_id": product_id, "MISSING_FROM_DB": True}
    p = rows[0]
    store = [dict(r) for r in conn.execute("SELECT name FROM stores WHERE id=?", (p["store_id"],))]
    siblings = [dict(r) for r in conn.execute(
        "SELECT id,price_cents FROM products WHERE store_id=? AND title=? AND id<>?",
        (p["store_id"], p["title"], product_id))]
    facts = {
        "product_id": product_id,
        "store_id": p["store_id"],
        "store": store[0]["name"] if store else None,
        "title": p["title"],
        "price_usd": p["price_cents"] / 100,
        "same_title_in_same_store": [
            {"product_id": s["id"], "price_usd": s["price_cents"] / 100} for s in siblings],
    }
    flags = []
    if p["title"] == "":
        flags.append("TITLE_IS_EMPTY")
    if p["price_cents"] < 0:
        flags.append("PRICE_IS_NEGATIVE")
    if siblings:
        flags.append("SAME_TITLE_IN_SAME_STORE")
    facts["flags"] = flags
    return facts


def referenced_entities(turns: list[dict]) -> tuple[list[int], list[int]]:
    """Order and product ids this conversation touched, from tool in and out."""
    orders: set[int] = set()
    products: set[int] = set()
    for t in turns:
        for s in t["steps"]:
            if s["kind"] != "tool":
                continue
            src = s.get("in") if isinstance(s.get("in"), dict) else {}
            if isinstance(src.get("order_id"), int):
                orders.add(src["order_id"])
            out = s.get("out") if isinstance(s.get("out"), dict) else {}
            records = []
            if isinstance(out.get("order"), dict):
                records.append(out["order"])
            records += [r for r in (out.get("orders") or []) if isinstance(r, dict)]
            for r in records:
                if isinstance(r.get("order_id"), int):
                    orders.add(r["order_id"])
                if isinstance(r.get("product_id"), int):
                    products.add(r["product_id"])
            for r in (out.get("products") or []):
                if isinstance(r, dict) and isinstance(r.get("product_id"), int):
                    products.add(r["product_id"])
    return sorted(orders), sorted(products)


def policy_provenance(turns: list[dict]) -> dict:
    """Which policy ids this conversation OPENED, which merely surfaced, which it cited.

    Three sets, not two, because the distinction carries the signal:

    * `opened`   -- returned by `get_policy`, so the agent read the body.
    * `surfaced` -- appeared in a `search_help_center` result list as a snippet.
    * `cited`    -- named in a reply, matched as a whole word so markdown bold
                    and bracket groups count.

    The first version collapsed opened and surfaced into one "retrieved" set,
    which made `cited_not_retrieved` empty on all 250 conversations. Separating
    them surfaces the 9 conversations that cite a policy they never opened --
    three of which called `get_policy` zero times.
    """
    opened: set[str] = set()
    surfaced: set[str] = set()
    cited: set[str] = set()
    for turn in turns:
        for s in turn["steps"]:
            if s["kind"] != "tool":
                continue
            out = s.get("out") if isinstance(s.get("out"), dict) else {}
            if s.get("tool") == "get_policy" and out.get("policy_id"):
                opened.add(out["policy_id"])
            for r in (out.get("results") or []):
                if isinstance(r, dict) and r.get("policy_id"):
                    surfaced.add(r["policy_id"])
        reply = turn["reply"] or ""
        cited |= {pid for pid, rx in _CITE_RE.items() if rx.search(reply)}
    return {
        "opened": sorted(opened),
        "surfaced": sorted(surfaced - opened),
        "cited": sorted(cited),
        "cited_not_opened": sorted(cited - opened),
        "cited_not_seen_at_all": sorted(cited - opened - surfaced),
        "known_policy_ids": sorted(KNOWN_POLICY_IDS),
    }


# --------------------------------------------------------------------------- #
# the view
# --------------------------------------------------------------------------- #

def build_view(traces: list[dict], conn: sqlite3.Connection | None = None,
               asof: datetime.date | None = None) -> dict:
    """One conversation, ready to render. `traces` must be one session's.

    `conn` is threaded so `all_views` opens one connection for 250 conversations
    instead of 250 that are committed but never closed.
    """
    ordered = sorted(traces, key=lambda t: t["timestamp"])
    a = attrs(ordered[0])
    turns = [build_turn(i + 1, t) for i, t in enumerate(ordered)]
    orders, products = referenced_entities(turns)
    own = conn is None
    conn = conn or _connect()
    try:
        asof = asof or world_asof(conn)
        facts = {
            "orders": [order_facts(o, conn, asof) for o in orders],
            "products": [product_facts(p, conn) for p in products],
        }
    finally:
        if own:
            conn.close()
    tool_calls = sum(1 for t in turns for s in t["steps"] if s["kind"] == "tool")
    distinct = {s["tool"] for t in turns for s in t["steps"] if s["kind"] == "tool"}
    return {
        "world_asof": asof.isoformat(),
        "session_id": a.get("cartwheel.session_id"),
        "scenario_id": scenario_of(ordered[0]),
        "user_role": a.get("cartwheel.user_role"),
        "user_id": a.get("cartwheel.user_id"),
        "prompt_version": a.get("cartwheel.prompt_version"),
        "totals": {
            "turns": len(turns),
            "traces": len(ordered),
            "tool_calls": tool_calls,
            "distinct_tools": len(distinct),
            "latency_s": round(sum(t["latency_s"] for t in turns), 2),
        },
        "trace_ids": [t["trace_id"] for t in turns],
        "turns": turns,
        "facts": facts,
        "policy": policy_provenance(turns),
    }


def all_views(pool: list[dict] | None = None) -> list[dict]:
    """Every conversation in the pool, newest last, invariant enforced."""
    pool = load_pool() if pool is None else pool
    check_one_session_per_scenario(pool)
    conn = _connect()
    try:
        asof = world_asof(conn)
        views = [build_view(g, conn, asof) for g in sessions(pool).values()]
    finally:
        conn.close()
    views.sort(key=lambda v: v["turns"][0]["timestamp"])
    return views


def view_for(key: str, pool: list[dict] | None = None) -> dict | None:
    """One conversation by session id or by scenario id."""
    pool = load_pool() if pool is None else pool
    for sess, group in sessions(pool).items():
        if key == sess or key == scenario_of(group[0]):
            return build_view(group)
    return None
