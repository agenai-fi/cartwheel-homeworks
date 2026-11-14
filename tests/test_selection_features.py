"""Merged conversations carry features of the whole conversation, and a
cluster representative is its cluster's nearest member. Runs on the real HW3
export, since the stale-feature defect only shows on multi-turn records."""

from __future__ import annotations

import pytest

from analysis.helpers import selection
from analysis.helpers.normalization import (
    _message_features,
    normalize_trace,
    normalize_traces,
)
from analysis.review_app import loader


@pytest.fixture(scope="module")
def pool():
    return loader.load_pool()


def test_merged_features_cover_every_turn(pool):
    per_scenario: dict[str, list[dict]] = {}
    for raw in pool:
        t = normalize_trace(raw)
        per_scenario.setdefault(t["meta"]["scenario_id"], []).append(t)
    merged = normalize_traces(pool)
    multi = [m for m in merged if len(per_scenario[m["meta"]["scenario_id"]]) > 1]
    assert len(multi) == 27
    for m in multi:
        group = per_scenario[m["meta"]["scenario_id"]]
        assert m["features"] == {
            **_message_features(m["trace"]),
            "tokens": sum(t["features"]["tokens"] for t in group),
        }
    # support-0246 ran five distinct tools across its turns; its first turn ran three
    s0246 = next(m for m in merged if m["meta"]["scenario_id"] == "support-0246")
    assert s0246["features"]["distinct_tools"] == 5


def test_representative_is_nearest_eligible_member(pool):
    traces = normalize_traces(pool)
    picks = selection.representatives(traces, k=8, exclude_ids=set())
    vectors = selection._standardize([selection._feature_vector(t) for t in traces])
    assign, centroids = selection._kmeans(vectors, k=8)
    index = {t["id"]: i for i, t in enumerate(traces)}

    def dist(i, c):
        return sum((a - b) ** 2 for a, b in zip(vectors[i], centroids[c]))

    assert len(picks) == 8 and len({p["trace_id"] for p in picks}) == 8
    for p in picks:
        i, c = index[p["trace_id"]], p["cluster"]
        assert assign[i] == c
        assert dist(i, c) == min(dist(j, c) for j, a in enumerate(assign) if a == c)

    # excluding a representative promotes the next-nearest member of its cluster
    top = picks[0]
    again = selection.representatives(traces, k=8, exclude_ids={top["trace_id"]})
    replaced = next(p for p in again if p["cluster"] == top["cluster"])
    assert replaced["trace_id"] not in {None, top["trace_id"]}
