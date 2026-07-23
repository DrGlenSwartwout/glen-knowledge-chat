import sqlite3

from dashboard.recommendation_events import (
    init_recommendation_events,
    record_event,
    product_sources,
    clear_events,
)


def _cx():
    cx = sqlite3.connect(":memory:")
    init_recommendation_events(cx)
    return cx


def test_clear_events_removes_only_matching_source_and_origin_ref():
    cx = _cx()
    email = "a@x.com"
    record_event(cx, email, "p1", "condition",
                 occurred_at="2026-07-23T00:00:00Z", origin_ref="glaucoma")
    record_event(cx, email, "p1", "purchased",
                 occurred_at="2026-07-23T00:00:00Z", origin_ref="order-1")

    deleted = clear_events(cx, email, "condition", "glaucoma")

    assert deleted == 1
    sources = product_sources(cx, email)
    assert len(sources) == 1
    keys = {s["source"] for s in sources[0]["sources"]}
    assert "condition" not in keys
    assert "purchased" in keys
