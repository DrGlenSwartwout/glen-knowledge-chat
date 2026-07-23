import sqlite3
from dashboard import remedies_block, recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    return cx


def test_condition_seeded_remedies_land_in_from_history_not_ranked():
    cx = _cx()
    re.init_recommendation_events(cx)
    re.record_event(cx, "a@b.com", "prodX", "condition",
                     occurred_at="2026-07-01T00:00:00+00:00", origin_ref="glaucoma")
    re.record_event(cx, "a@b.com", "prodY", "biofield",
                     occurred_at="2026-07-02T00:00:00+00:00", origin_ref="scan1")

    blk = remedies_block.build_block(cx, "a@b.com", True)

    from_history_keys = {p["product_key"] for p in blk["from_history"]}
    ranked_keys = {p["product_key"] for p in blk["ranked"]}

    assert "prodX" in from_history_keys
    assert "prodY" not in from_history_keys

    assert "prodY" in ranked_keys
    assert "prodX" not in ranked_keys


def test_from_history_empty_when_no_condition_events():
    cx = _cx()
    re.init_recommendation_events(cx)
    re.record_event(cx, "a@b.com", "prodY", "biofield",
                     occurred_at="2026-07-02T00:00:00+00:00", origin_ref="scan1")

    blk = remedies_block.build_block(cx, "a@b.com", True)
    assert blk["from_history"] == []


def test_from_history_shape_and_dedup():
    cx = _cx()
    re.init_recommendation_events(cx)
    re.record_event(cx, "a@b.com", "prodX", "condition",
                     occurred_at="2026-07-01T00:00:00+00:00", origin_ref="glaucoma")
    # A second condition event for the same product_key (different origin_ref) should
    # not produce a duplicate row.
    re.record_event(cx, "a@b.com", "prodX", "condition",
                     occurred_at="2026-07-03T00:00:00+00:00", origin_ref="glaucoma-2")

    blk = remedies_block.build_block(cx, "a@b.com", True)
    hist = blk["from_history"]
    assert len([p for p in hist if p["product_key"] == "prodX"]) == 1
    row = hist[0]
    assert set(row.keys()) == {"product_key", "name", "url", "reason"}
