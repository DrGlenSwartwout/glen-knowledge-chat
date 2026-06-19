import sqlite3
from dashboard import product_reviews as pr


def _cx():
    return sqlite3.connect(":memory:")


def test_upsert_and_done_state():
    cx = _cx()
    rid = pr.upsert_review(cx, "longevity", "a@x.com", "Ann", 5, body="great")
    assert isinstance(rid, int)
    assert pr.has_reviewed(cx, "longevity", "a@x.com") is True
    assert pr.has_reviewed(cx, "longevity", "b@x.com") is False
    # re-submit updates the same row (UNIQUE slug,email) and re-moderates
    rid2 = pr.upsert_review(cx, "longevity", "a@x.com", "Ann", 4, body="updated")
    assert rid2 == rid
    row = pr.get_review(cx, rid)
    assert row["rating"] == 4 and row["body"] == "updated" and row["status"] == "pending"


def test_ai_points_status_feature_setters():
    cx = _cx()
    rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, body="b")
    pr.set_ai_result(cx, rid, 2, "ok", 1)
    pr.set_points(cx, rid, 2)
    pr.set_status(cx, rid, "approved", by="Glen")
    pr.set_featured(cx, rid, True)
    r = pr.get_review(cx, rid)
    assert r["ai_score"] == 2 and r["points_awarded"] == 2
    assert r["status"] == "approved" and r["reviewed_by"] == "Glen" and r["featured"] == 1


def test_approved_for_slug_and_aggregate_exclude_unapproved():
    cx = _cx()
    a = pr.upsert_review(cx, "x", "a@x.com", "Ann", 4, body="a")
    b = pr.upsert_review(cx, "x", "b@x.com", "Bob", 2, body="b")
    c = pr.upsert_review(cx, "x", "c@x.com", "Cy", 5, body="c")
    pr.set_status(cx, a, "approved"); pr.set_status(cx, c, "approved")
    pr.set_status(cx, b, "rejected")
    pr.set_featured(cx, c, True)
    rows = pr.approved_for_slug(cx, "x")
    assert [r["email"] for r in rows] == ["c@x.com", "a@x.com"]   # featured first
    agg = pr.aggregate(cx, "x")
    assert agg["count"] == 2 and agg["avg"] == 4.5                 # (4+5)/2
    assert pr.aggregate(cx, "none") == {"count": 0, "avg": 0.0}


def test_pending_queue_only_pending():
    cx = _cx()
    a = pr.upsert_review(cx, "x", "a@x.com", "Ann", 4, body="a")
    b = pr.upsert_review(cx, "x", "b@x.com", "Bob", 5, body="b")
    pr.set_status(cx, a, "approved")
    q = pr.pending_queue(cx)
    assert [r["email"] for r in q] == ["b@x.com"]
