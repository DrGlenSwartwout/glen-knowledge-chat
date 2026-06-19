import sqlite3
from dashboard import product_reviews as pr
from dashboard import review_video_jobs as vj


def _cx():
    return sqlite3.connect(":memory:")


def test_video_columns_and_set_video_result():
    cx = _cx()
    rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, body="b", video_kind="upload", video_ref="v.webm")
    pr.set_video_result(cx, rid, 4, "great transcript", "scored", publish_risk=1, video_verdict="disease claim")
    r = pr.get_review(cx, rid)
    assert r["video_points"] == 4 and r["transcript"] == "great transcript"
    assert r["video_status"] == "scored" and r["publish_risk"] == 1 and r["video_verdict"] == "disease claim"


def test_has_successful_video():
    cx = _cx()
    rid = pr.upsert_review(cx, "x", "a@x.com", "Ann", 5, video_kind="upload", video_ref="v.webm")
    assert pr.has_successful_video(cx, "a@x.com") is False     # no points / not approved yet
    pr.set_video_result(cx, rid, 5, "t", "scored")
    assert pr.has_successful_video(cx, "a@x.com") is False     # scored but not approved
    pr.set_status(cx, rid, "approved")
    assert pr.has_successful_video(cx, "a@x.com") is True      # points + approved


def test_video_job_queue_roundtrip():
    cx = _cx()
    vj.enqueue(cx, 11)
    vj.enqueue(cx, 12)
    assert vj.claim_pending(cx) == [11, 12]
    vj.mark(cx, 11, "done")
    assert vj.claim_pending(cx) == [12]
    vj.enqueue(cx, 11)   # re-enqueue resets to pending
    assert set(vj.claim_pending(cx)) == {11, 12}
