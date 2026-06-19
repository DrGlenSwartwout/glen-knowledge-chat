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


from dashboard import review_scoring as rs


class _Blk:
    def __init__(self, t): self.type = "text"; self.text = t


class _Msg:
    def __init__(self, t): self.content = [_Blk(t)]


class _FakeClient:
    def __init__(self, payload): self._p = payload
    @property
    def messages(self):
        outer = self
        class _M:
            def create(self, **kw): return _Msg(outer._p)
        return _M()


def test_build_prompt_mentions_product_and_rules():
    system, user = rs.build_review_prompt({"name": "Longevity"}, "helped my energy")
    assert "Longevity" in user and "helped my energy" in user
    assert "structure" in (system + user).lower() or "disease" in (system + user).lower()


def test_score_review_quality_points():
    c = _FakeClient('{"compliance_ok": true, "reasons": "specific", "quality_points": 2, "recommend_publish": true}')
    out = rs.score_review(c, {"name": "X"}, "Detailed, specific, authentic review of my experience.")
    assert out["compliance_ok"] is True and out["quality_points"] == 2 and out["recommend_publish"] is True


def test_score_review_gates_disease_claim():
    c = _FakeClient('{"compliance_ok": false, "reasons": "disease claim", "quality_points": 0, "recommend_publish": false}')
    out = rs.score_review(c, {"name": "X"}, "This cured my cancer in two weeks!")
    assert out["compliance_ok"] is False and out["quality_points"] == 0


def test_score_review_bad_json_safe_default():
    c = _FakeClient("not json at all")
    out = rs.score_review(c, {"name": "X"}, "whatever")
    assert out["compliance_ok"] is False and out["quality_points"] == 0


def test_score_review_strips_dashes_in_reasons():
    c = _FakeClient('{"compliance_ok": true, "reasons": "good — solid", "quality_points": 1, "recommend_publish": true}')
    out = rs.score_review(c, {"name": "X"}, "ok", strip=lambda s: s.replace("—", ","))
    assert "—" not in out["reasons"]
