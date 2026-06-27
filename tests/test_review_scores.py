"""1-10 AI rating dimensions on reviews/testimonials (video + text):
compliance / publication / authenticity / specificity (10 = best)."""
import sqlite3
from dashboard import review_scoring as rs
from dashboard import product_reviews as pr


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


_DIMS = ("compliance_score", "publication_score", "authenticity_score", "specificity_score")


def test_score_review_returns_four_dimensions():
    c = _FakeClient('{"compliance_ok": true, "reasons": "x", "quality_points": 2, '
                    '"recommend_publish": true, "compliance_score": 9, "publication_score": 7, '
                    '"authenticity_score": 8, "specificity_score": 6}')
    out = rs.score_review(c, {"name": "X"}, "specific authentic story")
    assert out["compliance_score"] == 9 and out["publication_score"] == 7
    assert out["authenticity_score"] == 8 and out["specificity_score"] == 6
    # back-compat keys still present
    assert out["compliance_ok"] is True and out["quality_points"] == 2


def test_score_video_returns_four_dimensions():
    c = _FakeClient('{"video_points": 4, "publish_risk": false, "risk_reasons": "", '
                    '"recommend_publish": true, "compliance_score": 10, "publication_score": 8, '
                    '"authenticity_score": 9, "specificity_score": 7}')
    out = rs.score_video(c, {"name": "X"}, "a clear spoken transcript")
    assert [out[d] for d in _DIMS] == [10, 8, 9, 7]
    assert out["video_points"] == 4  # back-compat


def test_dimensions_clamped_to_0_10():
    c = _FakeClient('{"compliance_score": 15, "publication_score": -3, '
                    '"authenticity_score": 8, "specificity_score": 99}')
    out = rs.score_review(c, {"name": "X"}, "body")
    assert out["compliance_score"] == 10 and out["publication_score"] == 0
    assert out["authenticity_score"] == 8 and out["specificity_score"] == 10


def test_bad_json_dimensions_default_zero():
    out = rs.score_review(_FakeClient("not json"), {"name": "X"}, "body")
    assert [out[d] for d in _DIMS] == [0, 0, 0, 0]
    out_v = rs.score_video(_FakeClient("not json"), {"name": "X"}, "t")
    assert [out_v[d] for d in _DIMS] == [0, 0, 0, 0]


def test_prompts_mention_the_four_scales():
    sys_t, _ = rs.build_review_prompt({"name": "X"}, "b")
    sys_v, _ = rs.build_video_prompt({"name": "X"}, "t")
    for s in (sys_t.lower(), sys_v.lower()):
        for d in ("compliance_score", "publication_score", "authenticity_score", "specificity_score"):
            assert d in s


def test_product_reviews_stores_dimension_scores():
    cx = sqlite3.connect(":memory:")
    rid = pr.upsert_review(cx, "_results", "a@x.com", "A", 5, body="b", kind="testimonial")
    pr.set_scores(cx, rid, compliance=9, publication=7, authenticity=8, specificity=6)
    r = pr.get_review(cx, rid)
    assert r["compliance_score"] == 9 and r["publication_score"] == 7
    assert r["authenticity_score"] == 8 and r["specificity_score"] == 6
    # default 0 before scoring
    rid2 = pr.upsert_review(cx, "_results", "b@x.com", "B", 5, body="b", kind="testimonial")
    assert pr.get_review(cx, rid2)["compliance_score"] == 0
