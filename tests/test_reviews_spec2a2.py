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


# ---------------------------------------------------------------------------
# Task 2: score_video
# ---------------------------------------------------------------------------
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


def test_build_video_prompt_includes_transcript():
    system, user = rs.build_video_prompt({"name": "Longevity"}, "I felt more energy in two weeks")
    assert "Longevity" in user and "energy in two weeks" in user


def test_score_video_quality_points():
    c = _FakeClient('{"video_points": 5, "publish_risk": false, "risk_reasons": "", "recommend_publish": true}')
    out = rs.score_video(c, {"name": "X"}, "clear specific authentic spoken review")
    assert out["video_points"] == 5 and out["publish_risk"] is False and out["recommend_publish"] is True


def test_score_video_disease_claim_still_scores_but_flags_risk():
    c = _FakeClient('{"video_points": 4, "publish_risk": true, "risk_reasons": "claims to cure disease", "recommend_publish": false}')
    out = rs.score_video(c, {"name": "X"}, "this cured my illness completely")
    assert out["video_points"] == 4               # NOT zeroed
    assert out["publish_risk"] is True and "cure" in out["risk_reasons"].lower()


def test_score_video_spam_zero():
    c = _FakeClient('{"video_points": 0, "publish_risk": false, "risk_reasons": "low effort", "recommend_publish": false}')
    out = rs.score_video(c, {"name": "X"}, "uhh idk")
    assert out["video_points"] == 0


def test_score_video_bad_json_safe_default():
    out = rs.score_video(_FakeClient("not json"), {"name": "X"}, "x")
    assert out["video_points"] == 0 and out["publish_risk"] is False and out["recommend_publish"] is False


def test_score_video_strips_dashes():
    c = _FakeClient('{"video_points": 3, "publish_risk": true, "risk_reasons": "risk — here", "recommend_publish": false}')
    out = rs.score_video(c, {"name": "X"}, "x", strip=lambda s: s.replace("—", ","))
    assert "—" not in out["risk_reasons"]


# ---------------------------------------------------------------------------
# Task 3: REVIEWS_VIDEO flag, enqueue-on-submit, length-limits endpoint
# ---------------------------------------------------------------------------
import importlib
import io


def _reload_video_app(monkeypatch, tmp_path, video="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEWS_ENABLED", "true")
    monkeypatch.setenv("REVIEWS_VIDEO", video)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_paid_order(appmod, email, product_name):
    import sqlite3
    from dashboard import orders as o
    with sqlite3.connect(appmod.LOG_DB) as cx:
        o.upsert_order(cx, source="test", external_ref=f"t-{email}", email=email,
                       items=[{"name": product_name, "qty": 1}], total_cents=7000, status="paid")


def test_upload_review_enqueues_video_job(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    from dashboard import review_scoring as rs_mod
    monkeypatch.setattr(rs_mod, "score_review", lambda *a, **k: {
        "compliance_ok": True, "reasons": "", "quality_points": 1, "recommend_publish": True})
    c = appmod.app.test_client()
    data = {"slug": slug, "rating": "5", "body": "nice", "email": "buyer@x.com"}
    data["video"] = (io.BytesIO(b"FAKEWEBMDATA"), "clip.webm")
    r = c.post("/api/reviews", data=data, content_type="multipart/form-data").get_json()
    assert r is not None and r.get("ok"), f"submit failed: {r}"
    import sqlite3
    from dashboard import review_video_jobs as vj
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert r["review_id"] in vj.claim_pending(cx), "job not enqueued"
    assert r.get("video_status") == "pending"


def test_limits_endpoint_90_then_300(monkeypatch, tmp_path):
    appmod = _reload_video_app(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    base = c.get("/api/reviews/limits?email=new@x.com").get_json()
    assert base["max_seconds"] == 90
    import sqlite3
    from dashboard import product_reviews as pr_mod
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr_mod.upsert_review(cx, "x", "vet@x.com", "V", 5, video_kind="upload", video_ref="v.webm")
        pr_mod.set_video_result(cx, rid, 5, "t", "scored")
        pr_mod.set_status(cx, rid, "approved")
    up = c.get("/api/reviews/limits?email=vet@x.com").get_json()
    assert up["max_seconds"] == 300
