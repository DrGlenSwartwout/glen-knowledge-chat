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


import importlib


def _reload_reviews_app(monkeypatch, tmp_path, enabled="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEWS_ENABLED", enabled)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _seed_paid_order(appmod, email, product_name):
    import sqlite3
    from dashboard import orders as o
    with sqlite3.connect(appmod.LOG_DB) as cx:
        o.upsert_order(cx, source="test", external_ref=f"t-{email}", email=email,
                       items=[{"name": product_name, "qty": 1}], total_cents=7000, status="paid")


def test_submit_requires_verified_buyer(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    c = appmod.app.test_client()
    r = c.post("/api/reviews", json={"slug": slug, "rating": 5, "body": "great",
                                     "email": "stranger@x.com"})
    assert r.status_code == 403


def test_submit_scores_and_credits_points(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    # fake the scorer to pass with 2 points
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_review", lambda *a, **k: {
        "compliance_ok": True, "reasons": "ok", "quality_points": 2, "recommend_publish": True})
    c = appmod.app.test_client()
    r = c.post("/api/reviews", json={"slug": slug, "rating": 5, "body": "specific and useful",
                                     "email": "buyer@x.com"}).get_json()
    assert r["ok"] and r["points_awarded"] == 2 and r["status"] == "pending"
    import sqlite3
    from dashboard import points
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "buyer@x.com") == 200   # 2 points * 100c
    # idempotent: re-submit does not double-credit
    c.post("/api/reviews", json={"slug": slug, "rating": 4, "body": "edit", "email": "buyer@x.com"})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "buyer@x.com") == 200


def test_submit_gate_fail_no_points(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_review", lambda *a, **k: {
        "compliance_ok": False, "reasons": "disease claim", "quality_points": 0, "recommend_publish": False})
    c = appmod.app.test_client()
    r = c.post("/api/reviews", json={"slug": slug, "rating": 5, "body": "cured my disease",
                                     "email": "buyer@x.com"}).get_json()
    assert r["ok"] and r["points_awarded"] == 0
    import sqlite3
    from dashboard import points
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert points.balance(cx, "buyer@x.com") == 0


def test_submit_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_reviews_app(monkeypatch, tmp_path, enabled="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    r = appmod.app.test_client().post("/api/reviews", json={"slug": slug, "rating": 5})
    assert r.status_code == 404


# ── Task 4: reorder items annotated with reviewed flag ────────────────────────

def test_reorder_items_reviewed_annotation_false_when_no_review(monkeypatch, tmp_path):
    """Unreviewed item -> reviewed=False when flag on."""
    appmod = _reload_reviews_app(monkeypatch, tmp_path, enabled="true")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    c = appmod.app.test_client()
    c.set_cookie("rm_reorder_email", "buyer@x.com")
    d = c.get("/api/reorder/items").get_json()
    assert d and "items" in d
    item = next((i for i in d["items"] if i.get("slug") == slug), None)
    assert item is not None, f"slug {slug!r} not in reorder items"
    assert item["reviewed"] is False


def test_reorder_items_reviewed_annotation_true_after_review(monkeypatch, tmp_path):
    """After submitting a review, reviewed=True."""
    appmod = _reload_reviews_app(monkeypatch, tmp_path, enabled="true")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    import sqlite3
    from dashboard import product_reviews as _pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _pr.upsert_review(cx, slug, "buyer@x.com", "Buyer", 5, body="great")
    c = appmod.app.test_client()
    c.set_cookie("rm_reorder_email", "buyer@x.com")
    d = c.get("/api/reorder/items").get_json()
    item = next((i for i in d["items"] if i.get("slug") == slug), None)
    assert item is not None
    assert item["reviewed"] is True


def test_reorder_items_reviewed_true_when_flag_off(monkeypatch, tmp_path):
    """When REVIEWS_ENABLED=false, reviewed is always True (gate inert)."""
    appmod = _reload_reviews_app(monkeypatch, tmp_path, enabled="false")
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    name = appmod._get_product(slug)["name"]
    _seed_paid_order(appmod, "buyer@x.com", name)
    c = appmod.app.test_client()
    c.set_cookie("rm_reorder_email", "buyer@x.com")
    d = c.get("/api/reorder/items").get_json()
    item = next((i for i in d["items"] if i.get("slug") == slug), None)
    assert item is not None
    assert item["reviewed"] is True
