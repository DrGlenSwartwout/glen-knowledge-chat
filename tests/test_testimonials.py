"""Phase 1: in-house testimonial collection (Boast.io replacement).

Testimonials reuse the product_reviews engine under the reserved campaign slug
"_results", distinguished by kind='testimonial', with practitioner attribution
and public-use consent captured at collection time.
"""
import importlib
import sqlite3

from dashboard import product_reviews as pr


def _cx():
    return sqlite3.connect(":memory:")


# ── Pure product_reviews extension (no app import needed) ─────────────────────

def test_upsert_defaults_kind_product():
    """Existing callers that pass no kind get kind='product', attribution 0, no consent."""
    cx = _cx()
    rid = pr.upsert_review(cx, "longevity", "a@x.com", "Ann", 5, body="great")
    r = pr.get_review(cx, rid)
    assert r["kind"] == "product"
    assert r["practitioner_id"] == 0
    assert r["consent_public"] == 0


def test_upsert_testimonial_stores_new_fields():
    cx = _cx()
    rid = pr.upsert_review(cx, "_results", "a@x.com", "Ann", 5, body="changed my mornings",
                           kind="testimonial", practitioner_id=42, consent_public=1)
    r = pr.get_review(cx, rid)
    assert r["kind"] == "testimonial"
    assert r["practitioner_id"] == 42
    assert r["consent_public"] == 1
    assert r["product_slug"] == "_results"


def test_upsert_source_tag_stores_and_defaults():
    cx = _cx()
    tagged = pr.upsert_review(cx, "_results", "a@x.com", "Ann", 5, body="b",
                              kind="testimonial", source_tag="ash-cert-l1")
    plain = pr.upsert_review(cx, "_results", "b@x.com", "Bo", 5, body="b", kind="testimonial")
    assert pr.get_review(cx, tagged)["source_tag"] == "ash-cert-l1"
    assert pr.get_review(cx, plain)["source_tag"] == ""


def test_resubmit_testimonial_preserves_kind_and_consent():
    """Re-submitting (same slug,email) keeps testimonial kind/consent, not the product defaults."""
    cx = _cx()
    rid = pr.upsert_review(cx, "_results", "a@x.com", "Ann", 5, body="first",
                           kind="testimonial", practitioner_id=7, consent_public=1)
    rid2 = pr.upsert_review(cx, "_results", "a@x.com", "Ann", 4, body="second",
                            kind="testimonial", practitioner_id=7, consent_public=1)
    assert rid2 == rid
    r = pr.get_review(cx, rid)
    assert r["kind"] == "testimonial" and r["consent_public"] == 1
    assert r["rating"] == 4 and r["body"] == "second" and r["status"] == "pending"


# ── Endpoint tests (need app import; run under doppler for Pinecone-at-import) ─

def _reload_app(monkeypatch, tmp_path, *, testimonials="true", reviews="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REVIEWS_ENABLED", reviews)
    monkeypatch.setenv("TESTIMONIALS_ENABLED", testimonials)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def _pass_scorer(monkeypatch, compliance_ok=True):
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_review", lambda *a, **k: {
        "compliance_ok": compliance_ok, "reasons": "ok",
        "quality_points": 2, "recommend_publish": compliance_ok})


def test_results_page_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, testimonials="false")
    assert appmod.app.test_client().get("/results").status_code == 404


def test_results_page_200_when_flag_on(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    r = appmod.app.test_client().get("/results")
    assert r.status_code == 200
    assert b"pay it forward" in r.data.lower()


def test_api_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, testimonials="false")
    r = appmod.app.test_client().post("/api/testimonials",
                                      json={"name": "Ann", "email": "a@x.com", "rating": 5,
                                            "consent_public": "1"})
    assert r.status_code == 404


def test_happy_path_creates_testimonial(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    _pass_scorer(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/testimonials",
               json={"name": "Ann", "email": "ann@x.com", "rating": 5,
                     "body": "Specific, authentic story about my energy.",
                     "consent_public": "1"})
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] is True and j["status"] == "pending"
    from dashboard import product_reviews as _pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = _pr.get_review(cx, j["review_id"])
    assert row["kind"] == "testimonial"
    assert row["product_slug"] == "_results"
    assert row["consent_public"] == 1
    assert row["email"] == "ann@x.com" and row["rating"] == 5


def test_requires_consent(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    _pass_scorer(monkeypatch)
    r = appmod.app.test_client().post("/api/testimonials",
                                      json={"name": "Ann", "email": "a@x.com", "rating": 5,
                                            "body": "nice"})
    assert r.status_code == 400
    assert "consent" in (r.get_json() or {}).get("error", "").lower()


def test_requires_email(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    _pass_scorer(monkeypatch)
    r = appmod.app.test_client().post("/api/testimonials",
                                      json={"name": "Ann", "rating": 5, "consent_public": "1"})
    assert r.status_code == 400


def test_requires_valid_rating(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    _pass_scorer(monkeypatch)
    r = appmod.app.test_client().post("/api/testimonials",
                                      json={"name": "Ann", "email": "a@x.com", "rating": 0,
                                            "consent_public": "1"})
    assert r.status_code == 400


def test_no_points_credited(monkeypatch, tmp_path):
    """The ungated testimonial path never credits store-credit points."""
    appmod = _reload_app(monkeypatch, tmp_path)
    _pass_scorer(monkeypatch)
    c = appmod.app.test_client()
    j = c.post("/api/testimonials",
               json={"name": "Ann", "email": "ann@x.com", "rating": 5,
                     "body": "great", "consent_public": "1"}).get_json()
    assert "points_awarded" not in j or j["points_awarded"] == 0
    from dashboard import points
    with sqlite3.connect(appmod.LOG_DB) as cx:
        points.init_points_table(cx)  # endpoint never touches points; table proves no credit
        assert points.balance(cx, "ann@x.com") == 0


def test_compliance_result_recorded(monkeypatch, tmp_path):
    """A non-compliant body is still stored (manual moderation) with the AI verdict recorded."""
    appmod = _reload_app(monkeypatch, tmp_path)
    _pass_scorer(monkeypatch, compliance_ok=False)
    c = appmod.app.test_client()
    j = c.post("/api/testimonials",
               json={"name": "Ann", "email": "ann@x.com", "rating": 5,
                     "body": "This cured my disease!", "consent_public": "1"}).get_json()
    assert j["ok"] is True
    from dashboard import product_reviews as _pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = _pr.get_review(cx, j["review_id"])
    assert row["ai_recommend_publish"] == 0
    assert row["status"] == "pending"


def test_practitioner_attribution_from_token(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    _pass_scorer(monkeypatch)
    tok = appmod._testimonial_token_mint(99)
    assert appmod._testimonial_practitioner_id(tok) == 99
    assert appmod._testimonial_practitioner_id("garbage") == 0
    c = appmod.app.test_client()
    j = c.post("/api/testimonials",
               json={"name": "Ann", "email": "ann@x.com", "rating": 5,
                     "body": "great", "consent_public": "1", "p": tok}).get_json()
    from dashboard import product_reviews as _pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = _pr.get_review(cx, j["review_id"])
    assert row["practitioner_id"] == 99


def test_console_list_labels_testimonial(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""  # pass-through when unset
    from dashboard import product_reviews as _pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _pr.upsert_review(cx, "_results", "a@x.com", "Ann", 5, body="b",
                          kind="testimonial", consent_public=1)
    body = appmod.app.test_client().get("/api/console/reviews").get_json()
    row = next((r for r in body["pending"] if r["email"] == "a@x.com"), None)
    assert row is not None and row["product_name"] == "Testimonial"
    assert row["consent_public"] == 1


def test_source_tag_captured_from_query_and_listed(monkeypatch, tmp_path):
    """?tag= on the submission is stored and surfaced in the console queue (cohort grading)."""
    appmod = _reload_app(monkeypatch, tmp_path)
    _pass_scorer(monkeypatch)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""
    c = appmod.app.test_client()
    j = c.post("/api/testimonials",
               json={"name": "Ann", "email": "ann@x.com", "rating": 5,
                     "body": "the course changed how I see healing",
                     "consent_public": "1", "tag": "ash-cert-l1"}).get_json()
    from dashboard import product_reviews as _pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert _pr.get_review(cx, j["review_id"])["source_tag"] == "ash-cert-l1"
    listed = c.get("/api/console/reviews").get_json()["pending"]
    row = next((r for r in listed if r["email"] == "ann@x.com"), None)
    assert row is not None and row["source_tag"] == "ash-cert-l1"
