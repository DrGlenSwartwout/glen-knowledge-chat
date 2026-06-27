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
    uid = "fb6f2f66-28ab-4c11-9d3e-deadbeef9999"  # Supabase UUID
    tok = appmod._testimonial_token_mint(uid)
    assert appmod._testimonial_practitioner_id(tok) == uid
    assert appmod._testimonial_practitioner_id("garbage") == ""
    c = appmod.app.test_client()
    j = c.post("/api/testimonials",
               json={"name": "Ann", "email": "ann@x.com", "rating": 5,
                     "body": "great", "consent_public": "1", "p": tok}).get_json()
    from dashboard import product_reviews as _pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = _pr.get_review(cx, j["review_id"])
    assert row["practitioner_id"] == uid


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


# ── Cert-cohort: approving a tagged testimonial grants Level 1 credit ──────────

def _capture_grants(monkeypatch):
    from dashboard import reviews_actions as ra
    calls = []
    monkeypatch.setattr(ra, "_grant_cert", lambda email, level: calls.append((email, level)) or True)
    return calls


def test_cert_cohort_approval_grants_level1(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    calls = _capture_grants(monkeypatch)
    from dashboard import product_reviews as pr, dispatch as d
    from dashboard.rbac import Actor, OWNER
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "_results", "student@x.com", "Stu", 5,
                               body="the course was the most meaningful thing",
                               kind="testimonial", consent_public=1, source_tag="ash-cert-l1")
        res = d.dispatch_action(cx, "reviews.approve", {"id": rid},
                                Actor(role=OWNER, name="Glen"), source="panel")
    assert res["status"] == "done"
    assert calls == [("student@x.com", 1)]


def test_untagged_or_product_approval_grants_nothing(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    calls = _capture_grants(monkeypatch)
    from dashboard import product_reviews as pr, dispatch as d
    from dashboard.rbac import Actor, OWNER
    with sqlite3.connect(appmod.LOG_DB) as cx:
        # untagged testimonial → no grant
        r1 = pr.upsert_review(cx, "_results", "a@x.com", "A", 5, body="b",
                              kind="testimonial", consent_public=1)
        d.dispatch_action(cx, "reviews.approve", {"id": r1},
                          Actor(role=OWNER, name="Glen"), source="panel")
        # a PRODUCT review that happens to carry the tag → no grant (kind gate)
        r2 = pr.upsert_review(cx, "longevity", "b@x.com", "B", 5, body="b",
                              source_tag="ash-cert-l1")
        d.dispatch_action(cx, "reviews.approve", {"id": r2},
                          Actor(role=OWNER, name="Glen"), source="panel")
    assert calls == []


def test_dimension_scores_stored_and_listed(monkeypatch, tmp_path):
    """A scored testimonial stores the 4 dimensions and the console queue surfaces them."""
    appmod = _reload_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_review", lambda *a, **k: {
        "compliance_ok": True, "reasons": "ok", "quality_points": 2, "recommend_publish": True,
        "compliance_score": 9, "publication_score": 7, "authenticity_score": 8, "specificity_score": 6})
    c = appmod.app.test_client()
    j = c.post("/api/testimonials",
               json={"name": "Ann", "email": "ann@x.com", "rating": 5,
                     "body": "specific authentic story", "consent_public": "1"}).get_json()
    from dashboard import product_reviews as _pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = _pr.get_review(cx, j["review_id"])
    assert (row["compliance_score"], row["publication_score"],
            row["authenticity_score"], row["specificity_score"]) == (9, 7, 8, 6)
    listed = c.get("/api/console/reviews").get_json()["pending"]
    r = next((x for x in listed if x["email"] == "ann@x.com"), None)
    assert r and r["compliance_score"] == 9 and r["specificity_score"] == 6


def test_cohort_submission_lookup_by_email_and_pid():
    cx = sqlite3.connect(":memory:")
    from dashboard import product_reviews as pr
    uid = "22493df4-uuid-pid"  # Supabase UUID, not an int
    assert pr.cohort_submission(cx, "ash-cert-l1", email="a@x.com") is None
    pr.upsert_review(cx, "_results", "a@x.com", "A", 5, body="b",
                     kind="testimonial", source_tag="ash-cert-l1", practitioner_id=uid)
    assert pr.cohort_submission(cx, "ash-cert-l1", email="a@x.com")["email"] == "a@x.com"
    assert pr.cohort_submission(cx, "ash-cert-l1", practitioner_id=uid)["practitioner_id"] == uid
    # wrong tag / wrong person -> None
    assert pr.cohort_submission(cx, "other-tag", email="a@x.com") is None
    assert pr.cohort_submission(cx, "ash-cert-l1", email="z@x.com") is None


def test_assignment_status_pure(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    assert appmod._assignment_status(0, False) == "not_started"
    assert appmod._assignment_status(0, True) == "in_review"
    assert appmod._assignment_status(1, True) == "complete"
    assert appmod._assignment_status(3, False) == "complete"  # already past L1


def test_testimonial_token_for_practitioner_get_or_mint(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    uid = "22493df4-7a3e-4a15-a8e2-0a4334dca8b8"  # Supabase UUID — NOT an int
    t1 = appmod._testimonial_token_for_practitioner(uid)
    t2 = appmod._testimonial_token_for_practitioner(uid)  # stable: reuse, don't re-mint
    assert t1 and t1 == t2
    assert appmod._testimonial_practitioner_id(t1) == uid  # round-trips the UUID string
    assert appmod._testimonial_token_for_practitioner(0) == ""   # no pid -> no token
    assert appmod._testimonial_token_for_practitioner("") == ""


def test_cert_l1_assignment_block(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    uid = "eaf8ccf5-ea00-4b9c-9f0e-deadbeef0001"  # UUID pid must not crash int()
    a = appmod._cert_l1_assignment(uid, "stu@x.com", 0)
    assert a["status"] == "not_started"
    assert "tag=ash-cert-l1" in a["record_url"] and "/results" in a["record_url"]
    assert "p=" in a["record_url"]  # auto-attributed
    # already at level 1 -> complete
    assert appmod._cert_l1_assignment(uid, "stu@x.com", 1)["status"] == "complete"


def test_notify_cohort_endpoint_dry_run_then_send(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "")  # pass-through (_console_key_ok)
    from dashboard import practitioner_admin as pa, cert_notify as cn
    monkeypatch.setattr(pa, "list_practitioners", lambda *a, **k: [
        {"id": "uuid-coach", "email": "coach@x.com", "name": "Coach", "portal_role": "coach", "modules_completed": 0},
        {"id": "uuid-org", "email": "org@x.com", "name": "Org", "portal_role": "org_member", "modules_completed": 0},
        {"id": "uuid-glen", "email": "drglenswartwout@gmail.com", "name": "Glen", "portal_role": "coach", "modules_completed": 0},
    ])
    sent = []
    monkeypatch.setattr(cn, "send_assignment_notice",
                        lambda email, name, url, **k: sent.append(email) or True)
    c = appmod.app.test_client()
    dry = c.post("/api/console/cert/notify-cohort?dry_run=1").get_json()
    # coaches only, AND the owner's own coach record is excluded
    assert dry["count"] == 1 and dry["recipients"][0]["email"] == "coach@x.com"
    assert sent == []  # dry run sends nothing
    res = c.post("/api/console/cert/notify-cohort").get_json()
    assert res["count"] == 1 and sent == ["coach@x.com"]


def test_review_fires_feedback_ready_email(monkeypatch, tmp_path):
    """Approving/rejecting a cert-tagged testimonial auto-sends the feedback-ready email."""
    appmod = _reload_app(monkeypatch, tmp_path)
    from dashboard import cert_notify as cn
    calls = []
    monkeypatch.setattr(cn, "send_feedback_ready",
                        lambda email, name, outcome, **k: calls.append((email, outcome)) or True)
    from dashboard import product_reviews as pr, dispatch as d
    from dashboard.rbac import Actor, OWNER
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "_results", "stu@x.com", "Stu", 5, body="b",
                               kind="testimonial", source_tag="ash-cert-l1", practitioner_id=7)
        d.dispatch_action(cx, "reviews.approve", {"id": rid}, Actor(role=OWNER, name="Glen"), source="panel")
        # an untagged testimonial does NOT email
        r2 = pr.upsert_review(cx, "_results", "b@x.com", "B", 5, body="b", kind="testimonial")
        d.dispatch_action(cx, "reviews.reject", {"id": r2}, Actor(role=OWNER, name="Glen"), source="panel")
    assert calls == [("stu@x.com", "approved")]


def test_cert_feedback_hidden_until_reviewed(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    assert appmod._cert_feedback(None) is None
    assert appmod._cert_feedback({"status": "pending", "compliance_score": 9}) is None


def test_cert_feedback_approved_outcome_and_levels(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    sub = {"status": "approved", "compliance_score": 9, "publication_score": 7,
           "authenticity_score": 8, "specificity_score": 5, "audio_quality": 6, "visual_quality": 0}
    fb = appmod._cert_feedback(sub)
    assert fb["outcome"] == "approved"
    by = {s["label"]: s for s in fb["scores"]}
    # only scored (>0) dims appear -> Visual (0) excluded
    assert "Visual" not in by
    assert by["Compliance"]["level"] == "strong"      # 9 >= 8
    assert by["Publication"]["level"] == "ok"         # 7 in [6,8)
    assert by["Specificity"]["level"] == "low"        # 5 < 6
    assert by["Audio"]["level"] == "ok"               # 6


def test_cert_feedback_rejected_outcome(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    fb = appmod._cert_feedback({"status": "rejected", "compliance_score": 4})
    assert fb["outcome"] == "refine"
    comp = next(s for s in fb["scores"] if s["label"] == "Compliance")
    assert comp["level"] == "low"                     # 4 < 8 compliance bar


def test_assignment_block_includes_feedback_when_reviewed(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    from dashboard import product_reviews as pr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "_results", "stu@x.com", "Stu", 5, body="b",
                               kind="testimonial", source_tag="ash-cert-l1", practitioner_id=77)
        pr.set_scores(cx, rid, compliance=9, publication=7, authenticity=8, specificity=6)
        pr.set_status(cx, rid, "approved", by="Glen")
    a = appmod._cert_l1_assignment(77, "stu@x.com", 1)
    assert a.get("feedback") and a["feedback"]["outcome"] == "approved"


def test_set_quality_action_sets_audio_visual(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    from dashboard import product_reviews as pr, dispatch as d
    from dashboard.rbac import Actor, OWNER
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rid = pr.upsert_review(cx, "_results", "a@x.com", "A", 5, body="b", kind="testimonial")
        res = d.dispatch_action(cx, "reviews.set_quality", {"id": rid, "audio": 9, "visual": 7},
                                Actor(role=OWNER, name="Glen"), source="panel")
        assert res["status"] == "done"
        r = pr.get_review(cx, rid)
    assert r["audio_quality"] == 9 and r["visual_quality"] == 7
