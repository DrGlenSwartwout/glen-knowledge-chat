import sqlite3
import app as appmod


def _db(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    # auth_tokens table (mirror the app's schema essentials)
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT PRIMARY KEY, email TEXT, "
               "purpose TEXT, extra TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
    cx.commit(); cx.close()
    return db


def test_gift_note_token_roundtrip(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    tok = appmod._mint_gift_note_link("B@x.com", order_ref="o1")
    out = appmod._validate_gift_note_link(tok)
    assert out == {"email": "b@x.com", "order_ref": "o1"}


def test_gift_note_token_consumed_is_rejected(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    tok = appmod._mint_gift_note_link("b@x.com", order_ref="o1")
    appmod._consume_gift_note_token(tok)
    assert appmod._validate_gift_note_link(tok) is None


def test_gift_note_token_bad_is_none(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    assert appmod._validate_gift_note_link("not-a-real-token") is None


def test_gift_note_token_rejects_other_purpose(monkeypatch, tmp_path):
    """Prove a token minted under a different purpose cannot validate as a gift-note token."""
    import json
    _db(monkeypatch, tmp_path)
    # mint a token row under a different purpose with the SAME hashing scheme
    plain = "someplaintexttoken123"
    th = appmod._hash_token(plain)
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
               "VALUES (?,?,?,?,?,?)",
               (th, "x@x.com", "membership_magic_link", json.dumps({"order_ref": "o1"}),
                "2026-01-01T00:00:00Z", "2099-01-01T00:00:00Z"))
    cx.commit(); cx.close()
    assert appmod._validate_gift_note_link(plain) is None   # wrong purpose -> rejected


def test_gift_note_token_expired_is_none(monkeypatch, tmp_path):
    """A token minted with ttl_min=0 (already expired) must validate to None."""
    _db(monkeypatch, tmp_path)
    tok = appmod._mint_gift_note_link("b@x.com", order_ref="o1", ttl_min=0)
    assert appmod._validate_gift_note_link(tok) is None


from dashboard import referrals, points  # noqa


def _client(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "PAY_IT_FORWARD_ENABLED", True)
    # avoid a live LLM call: stub the scorer to a compliant result
    from dashboard import review_scoring as rs
    monkeypatch.setattr(rs, "score_review", lambda *a, **k: {
        "compliance_ok": True, "reasons": "", "quality_points": 3,
        "recommend_publish": False, "compliance_score": 8,
        "publication_score": 5, "authenticity_score": 7, "specificity_score": 6})
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    referrals.init_tables(cx)
    # seed a redemption: owner a@x gifted referee b@x, coupon GIFT9 -> neuro-magnesium
    referrals.record_redemption(cx, "GIFT9", "a@x.com", "b@x.com", "o1")
    cx.execute("CREATE TABLE IF NOT EXISTS coupons (code TEXT PRIMARY KEY, product_slug TEXT)")
    cx.execute("INSERT INTO coupons (code, product_slug) VALUES ('GIFT9','neuro-magnesium')")
    cx.commit(); cx.close()
    return appmod.app.test_client()


def test_gift_note_submit_records_attributed_review(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    tok = appmod._mint_gift_note_link("b@x.com", order_ref="o1")
    r = c.post("/api/pif/gift-note", json={"token": tok, "name": "Bob",
                                           "body": "helping my sleep already", "consent_public": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    from dashboard import product_reviews as pr
    cx = sqlite3.connect(appmod.LOG_DB)
    row = pr.get_review(cx, body["review_id"])
    assert row["kind"] == "gift"
    assert row["product_slug"] == "neuro-magnesium"
    assert row["gift_owner_email"] == "a@x.com"
    assert row["consent_public"] == 1
    assert row["compliance_score"] == 8
    # token consumed -> second submit rejected
    r2 = c.post("/api/pif/gift-note", json={"token": tok, "name": "Bob", "body": "x", "consent_public": True})
    assert r2.status_code == 400


def test_gift_note_submit_flag_off_404(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "PAY_IT_FORWARD_ENABLED", False)
    tok = appmod._mint_gift_note_link("b@x.com", order_ref="o1")
    r = c.post("/api/pif/gift-note", json={"token": tok, "name": "Bob", "body": "x", "consent_public": True})
    assert r.status_code == 404


def test_gift_note_submit_bad_token_400(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/pif/gift-note", json={"token": "nope", "name": "Bob", "body": "x", "consent_public": True})
    assert r.status_code == 400
