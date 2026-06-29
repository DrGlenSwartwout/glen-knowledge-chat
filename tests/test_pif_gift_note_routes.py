import sqlite3
from datetime import datetime, timedelta

import app as appmod
from dashboard import referrals, points  # noqa


def _ago(days):
    """Return an ISO-format datetime string `days` ago (UTC)."""
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


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


def test_gift_note_rejects_cross_attribution(monkeypatch, tmp_path):
    """A token minted for attacker@x.com cannot be used against order o1 whose
    referee is b@x.com — different email means the route returns 400."""
    c = _client(monkeypatch, tmp_path)  # seeds redemption o1 with referee b@x.com
    # mint a token for a DIFFERENT recipient but pointing at o1 (whose referee is b@x.com)
    tok = appmod._mint_gift_note_link("attacker@x.com", order_ref="o1")
    r = c.post("/api/pif/gift-note", json={"token": tok, "name": "X",
                                           "body": "x", "consent_public": True})
    assert r.status_code == 400


def test_invite_cron_sends_and_marks(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)  # seeds redemption o1 (created now) — make it old:
    cx = sqlite3.connect(appmod.LOG_DB)
    from dashboard import pif_gift_notes as gn
    gn.ensure_columns(cx)
    # 20 days ago: past the 14-day delay, within the 60-day max_age window
    cx.execute("UPDATE referral_redemptions SET created_at=? WHERE order_ref='o1'", (_ago(20),))
    cx.commit(); cx.close()
    sent = []
    monkeypatch.setattr(appmod, "_send_inquiry_email",
                        lambda **k: sent.append(k) or True)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "testsecret")
    monkeypatch.setattr(appmod.dashboard, "CONSOLE_SECRET", "testsecret")
    r = c.post("/api/cron/pif-gift-note-invites?key=testsecret")
    assert r.status_code == 200
    assert r.get_json()["invited"] == 1
    assert len(sent) == 1 and sent[0]["to_email"] == "b@x.com"
    # idempotent: second run sends nothing
    r2 = c.post("/api/cron/pif-gift-note-invites?key=testsecret")
    assert r2.get_json()["invited"] == 0


def test_invite_cron_dry_run_sends_nothing(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    cx = sqlite3.connect(appmod.LOG_DB)
    from dashboard import pif_gift_notes as gn
    gn.ensure_columns(cx)
    # 20 days ago: past the 14-day delay, within the 60-day max_age window
    cx.execute("UPDATE referral_redemptions SET created_at=? WHERE order_ref='o1'", (_ago(20),))
    cx.commit(); cx.close()
    sent = []
    monkeypatch.setattr(appmod, "_send_inquiry_email", lambda **k: sent.append(k) or True)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "testsecret")
    monkeypatch.setattr(appmod.dashboard, "CONSOLE_SECRET", "testsecret")
    r = c.post("/api/cron/pif-gift-note-invites?key=testsecret&dry_run=1")
    assert r.status_code == 200 and r.get_json()["dry_run"] is True
    assert sent == []
    # still pending (not marked) after dry run
    r2 = c.post("/api/cron/pif-gift-note-invites?key=testsecret")
    assert r2.get_json()["invited"] == 1


def test_gift_note_no_coupon_uses_gift_slug(monkeypatch, tmp_path):
    """When the coupon code has no coupons row, _product_for_code returns '' and the
    review must be written with product_slug '_gift' (not '_results') to avoid colliding
    with the recipient's own testimonial stored under the _results reserved slug.
    Uses a distinct recipient (c@x.com) to avoid the referee_email PK collision
    with the b@x.com redemption already seeded by _client()."""
    c = _client(monkeypatch, tmp_path)
    # seed a redemption for a NEW recipient with a code NOT in the coupons table
    cx = sqlite3.connect(appmod.LOG_DB)
    referrals.record_redemption(cx, "NOCODE", "a@x.com", "c@x.com", "o_nocoupon")
    cx.commit(); cx.close()
    tok = appmod._mint_gift_note_link("c@x.com", order_ref="o_nocoupon")
    r = c.post("/api/pif/gift-note", json={"token": tok, "name": "Carol",
                                           "body": "made a real difference", "consent_public": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    from dashboard import product_reviews as pr
    cx = sqlite3.connect(appmod.LOG_DB)
    row = pr.get_review(cx, body["review_id"])
    assert row["product_slug"] == "_gift"   # NOT "_results"
    assert row["kind"] == "gift"
    assert row["gift_owner_email"] == "a@x.com"
