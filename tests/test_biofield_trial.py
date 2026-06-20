# tests/test_biofield_trial.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals, subscriptions
    with sqlite3.connect(db) as cx:
        biofield_reveals.init_table(cx)
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
    return db


def _approved_reveal(app_module, db, email="t@x.com"):
    """Create an approved reveal + an auth_tokens biofield_reveal token; return the plaintext token."""
    import secrets as _s
    from datetime import datetime, timezone, timedelta
    from dashboard import biofield_reveals as br
    token = "tk_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        rid, _ = br.upsert(cx, email, "2026-06-20", {"greeting": "Hi", "body": "b"},
                           [{"name": "Top", "slug": "top", "meaning": "m"},
                            {"name": "Deep1", "slug": "deep1", "meaning": "m2"},
                            {"name": "Deep2", "slug": "deep2", "meaning": "m3"}], "s")
        br.set_token(cx, rid, th)
        br.approve_first(cx, rid, "glen")
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def test_unlock_checkout_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", False, raising=False)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/unlock-checkout")
    assert r.get_json().get("ok") is False


def test_unlock_checkout_creates_dollar_session(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    captured = {}
    from dashboard import stripe_pay
    def _fake(amount_cents, **kw):
        captured["amount"] = amount_cents; captured.update(kw)
        return {"id": "cs_1", "url": "https://stripe.test/cs_1"}
    monkeypatch.setattr(stripe_pay, "create_checkout_session", _fake)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/unlock-checkout")
    body = r.get_json()
    assert body["ok"] is True and body["url"] == "https://stripe.test/cs_1"
    assert captured["amount"] == 100 and captured["save_card"] is True
    assert captured["metadata"]["kind"] == "biofield_trial" and captured["metadata"]["email"] == "t@x.com"


def test_unlock_checkout_already_member(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().post(f"/begin/biofield/{token}/unlock-checkout")
    assert r.get_json() == {"ok": True, "already": True}


def _mock_paid_session(app_module, monkeypatch, email="t@x.com", sid="cs_1"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "biofield_trial", "email": email}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def test_return_creates_membership_and_grant(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_session(app_module, monkeypatch)
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subs = cx.execute("SELECT amount_cents, status, kind FROM subscriptions WHERE email='t@x.com'").fetchall()
        grants = cx.execute("SELECT source FROM memberships WHERE email='t@x.com'").fetchall()
    assert len(subs) == 1 and subs[0] == (9900, "active", "membership")
    assert len(grants) == 1 and grants[0][0] == "biofield_trial"
    assert app_module._active_membership_for_email("t@x.com") is not None


def test_return_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_session(app_module, monkeypatch)
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 1
        assert cx.execute("SELECT COUNT(*) FROM memberships WHERE email='t@x.com'").fetchone()[0] == 1


def test_return_unpaid_creates_nothing(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session", lambda s: {"metadata": {"kind": "biofield_trial", "email": "t@x.com"}, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent", lambda pi: {"customer": "", "payment_method": "", "status": "requires_payment_method"})
    app_module.app.test_client().get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Task 3: paid member gets full remedies; non-paid gets blurred + no deep content
# ---------------------------------------------------------------------------

import json as _json


def _extract_reveal(html_bytes):
    """Parse window.__REVEAL__ from the served HTML bytes."""
    import re
    m = re.search(rb"window\.__REVEAL__\s*=\s*(\{.*?\});", html_bytes, re.DOTALL)
    assert m, "No __REVEAL__ found in HTML"
    raw = m.group(1).decode("utf-8")
    # Unescape the JSON-safe escapes applied by the route
    raw = raw.replace("\\u003c", "<").replace("\\u003e", ">").replace("\\u0026", "&")
    return _json.loads(raw)


def test_paid_member_gets_full_remedies(monkeypatch, tmp_path):
    """A paid member's reveal page must include Deep1 and Deep2 names, blurred_count=0, paid=True."""
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "is_member", lambda **kw: True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"status": "active"})
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    html = r.data
    # Deep remedy names must appear in the rendered HTML
    assert b"Deep1" in html, "Deep1 missing from paid-member HTML"
    assert b"Deep2" in html, "Deep2 missing from paid-member HTML"
    reveal = _extract_reveal(html)
    assert reveal.get("blurred_count") == 0, f"Expected blurred_count=0, got {reveal.get('blurred_count')}"
    assert reveal.get("paid") is True, f"Expected paid=True, got {reveal.get('paid')}"
    assert reveal.get("trial_enabled") is True, f"Expected trial_enabled=True"
    # remedies list must contain all 3 entries (Top + Deep1 + Deep2)
    remedies = reveal.get("remedies") or []
    names = [r_["name"] for r_ in remedies]
    assert "Deep1" in names and "Deep2" in names and "Top" in names, f"Unexpected remedies: {names}"


def test_nonpaid_member_no_deep_content(monkeypatch, tmp_path):
    """A non-paid member must NOT get Deep1/Deep2 in the HTML; blurred_count > 0; paid=False."""
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "is_member", lambda **kw: True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    token = _approved_reveal(app_module, db)
    r = app_module.app.test_client().get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    html = r.data
    # Deep remedy names must NOT appear (anti-bypass)
    assert b"Deep1" not in html, "Deep1 leaked in non-paid HTML"
    assert b"Deep2" not in html, "Deep2 leaked in non-paid HTML"
    reveal = _extract_reveal(html)
    assert reveal.get("blurred_count", 0) > 0, f"Expected blurred_count>0, got {reveal.get('blurred_count')}"
    assert reveal.get("paid") is False, f"Expected paid=False, got {reveal.get('paid')}"
    assert reveal.get("trial_enabled") is True, f"Expected trial_enabled=True"
    # No 'remedies' key with deep content
    remedies = reveal.get("remedies") or []
    rem_names = [r_["name"] for r_ in remedies]
    assert "Deep1" not in rem_names and "Deep2" not in rem_names, f"Deep remedies leaked: {rem_names}"
