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
        subscriptions.migrate_add_term_cap_column(cx)
        cx.execute(
            "CREATE TABLE IF NOT EXISTS auth_tokens "
            "(token_hash TEXT PRIMARY KEY, email TEXT NOT NULL, purpose TEXT NOT NULL, "
            "extra TEXT, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, consumed_at TEXT)"
        )
        cx.commit()
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


def test_return_creates_grant_no_subscription(monkeypatch, tmp_path):
    """Model #2: the $1 deposit unlocks a day-based grant (Biofield Analysis unlocked)
    but creates NO subscription row -> nothing auto-charges. The member DISCOUNT is
    withheld during preview (category 'trial', _is_paid_member False)."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_session(app_module, monkeypatch)
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    with sqlite3.connect(db) as cx:
        subs = cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0]
        grants = cx.execute("SELECT source FROM memberships WHERE email='t@x.com'").fetchall()
    assert subs == 0, "Model #2: no subscription row -> nothing can auto-charge"
    assert len(grants) == 1 and grants[0][0] == "biofield_trial"
    # Analysis unlocked (active grant) but discount withheld (preview = 'trial').
    assert app_module._active_membership_for_email("t@x.com") is not None
    assert app_module.membership_category("t@x.com") == "trial"
    assert app_module._is_paid_member("t@x.com") is False


def test_return_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_session(app_module, monkeypatch)
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='t@x.com'").fetchone()[0] == 0
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


# ---------------------------------------------------------------------------
# Task 4: one-click cancel via tokened /membership/cancel/<token>
# ---------------------------------------------------------------------------


def _seed_cancel_token(app_module, db, email="t@x.com"):
    """Insert an active membership subscription + a membership_cancel auth_token.
    Returns the plaintext cancel token."""
    from datetime import datetime, timezone, timedelta
    from dashboard import subscriptions
    cancel_tok = "ct_" + __import__("secrets").token_urlsafe(16)
    th = app_module._hash_token(cancel_tok)
    with sqlite3.connect(db) as cx:
        # Ensure auth_tokens table exists
        cx.execute(
            "CREATE TABLE IF NOT EXISTS auth_tokens "
            "(token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        # Create an active membership subscription
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        subscriptions.migrate_add_term_cap_column(cx)
        subscriptions.create_membership(
            cx, email=email, stripe_customer_id="cus_test",
            stripe_payment_method_id="pm_test", amount_cents=9900,
            next_charge_date="2026-07-20")
        # Insert the cancel token
        now = datetime.now(timezone.utc)
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
            "VALUES (?,?,?,?,?)",
            (th, email, "membership_cancel", now.isoformat(),
             (now + timedelta(days=60)).isoformat()))
        cx.commit()
    return cancel_tok


def test_cancel_valid_token_cancels_subscription(monkeypatch, tmp_path):
    """A valid membership_cancel token cancels the active subscription."""
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    cancel_tok = _seed_cancel_token(app_module, db)
    r = app_module.app.test_client().get(f"/membership/cancel/{cancel_tok}")
    assert r.status_code == 200
    with sqlite3.connect(db) as cx:
        status = cx.execute(
            "SELECT status FROM subscriptions WHERE email='t@x.com' AND kind='membership'"
        ).fetchone()
    assert status is not None and status[0] == "cancelled", f"Expected cancelled, got {status}"


def test_cancel_invalid_token_friendly_no_change(monkeypatch, tmp_path):
    """An invalid token returns a friendly 200 page; the subscription stays active."""
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    _seed_cancel_token(app_module, db)
    r = app_module.app.test_client().get("/membership/cancel/notarealtoken")
    assert r.status_code == 200
    with sqlite3.connect(db) as cx:
        status = cx.execute(
            "SELECT status FROM subscriptions WHERE email='t@x.com' AND kind='membership'"
        ).fetchone()
    assert status is not None and status[0] == "active", f"Expected active, got {status}"


def test_cancel_idempotent(monkeypatch, tmp_path):
    """Calling cancel twice leaves the subscription cancelled with no error."""
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    cancel_tok = _seed_cancel_token(app_module, db)
    c = app_module.app.test_client()
    r1 = c.get(f"/membership/cancel/{cancel_tok}")
    r2 = c.get(f"/membership/cancel/{cancel_tok}")
    assert r1.status_code == 200 and r2.status_code == 200
    with sqlite3.connect(db) as cx:
        status = cx.execute(
            "SELECT status FROM subscriptions WHERE email='t@x.com' AND kind='membership'"
        ).fetchone()
    assert status is not None and status[0] == "cancelled", f"Expected cancelled, got {status}"


# ---------------------------------------------------------------------------
# Task 5: Reveal page scaffold assertions (CTA wiring + paid render)
# ---------------------------------------------------------------------------


def test_reveal_page_scaffold_unlock_checkout(monkeypatch, tmp_path):
    """The served begin-biofield.html must contain 'unlock-checkout', 'paid', and
    'trial_enabled' -- confirming the JS CTA wiring and paid-branch are present."""
    app_module = _load_app()
    _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    # Any GET serves the same static HTML (invalid token -> null reveal is fine)
    r = client.get("/begin/biofield/any-token")
    assert r.status_code == 200
    html = r.data.decode()
    assert "unlock-checkout" in html, "'unlock-checkout' not found in begin-biofield.html"
    assert "paid" in html, "'paid' not found in begin-biofield.html"
    assert "trial_enabled" in html, "'trial_enabled' not found in begin-biofield.html"
