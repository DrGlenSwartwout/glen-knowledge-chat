# tests/test_prepay_checkout.py
"""Route tests for the prepay ladder (POST /prepay/checkout, GET /prepay/return).

Mirrors test_biofield_trial.py: tmp-file sqlite via monkeypatched LOG_DB, Stripe
faked by monkeypatching dashboard.stripe_pay.
"""
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
    monkeypatch.setattr(app_module, "PUBLIC_BASE_URL", "https://test.local", raising=False)
    from dashboard import subscriptions
    with sqlite3.connect(db) as cx:
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        app_module.init_membership_tables(cx)
        cx.commit()
    # Keep the return path's best-effort side-effects out of the test.
    monkeypatch.setattr(app_module, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: None, raising=False)
    return db


# ---------------------------------------------------------------------------
# POST /prepay/checkout
# ---------------------------------------------------------------------------

def test_checkout_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", False, raising=False)
    r = app_module.app.test_client().post("/prepay/checkout", json={"email": "a@b.com", "tier_key": "6mo"})
    assert r.get_json().get("ok") is False


def test_checkout_creates_session_at_tier_price(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    captured = {}
    from dashboard import stripe_pay
    def _fake(amount_cents, **kw):
        captured["amount"] = amount_cents; captured.update(kw)
        return {"id": "cs_1", "url": "https://stripe.test/cs_1"}
    monkeypatch.setattr(stripe_pay, "create_checkout_session", _fake)
    r = app_module.app.test_client().post("/prepay/checkout", json={"email": "A@b.com", "tier_key": "12mo"})
    body = r.get_json()
    assert body["ok"] is True and body["url"] == "https://stripe.test/cs_1"
    assert captured["amount"] == 99000
    assert captured["save_card"] is False
    assert captured["metadata"]["kind"] == "prepay_term"
    assert captured["metadata"]["tier_key"] == "12mo"
    assert captured["metadata"]["email"] == "a@b.com"


def test_checkout_rejects_unknown_tier(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    r = app_module.app.test_client().post("/prepay/checkout", json={"email": "a@b.com", "tier_key": "9mo"})
    assert r.get_json().get("ok") is False


# ---------------------------------------------------------------------------
# GET /prepay/return
# ---------------------------------------------------------------------------

def _mock_paid_prepay_session(app_module, monkeypatch, email="a@b.com", tier_key="6mo"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "prepay_term", "email": email, "tier_key": tier_key},
                   "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})


def test_return_grants_daybased_term_no_subscription(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_prepay_session(app_module, monkeypatch, tier_key="6mo")
    app_module.app.test_client().get("/prepay/return?session_id=cs_1")
    with sqlite3.connect(db) as cx:
        grants = cx.execute("SELECT source FROM memberships WHERE email='a@b.com'").fetchall()
        subs = cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='a@b.com'").fetchone()[0]
    assert len(grants) == 1 and grants[0][0] == "prepay_6mo"
    assert subs == 0, "prepay term must NOT create a subscriptions row (no auto-renew)"
    # Grant-only member is a paid member for the term -> gets member pricing.
    assert app_module._is_paid_member("a@b.com") is True


def test_return_term_length_matches_tier(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_prepay_session(app_module, monkeypatch, tier_key="12mo")
    app_module.app.test_client().get("/prepay/return?session_id=cs_1")
    from datetime import datetime
    with sqlite3.connect(db) as cx:
        row = cx.execute(
            "SELECT granted_at, expires_at FROM memberships WHERE email='a@b.com'").fetchone()
    granted = datetime.fromisoformat(row[0].rstrip("Z"))
    expires = datetime.fromisoformat(row[1].rstrip("Z"))
    days = (expires - granted).days
    # ~12 months out; 365 or 366 depending on the calendar span.
    assert 364 <= days <= 366, f"expected ~1yr term, got {days} days"


def test_return_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_prepay_session(app_module, monkeypatch, tier_key="6mo")
    c = app_module.app.test_client()
    c.get("/prepay/return?session_id=cs_1")
    c.get("/prepay/return?session_id=cs_1")
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM memberships WHERE email='a@b.com'").fetchone()[0]
    assert n == 1, "replay must not double-grant"


# ---------------------------------------------------------------------------
# Picker page + tiers endpoint
# ---------------------------------------------------------------------------

def test_tiers_endpoint_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", False, raising=False)
    r = app_module.app.test_client().get("/api/prepay/tiers")
    assert r.get_json().get("ok") is False


def test_tiers_endpoint_returns_four_rungs(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    body = app_module.app.test_client().get("/api/prepay/tiers").get_json()
    assert body["ok"] is True
    keys = [t["key"] for t in body["tiers"]]
    assert keys == ["1mo", "3mo", "6mo", "12mo"]
    six = next(t for t in body["tiers"] if t["key"] == "6mo")
    assert six["per_month_cents"] == 9100 and six["savings_pct"] == 8 and six["badge"] == "Most Popular"


def test_prepay_page_flag_off_redirects(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", False, raising=False)
    r = app_module.app.test_client().get("/prepay")
    assert r.status_code in (301, 302)


def test_prepay_page_scaffold(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    r = app_module.app.test_client().get("/prepay")
    assert r.status_code == 200
    html = r.data.decode()
    assert "/prepay/checkout" in html
    assert "/api/prepay/tiers" in html


def _grant(cx, app_module, email, source, days=90):
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    cx.execute(
        "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source, truly_vip_ref, notes) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (source, email, now.isoformat() + "Z", (now + timedelta(days=days)).isoformat() + "Z",
         source, source, "", ""))
    cx.commit()


def test_pure_deposit_buyer_withholds_discount(monkeypatch, tmp_path):
    """A $1 deposit grant alone => 'trial' => discount withheld (analysis still unlocked)."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        _grant(cx, app_module, "d@x.com", "biofield_trial")
    assert app_module.membership_category("d@x.com") == "trial"
    assert app_module._is_paid_member("d@x.com") is False


def test_deposit_buyer_who_prepays_gets_member_pricing(monkeypatch, tmp_path):
    """Regression (sticky-trial): once a deposit buyer converts via a prepay term, the
    lingering biofield_trial grant must NOT keep withholding the member discount."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        _grant(cx, app_module, "d@x.com", "biofield_trial")
        _grant(cx, app_module, "d@x.com", "prepay_6mo")
    assert app_module.membership_category("d@x.com") != "trial"
    assert app_module._is_paid_member("d@x.com") is True


def test_founding_grant_member_stays_paid(monkeypatch, tmp_path):
    """A pure founding/comp grant (never deposited) is a paid member (discount kept)."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        _grant(cx, app_module, "f@x.com", "founding")
    assert app_module._is_paid_member("f@x.com") is True


def _post_webhook(app_module, monkeypatch, sid="cs_1"):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    import json as _json
    payload = _json.dumps({"type": "checkout.session.completed", "data": {"object": {"id": sid}}})
    return app_module.app.test_client().post("/webhook/stripe", data=payload,
                                             content_type="application/json")


def test_webhook_fulfills_prepay_closed_tab(monkeypatch, tmp_path):
    """Safety net: the Stripe webhook grants a prepaid term even when the browser
    never lands on /prepay/return (closed tab / dropped redirect)."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_prepay_session(app_module, monkeypatch, tier_key="6mo")
    r = _post_webhook(app_module, monkeypatch)
    assert r.status_code == 200
    with sqlite3.connect(db) as cx:
        grants = cx.execute("SELECT source FROM memberships WHERE email='a@b.com'").fetchall()
        subs = cx.execute("SELECT COUNT(*) FROM subscriptions WHERE email='a@b.com'").fetchone()[0]
    assert len(grants) == 1 and grants[0][0] == "prepay_6mo"
    assert subs == 0


def test_webhook_and_return_single_grant(monkeypatch, tmp_path):
    """Redirect + webhook both firing (any order) must grant exactly once."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_paid_prepay_session(app_module, monkeypatch, tier_key="12mo")
    _post_webhook(app_module, monkeypatch)
    app_module.app.test_client().get("/prepay/return?session_id=cs_1")
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM memberships WHERE email='a@b.com'").fetchone()[0]
    assert n == 1


def test_return_unpaid_grants_nothing(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PREPAY_LADDER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda s: {"metadata": {"kind": "prepay_term", "email": "a@b.com", "tier_key": "6mo"},
                   "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi: {"customer": "", "payment_method": "", "status": "requires_payment_method"})
    app_module.app.test_client().get("/prepay/return?session_id=cs_1")
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM memberships WHERE email='a@b.com'").fetchone()[0]
    assert n == 0
