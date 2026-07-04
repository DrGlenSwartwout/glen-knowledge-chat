# tests/test_program_front_door.py
"""Task 2 of Program -> deposit front door: a paid program (biofield) purchase grants a
30-day Continuous Care taster window behind PROGRAM_CARE_TASTER_ENABLED. Source
"care_taster" (NOT "biofield_trial") so it reads as a paid grant (member pricing kept).

Mirrors the fixture pattern from tests/test_prepay_checkout.py and drives the biofield
return the way tests/test_biofield_checkout.py does.
"""
import importlib, sqlite3, sys
from datetime import datetime
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
        subscriptions.migrate_add_term_cap_column(cx)
        subscriptions.migrate_add_attribution_column(cx)
        app_module.init_membership_tables(cx)
        cx.commit()
    # Keep the return path's best-effort side-effects out of the test.
    monkeypatch.setattr(app_module, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: None, raising=False)
    return db


def _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com",
                                 invoice_id="INVP", customer_id="C1"):
    from dashboard import stripe_pay as _sp
    monkeypatch.setattr(_sp, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid", "amount_total": 30000,
        "payment_intent": "pi_1",
        "metadata": {"kind": "biofield", "email": email, "tier": "scalable",
                     "invoice_id": invoice_id, "customer_id": customer_id,
                     "points_redeemed_cents": "0"},
    })
    monkeypatch.setattr(app_module.stripe_pay, "get_session", _sp.get_session, raising=False)
    monkeypatch.setattr(_sp, "get_payment_intent",
        lambda pi: {"customer": customer_id, "payment_method": "pm_1", "status": "succeeded"})
    monkeypatch.setattr(app_module.stripe_pay, "get_payment_intent", _sp.get_payment_intent,
                        raising=False)

    # Neutralize the other best-effort side-effects on this same branch (QBO, points,
    # biofield_store) so the test is scoped to the care-taster grant.
    from dashboard import qbo_billing as _qb
    monkeypatch.setattr(_qb, "record_payment", lambda *a, **k: None)
    monkeypatch.setattr(app_module._bos_orders, "find_order_by_external_ref",
                        lambda cx, ref: None)


def test_program_return_grants_one_care_taster_window(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com")
    r = app_module.app.test_client().get("/begin/checkout-return?session_id=cs_1")
    assert r.status_code == 302

    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        grants = cx.execute(
            "SELECT granted_at, expires_at, source FROM memberships "
            "WHERE email='buyer@x.com' AND source='care_taster'").fetchall()
    assert len(grants) == 1
    granted = datetime.fromisoformat(grants[0]["granted_at"].rstrip("Z"))
    expires = datetime.fromisoformat(grants[0]["expires_at"].rstrip("Z"))
    days = (expires - granted).days
    assert 29 <= days <= 31, f"expected ~30 day taster window, got {days} days"
    assert app_module._is_paid_member("buyer@x.com") is True


def test_program_return_care_taster_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com")
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?session_id=cs_1")
    c.get("/begin/checkout-return?session_id=cs_1")

    with sqlite3.connect(db) as cx:
        n = cx.execute(
            "SELECT COUNT(*) FROM memberships WHERE email='buyer@x.com' "
            "AND source='care_taster'").fetchone()[0]
    assert n == 1, "replay must not double-grant the care taster window"


def test_program_return_flag_off_grants_nothing(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", False, raising=False)
    _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com")
    app_module.app.test_client().get("/begin/checkout-return?session_id=cs_1")

    with sqlite3.connect(db) as cx:
        n = cx.execute(
            "SELECT COUNT(*) FROM memberships WHERE email='buyer@x.com' "
            "AND source='care_taster'").fetchone()[0]
    assert n == 0


def _post_biofield_webhook(app_module, monkeypatch, sid="cs_1"):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    import json as _json
    payload = _json.dumps({"type": "checkout.session.completed", "data": {"object": {"id": sid}}})
    return app_module.app.test_client().post("/webhook/stripe", data=payload,
                                             content_type="application/json")


def test_program_webhook_grants_care_taster_closed_tab(monkeypatch, tmp_path):
    """Safety net: the Stripe webhook grants the paid 30-day care window even when the
    browser never lands on /begin/checkout-return (closed tab / dropped redirect)."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com")
    r = _post_biofield_webhook(app_module, monkeypatch)
    assert r.status_code == 200
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM memberships WHERE email='buyer@x.com' "
                       "AND source='care_taster'").fetchone()[0]
    assert n == 1
    assert app_module._is_paid_member("buyer@x.com") is True


def test_program_webhook_and_redirect_single_care_grant(monkeypatch, tmp_path):
    """Webhook + redirect (any order) grants the care window exactly once."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    _mock_paid_biofield_session(app_module, monkeypatch, email="buyer@x.com")
    _post_biofield_webhook(app_module, monkeypatch)
    app_module.app.test_client().get("/begin/checkout-return?session_id=cs_1")
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM memberships WHERE email='buyer@x.com' "
                       "AND source='care_taster'").fetchone()[0]
    assert n == 1


def _grant(cx, email, source, days=90):
    from datetime import timedelta
    import uuid
    now = datetime.utcnow()
    cx.execute(
        "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source, truly_vip_ref, notes) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), email, now.isoformat() + "Z", (now + timedelta(days=days)).isoformat() + "Z",
         source, source, "", ""))
    cx.commit()


def test_care_taster_outranks_lingering_trial_grant(monkeypatch, tmp_path):
    """Regression (sticky-trial, same shape as the prepay-term test): a biofield_trial
    grant alone reads as 'trial' (discount withheld). Once the SAME email also holds a
    care_taster grant (a real paid program purchase), it must no longer read as trial and
    must be treated as a paid member."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        _grant(cx, "d@x.com", "biofield_trial")
        _grant(cx, "d@x.com", "care_taster", days=30)
    assert app_module.membership_category("d@x.com") != "trial"
    assert app_module._is_paid_member("d@x.com") is True


# ── Task 3: credit the $1 deposit to points, auto-redeem at program checkout ──

def _mock_paid_biofield_trial_session(app_module, monkeypatch, email="buyer@x.com",
                                       pi="pi_dep1"):
    from dashboard import stripe_pay as _sp
    monkeypatch.setattr(_sp, "get_session", lambda sid: {
        "metadata": {"kind": "biofield_trial", "email": email},
        "payment_intent": pi, "amount_total": 100,
    })
    monkeypatch.setattr(app_module.stripe_pay, "get_session", _sp.get_session, raising=False)
    monkeypatch.setattr(_sp, "get_payment_intent",
        lambda p: {"customer": "cus_1", "payment_method": "pm_1", "status": "succeeded"})
    monkeypatch.setattr(app_module.stripe_pay, "get_payment_intent", _sp.get_payment_intent,
                        raising=False)


def test_deposit_credit_granted_on_trial_fulfillment(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    _mock_paid_biofield_trial_session(app_module, monkeypatch, email="buyer@x.com",
                                       pi="pi_dep1")
    c = app_module.app.test_client()
    r = c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    assert r.status_code == 302

    from dashboard import points as _points
    with sqlite3.connect(db) as cx:
        _points.init_points_table(cx)
        bal = _points.balance(cx, "buyer@x.com")
    assert bal == 100, "the $1 deposit must credit 100 cents of redemption value"

    # Replay (closed tab retried, or webhook + redirect racing) must not double-credit.
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")
    with sqlite3.connect(db) as cx:
        bal2 = _points.balance(cx, "buyer@x.com")
    assert bal2 == 100, "replay must not double-credit the deposit (idempotent on the PI id)"


def test_deposit_credit_skipped_when_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", False, raising=False)
    monkeypatch.setattr(app_module, "BIOFIELD_TRIAL_ENABLED", True, raising=False)
    _mock_paid_biofield_trial_session(app_module, monkeypatch, email="buyer@x.com",
                                       pi="pi_dep2")
    c = app_module.app.test_client()
    c.get("/begin/checkout-return?kind=biofield_trial&session_id=cs_1")

    from dashboard import points as _points
    with sqlite3.connect(db) as cx:
        _points.init_points_table(cx)
        bal = _points.balance(cx, "buyer@x.com")
    assert bal == 0


def test_program_checkout_auto_redeems_deposit_credit(monkeypatch, tmp_path):
    """A pre-seeded 100c deposit credit auto-applies (no explicit redeem passed) as $1
    off a scalable-tier ($100) program checkout: charged 9900, metadata reflects it."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(app_module, "_QBO_PAYMENTS_ACTIVE", True, raising=False)
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")

    from dashboard import points as _points
    with sqlite3.connect(db) as cx:
        _points.init_points_table(cx)
        _points.credit(cx, "buyer@x.com", value_cents=100, reason="deposit_credit",
                       order_ref="pi_seed")

    cap = {}
    import dashboard.qbo_billing as _qb
    monkeypatch.setattr(_qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    def fake_invoice(cust, lines, **kw):
        cap["lines"] = lines
        cap["invoice_kw"] = kw
        return {"Id": "INVB", "SyncToken": "0", "DocNumber": "9", "TotalAmt": 99.0}
    monkeypatch.setattr(_qb, "create_invoice", fake_invoice)
    monkeypatch.setattr(_qb, "get_invoice_pay_link", lambda inv: "")

    import dashboard.stripe_pay as _sp

    def fake_session(amount_cents, *, customer_email, description, metadata,
                     success_url, cancel_url, save_card=False):
        cap["stripe_amount"] = amount_cents
        cap["stripe_metadata"] = metadata
        return {"id": "cs_test", "url": "https://stripe/biofield"}
    monkeypatch.setattr(_sp, "create_checkout_session", fake_session)
    monkeypatch.setattr(app_module.stripe_pay, "create_checkout_session", fake_session,
                        raising=False)

    c = app_module.app.test_client()
    r = c.post("/biofield/checkout",
               json={"email": "buyer@x.com", "name": "B", "tier": "scalable"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert cap["stripe_amount"] == 9900, "the $1 deposit credit must auto-apply as $1 off"
    assert int(cap["stripe_metadata"]["points_redeemed_cents"]) == 100


# ── Task 5: flag-gated $100 program CTA on the biofield reveal page ──

def test_reveal_page_exposes_program_cta(monkeypatch, tmp_path):
    """Flag ON: the served begin-biofield.html/JS must carry the program CTA wiring
    (reads window.__REVEAL__.program_enabled, posts tier 'scalable' to /biofield/checkout).
    Invalid token -> payload null is fine (scaffold test, mirrors
    test_reveal_page_scaffold_unlock_checkout in test_biofield_trial.py) -- the CTA
    markup/JS lives in the static HTML regardless of which reveal state renders."""
    app_module = _load_app()
    _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "PROGRAM_CARE_TASTER_ENABLED", True, raising=False)
    client = app_module.app.test_client()
    r = client.get("/begin/biofield/any-token")
    assert r.status_code == 200
    html = r.data.decode()
    assert "program_enabled" in html, "'program_enabled' not found in begin-biofield.html"
    assert "/biofield/checkout" in html, "'/biofield/checkout' not found in begin-biofield.html"
    assert "scalable" in html, "'scalable' not found in begin-biofield.html"
