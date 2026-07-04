# tests/test_continuous_care_monthly.py
"""Route + fulfiller tests for Continuous Care MONTHLY (card-on-file, 6- or
12-month capped term): POST /continuous-care/checkout, GET /continuous-care/return,
_fulfill_continuous_care_monthly.

Mirrors test_prepay_checkout.py / test_biofield_trial.py: tmp-file sqlite via
monkeypatched LOG_DB (no shared-LOG_DB pollution), Stripe faked by monkeypatching
dashboard.stripe_pay (app.py's module-level `stripe_pay` alias is the same module
object, so appmod.stripe_pay.* and dashboard.stripe_pay.* patches are equivalent).
The real Stripe-helper accessor names — confirmed by reading _fulfill_prepay_term
(app.py) and dashboard/stripe_pay.py — are get_session(session_id) and
get_payment_intent(pi_id), NOT get_checkout_session/get_payment_intent(pi=...).
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
        subscriptions.migrate_add_term_cap_column(cx)
        subscriptions.migrate_add_attribution_column(cx)
        subscriptions.migrate_add_consent_column(cx)
        app_module.init_membership_tables(cx)
        cx.commit()
    # Keep the fulfiller's best-effort side-effects out of the test by default;
    # individual tests override where they need to assert on them.
    monkeypatch.setattr(app_module, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app_module, "_send_subscription_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: None, raising=False)
    return db


# ---------------------------------------------------------------------------
# POST /continuous-care/checkout
# ---------------------------------------------------------------------------

def test_checkout_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", False, raising=False)
    r = app_module.app.test_client().post(
        "/continuous-care/checkout", json={"email": "a@x.com", "term_months": 6})
    assert r.status_code == 200
    assert r.get_json()["ok"] is False


def test_checkout_builds_vaulted_session(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    cap = {}

    def fake_sess(amount, **kw):
        cap["amount"] = amount
        cap["kw"] = kw
        return {"id": "cs_1", "url": "https://stripe/x"}

    monkeypatch.setattr(app_module.stripe_pay, "create_checkout_session", fake_sess)
    r = app_module.app.test_client().post(
        "/continuous-care/checkout", json={"email": "a@x.com", "term_months": 6})
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == "https://stripe/x"
    from dashboard import prepay
    assert cap["amount"] == prepay.MONTHLY_ANCHOR_CENTS
    assert cap["kw"]["save_card"] is True
    assert cap["kw"]["metadata"]["kind"] == "continuous_care_monthly"
    assert cap["kw"]["metadata"]["term_months"] == "6"
    assert cap["kw"]["metadata"]["email"] == "a@x.com"


def test_checkout_rejects_bad_term(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    r = app_module.app.test_client().post(
        "/continuous-care/checkout", json={"email": "a@x.com", "term_months": 3})
    assert r.get_json()["ok"] is False


# ---------------------------------------------------------------------------
# _fulfill_continuous_care_monthly (+ GET /continuous-care/return, webhook)
# ---------------------------------------------------------------------------

def _mock_stripe_success(app_module, monkeypatch, email, term="6", session_id="cs_1"):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda sid: {"metadata": {"email": email, "kind": "continuous_care_monthly",
                                  "term_months": term},
                    "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi_id: {"status": "succeeded", "id": "pi_1",
                       "customer": "cus_1", "payment_method": "pm_1"})


def test_fulfill_creates_capped_membership_and_grants_access(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_stripe_success(app_module, monkeypatch, "m@x.com")
    app_module._fulfill_continuous_care_monthly("sess_ccm_1")
    from dashboard import subscriptions as subs
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "m@x.com")
    assert len(rows) == 1
    assert rows[0]["term_charges_total"] == 6
    assert rows[0]["order_count"] == 1
    assert rows[0]["stripe_payment_method_id"] == "pm_1"
    assert rows[0]["stripe_customer_id"] == "cus_1"
    assert app_module._is_paid_member("m@x.com") is True
    assert app_module.membership_category("m@x.com") == "full"


def test_fulfill_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_stripe_success(app_module, monkeypatch, "idem@x.com")
    app_module._fulfill_continuous_care_monthly("sess_ccm_2")
    app_module._fulfill_continuous_care_monthly("sess_ccm_2")  # replay
    from dashboard import subscriptions as subs
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "idem@x.com")
    assert len(rows) == 1  # no double sub


def test_fulfill_no_second_membership_for_existing_member(monkeypatch, tmp_path):
    """Two DIFFERENT checkout sessions for the same email (per-session idempotency
    can't catch this) must NOT create a second active membership — else the charge
    cron would bill $99/mo twice. Guard mirrors the sibling minting paths."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_stripe_success(app_module, monkeypatch, "dup@x.com")
    app_module._fulfill_continuous_care_monthly("sess_dup_A")
    # A distinct session id for the same email -> claim table doesn't dedup it.
    res = app_module._fulfill_continuous_care_monthly("sess_dup_B")
    assert res.get("duplicate_member") is True
    from dashboard import subscriptions as subs
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "dup@x.com")
    assert len(rows) == 1  # still exactly one billing sub


def test_fulfill_no_membership_when_pi_not_succeeded(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda sid: {"metadata": {"email": "fail@x.com", "kind": "continuous_care_monthly",
                                  "term_months": "6"},
                    "payment_intent": "pi_x"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi_id: {"status": "requires_payment_method", "id": "pi_x"})
    app_module._fulfill_continuous_care_monthly("sess_ccm_3")
    from dashboard import subscriptions as subs
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "fail@x.com")
    assert rows == []


def test_fulfill_no_membership_when_card_not_vaulted(monkeypatch, tmp_path):
    """Succeeded PI but no customer/payment_method (shouldn't happen with
    save_card=True, but guard anyway): no card on file to bill => no membership."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_session",
        lambda sid: {"metadata": {"email": "nocard@x.com", "kind": "continuous_care_monthly",
                                  "term_months": "6"},
                    "payment_intent": "pi_y"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
        lambda pi_id: {"status": "succeeded", "id": "pi_y", "customer": "", "payment_method": ""})
    app_module._fulfill_continuous_care_monthly("sess_ccm_4")
    from dashboard import subscriptions as subs
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "nocard@x.com")
    assert rows == []


def test_return_route_creates_membership_and_redirects(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_stripe_success(app_module, monkeypatch, "ret@x.com", term="12")
    r = app_module.app.test_client().get("/continuous-care/return?session_id=sess_ret_1")
    assert r.status_code in (301, 302)
    assert "care=ok" in r.headers.get("Location", "")
    from dashboard import subscriptions as subs
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "ret@x.com")
    assert len(rows) == 1
    assert rows[0]["term_charges_total"] == 12


def _post_webhook(app_module, monkeypatch, sid="cs_1"):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    import json as _json
    payload = _json.dumps({"type": "checkout.session.completed", "data": {"object": {"id": sid}}})
    return app_module.app.test_client().post("/webhook/stripe", data=payload,
                                             content_type="application/json")


def test_webhook_fulfills_continuous_care_closed_tab(monkeypatch, tmp_path):
    """Safety net: the Stripe webhook creates the membership even when the
    browser never lands on /continuous-care/return (closed tab / dropped redirect)."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _mock_stripe_success(app_module, monkeypatch, "wh@x.com", term="6")
    r = _post_webhook(app_module, monkeypatch, sid="cs_1")
    assert r.status_code == 200
    from dashboard import subscriptions as subs
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "wh@x.com")
    assert len(rows) == 1


def test_webhook_and_return_single_membership(monkeypatch, tmp_path):
    """Redirect + webhook both firing (any order) must create exactly one sub."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    _mock_stripe_success(app_module, monkeypatch, "race@x.com", term="6")
    _post_webhook(app_module, monkeypatch, sid="cs_1")
    app_module.app.test_client().get("/continuous-care/return?session_id=cs_1")
    from dashboard import subscriptions as subs
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "race@x.com")
    assert len(rows) == 1
