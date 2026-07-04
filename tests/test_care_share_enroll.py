# tests/test_care_share_enroll.py
"""Dispensary "Start Continuous Care" attributed enrollment (Task 6).

A patient starts Continuous Care ($99/mo) through their doctor's dispensary
channel: POST /dispensary/<code>/continuous-care resolves the doctor by code,
runs the same consent gate as the dispensary product checkout, and starts the
SAME card-on-file Stripe checkout as /continuous-care/checkout — threading the
doctor's practitioner id through the session metadata. On fulfilment the
membership is stamped attributed_practitioner_id=<pid> (kind=membership) and the
enrollment (month-1) charge credits the doctor via care_share.credit_for_charge.

Mirrors tests/test_continuous_care_monthly.py: tmp-file sqlite via monkeypatched
LOG_DB, Stripe faked by monkeypatching dashboard.stripe_pay (app.py's module
alias is the same object). practitioner lookup + consent gate are stubbed the
way the dispensary route tests stub them; credit_for_charge is patched to a
recorder so no wallet/Supabase call is made.
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
    monkeypatch.setattr(app_module, "_ingest_order", lambda **kw: None, raising=False)
    monkeypatch.setattr(app_module, "_send_subscription_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app_module, "CONTINUOUS_CARE_MONTHLY_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_STRIPE_ACTIVE", True, raising=False)
    return db


def _stub_dispensary(app_module, monkeypatch, code="doc1", pid="prac-42"):
    monkeypatch.setattr(app_module._pp, "practitioner_id_by_dispensary_code",
                        lambda c: pid if c == code else None)


# ---------------------------------------------------------------------------
# POST /dispensary/<code>/continuous-care
# ---------------------------------------------------------------------------

def test_unknown_code_404(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    _stub_dispensary(app_module, monkeypatch)
    monkeypatch.setattr(app_module, "is_member", lambda sid, email: True, raising=False)
    r = app_module.app.test_client().post(
        "/dispensary/nope/continuous-care", json={"email": "p@x.com"})
    assert r.status_code == 404


def test_consent_gate_blocks(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    _stub_dispensary(app_module, monkeypatch)
    monkeypatch.setattr(app_module, "is_member", lambda sid, email: False, raising=False)
    r = app_module.app.test_client().post(
        "/dispensary/doc1/continuous-care", json={"email": "p@x.com"})
    assert r.status_code == 403
    assert r.get_json().get("need_optin") is True


def test_enroll_attributes_membership_and_credits_doctor(monkeypatch, tmp_path):
    """The full attributed-enrollment flow: the dispensary endpoint starts the
    $99/mo card-on-file checkout stamped with dispensary_pid, and fulfilment
    creates an attributed membership AND credits the enrolling doctor for the
    month-1 charge (charge_cents=9900)."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    _stub_dispensary(app_module, monkeypatch)
    monkeypatch.setattr(app_module, "is_member", lambda sid, email: True, raising=False)

    # 1) The endpoint starts the SAME Stripe card-on-file checkout, capturing metadata.
    cap = {}

    def fake_checkout(amount, **kw):
        cap["amount"] = amount
        cap["kw"] = kw
        return {"id": "cs_disp", "url": "https://stripe/x"}

    monkeypatch.setattr(app_module.stripe_pay, "create_checkout_session", fake_checkout)
    r = app_module.app.test_client().post(
        "/dispensary/doc1/continuous-care",
        json={"email": "pat@x.com", "term_months": 12})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == "https://stripe/x"

    md = cap["kw"]["metadata"]
    assert md["kind"] == "continuous_care_monthly"
    assert md["dispensary_pid"] == "prac-42"
    assert md["email"] == "pat@x.com"
    assert cap["amount"] == 9900              # MONTHLY_ANCHOR_CENTS ($99)
    assert cap["kw"]["save_card"] is True     # card-on-file (months 2..N)

    # 2) Fulfilment (as the Stripe return / webhook would drive it): feed the
    #    endpoint's own metadata back through get_session.
    from dashboard import stripe_pay, care_share, subscriptions as subs
    monkeypatch.setattr(stripe_pay, "get_session",
                        lambda sid: {"metadata": md, "payment_intent": "pi_1"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
                        lambda pi: {"status": "succeeded", "id": "pi_1",
                                    "customer": "cus_1", "payment_method": "pm_1"})

    rec = []

    def fake_credit(sub, *, charge_cents, **kw):
        rec.append((sub, charge_cents))
        return charge_cents

    monkeypatch.setattr(care_share, "credit_for_charge", fake_credit)

    res = app_module._fulfill_continuous_care_monthly("cs_disp")
    assert res.get("ok") is True

    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "pat@x.com")
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "membership"
    assert row["attributed_practitioner_id"] == "prac-42"
    assert row["order_count"] == 1            # month 1 charged at checkout

    # The enrollment charge credited the doctor exactly once, at $99.
    assert len(rec) == 1
    credited_sub, credited_cents = rec[0]
    assert credited_cents == 9900
    assert credited_sub["id"] == row["id"]
    assert credited_sub["attributed_practitioner_id"] == "prac-42"


def test_direct_enrollment_still_unattributed(monkeypatch, tmp_path):
    """Additive-only guard: the existing (non-dispensary) continuous-care
    fulfilment path stays unattributed and fires no care-share credit."""
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import stripe_pay, care_share, subscriptions as subs
    monkeypatch.setattr(stripe_pay, "get_session",
                        lambda sid: {"metadata": {"email": "solo@x.com",
                                                  "kind": "continuous_care_monthly",
                                                  "term_months": "6"},
                                     "payment_intent": "pi_2"})
    monkeypatch.setattr(stripe_pay, "get_payment_intent",
                        lambda pi: {"status": "succeeded", "id": "pi_2",
                                    "customer": "cus_2", "payment_method": "pm_2"})
    rec = []
    monkeypatch.setattr(care_share, "credit_for_charge",
                        lambda sub, **kw: rec.append(sub))
    app_module._fulfill_continuous_care_monthly("cs_solo")
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        rows = subs.active_memberships_by_email(cx, "solo@x.com")
    assert len(rows) == 1
    assert rows[0]["attributed_practitioner_id"] in (None, "")
    assert rec == []
