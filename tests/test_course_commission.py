"""Ambassador commission tests for the four course-fulfillment credit sites:
_fulfill_course_purchase (payment mode only), _fulfill_module_certification,
_course_membership_renew (drip invoices), and _course_plan_charge (plan
invoices). Each site wraps _credit_referrer_by_slug via the thin
_credit_course_referral gate. Reuses the webhook fixtures from
tests/test_courses_webhook.py and tests/test_module_cert_webhook.py, and the
affiliate-seed helper pattern from tests/test_credit_referrer_by_slug.py.
"""
import hashlib
import hmac
import json
import sqlite3
import time

import pytest

from dashboard import points, rewards


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    # Do NOT importlib.reload(app): see tests/test_courses_webhook.py for why.
    import app as m
    db_path = tmp_path / "chat_log.db"
    monkeypatch.setattr(m, "LOG_DB", db_path)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(m, "send_mentorship_setup_link", lambda *a, **k: ("test", None))
    m.app.config["TESTING"] = True
    # Isolate from live Supabase (cert-tiered referral pct lookup).
    monkeypatch.setattr(m._pp, "modules_completed_for_email", lambda e: None)
    cx = sqlite3.connect(str(db_path))
    cx.execute("CREATE TABLE affiliate_signups (slug TEXT UNIQUE, email TEXT, status TEXT)")
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, tags TEXT DEFAULT '[]')")
    cx.execute("CREATE TABLE referral_events (received_at TEXT, email TEXT, utm_source TEXT)")
    cx.commit()
    cx.close()
    return m


def _seed_affiliate(appmod, slug="doc", ref_email="doc@x.com", status="approved"):
    cx = sqlite3.connect(str(appmod.LOG_DB))
    cx.execute("INSERT INTO affiliate_signups VALUES (?,?,?)", (slug, ref_email, status))
    cx.execute("INSERT INTO people (email, tags) VALUES (?,?)", (ref_email, json.dumps(["type:practitioner"])))
    cx.commit()
    cx.close()


def _expected_reward(appmod, ref_email, amount_cents):
    settings = rewards.load_settings(appmod._rewards_settings())
    pct = appmod._referral_pct_for_referrer(ref_email, settings)
    return round(amount_cents * pct)


def _balance(appmod, email):
    cx = sqlite3.connect(str(appmod.LOG_DB))
    points.init_points_table(cx)
    b = points.balance(cx, email)
    cx.close()
    return b


def _sign(body: bytes, secret: str, ts: int) -> str:
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def _post_event(appmod, event: dict):
    body = json.dumps(event).encode()
    ts = int(time.time())
    return appmod.app.test_client().post(
        "/webhook/stripe", data=body,
        headers={"Stripe-Signature": _sign(body, "whsec_test", ts)},
        content_type="application/json")


# ---------------------------------------------------------------------------
# Site 1: _fulfill_course_purchase — one-time bundle (mode == "payment") only.
# ---------------------------------------------------------------------------

def _onetime_session(**overrides):
    base = {
        "id": "cs_ot_1",
        "mode": "payment",
        "payment_status": "paid",
        "amount_total": 60000,
        "customer": "cus_1",
        "metadata": {"kind": "course_purchase", "email": "buyer@x.com",
                     "product": "onetime", "ref": "doc"},
        "customer_details": {"email": "buyer@x.com"},
    }
    base.update(overrides)
    return base


def test_onetime_purchase_credits_referring_ambassador(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    session = _onetime_session()
    expected = _expected_reward(appmod, "doc@x.com", 60000)
    result = appmod._fulfill_course_purchase(session)
    assert result == "ok"
    assert _balance(appmod, "doc@x.com") == expected
    assert expected > 0


def test_onetime_purchase_replay_does_not_double_credit(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    session = _onetime_session()
    appmod._fulfill_course_purchase(session)
    first = _balance(appmod, "doc@x.com")
    appmod._fulfill_course_purchase(session)
    assert _balance(appmod, "doc@x.com") == first


def test_onetime_purchase_rewards_disabled_credits_nothing(appmod, monkeypatch):
    monkeypatch.delenv("REWARDS_TIERS_ENABLED", raising=False)
    _seed_affiliate(appmod)
    session = _onetime_session()
    appmod._fulfill_course_purchase(session)
    assert _balance(appmod, "doc@x.com") == 0


def test_onetime_purchase_no_ref_credits_nothing(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    session = _onetime_session(metadata={"kind": "course_purchase", "email": "buyer@x.com",
                                          "product": "onetime"})
    appmod._fulfill_course_purchase(session)
    assert _balance(appmod, "doc@x.com") == 0


def test_subscription_checkout_completed_does_not_credit_only_invoice_does(appmod, monkeypatch):
    # A membership/plan checkout.session.completed must NOT credit inside
    # _fulfill_course_purchase — only the resulting invoice.paid does, so the
    # first payment is never double-credited.
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    sub_id = "sub_drip_ref"
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": sub_id, "status": "active", "current_period_end": 9_999_999_999,
        "customer": "cus_d", "metadata": {"kind": "course_membership", "email": "m@x.com", "ref": "doc"}})
    session = {
        "id": "cs_drip_ref", "mode": "subscription", "subscription": sub_id, "customer": "cus_d",
        "metadata": {"kind": "course_purchase", "email": "m@x.com", "product": "membership"},
        "customer_details": {"email": "m@x.com"},
    }
    result = appmod._fulfill_course_purchase(session)
    assert result == "ok"
    # checkout.session.completed itself must not have credited anything
    assert _balance(appmod, "doc@x.com") == 0
    # the subscription's first invoice.paid credits exactly once
    invoice = {"subscription": sub_id, "id": "in_first", "amount_paid": 9900,
               "billing_reason": "subscription_create"}
    expected = _expected_reward(appmod, "doc@x.com", 9900)
    appmod._course_membership_renew(invoice)
    assert _balance(appmod, "doc@x.com") == expected
    assert expected > 0


# ---------------------------------------------------------------------------
# Site 2: _fulfill_module_certification
# ---------------------------------------------------------------------------

def _modcert_session(**overrides):
    base = {
        "id": "cs_modcert_ref_1",
        "amount_total": 20000,
        "metadata": {"kind": "module_certification", "email": "buyer@x.com",
                     "course": "ash-certification", "module": "02-body", "ref": "doc"},
        "customer_details": {"email": "buyer@x.com"},
    }
    base.update(overrides)
    return base


def test_module_cert_credits_referring_ambassador(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    session = _modcert_session()
    expected = _expected_reward(appmod, "doc@x.com", 20000)
    result = appmod._fulfill_module_certification(session)
    assert result == "ok"
    assert _balance(appmod, "doc@x.com") == expected
    assert expected > 0


def test_module_cert_replay_does_not_double_credit(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    session = _modcert_session()
    appmod._fulfill_module_certification(session)
    first = _balance(appmod, "doc@x.com")
    appmod._fulfill_module_certification(session)
    assert _balance(appmod, "doc@x.com") == first


def test_module_cert_rewards_disabled_credits_nothing(appmod, monkeypatch):
    monkeypatch.delenv("REWARDS_TIERS_ENABLED", raising=False)
    _seed_affiliate(appmod)
    session = _modcert_session()
    appmod._fulfill_module_certification(session)
    assert _balance(appmod, "doc@x.com") == 0


def test_module_cert_no_ref_credits_nothing(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    session = _modcert_session(metadata={"kind": "module_certification", "email": "buyer@x.com",
                                          "course": "ash-certification", "module": "02-body"})
    appmod._fulfill_module_certification(session)
    assert _balance(appmod, "doc@x.com") == 0


# ---------------------------------------------------------------------------
# Site 3: _course_membership_renew — drip ($99/mo), per-payment.
# ---------------------------------------------------------------------------

def _drip_sub(email="drip@x.com", ref="doc", cpe=9_999_999_999):
    md = {"kind": "course_membership", "email": email}
    if ref is not None:
        md["ref"] = ref
    return {"id": "sub_drip_c", "status": "active", "current_period_end": cpe,
            "customer": "cus_d", "metadata": md}


def test_drip_invoice_credits_referring_ambassador(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _drip_sub())
    invoice = {"subscription": "sub_drip_c", "id": "in_drip_1", "amount_paid": 9900}
    expected = _expected_reward(appmod, "doc@x.com", 9900)
    result = appmod._course_membership_renew(invoice)
    assert result == "ok"
    assert _balance(appmod, "doc@x.com") == expected
    assert expected > 0


def test_drip_invoice_replay_does_not_double_credit(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _drip_sub())
    invoice = {"subscription": "sub_drip_c", "id": "in_drip_2", "amount_paid": 9900}
    appmod._course_membership_renew(invoice)
    first = _balance(appmod, "doc@x.com")
    appmod._course_membership_renew(invoice)
    assert _balance(appmod, "doc@x.com") == first


def test_drip_invoice_rewards_disabled_credits_nothing(appmod, monkeypatch):
    monkeypatch.delenv("REWARDS_TIERS_ENABLED", raising=False)
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _drip_sub())
    invoice = {"subscription": "sub_drip_c", "id": "in_drip_3", "amount_paid": 9900}
    appmod._course_membership_renew(invoice)
    assert _balance(appmod, "doc@x.com") == 0


def test_drip_invoice_no_ref_credits_nothing(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _drip_sub(ref=None))
    invoice = {"subscription": "sub_drip_c", "id": "in_drip_4", "amount_paid": 9900}
    appmod._course_membership_renew(invoice)
    assert _balance(appmod, "doc@x.com") == 0


def test_drip_invoice_credits_on_renewal_too(appmod, monkeypatch):
    # Per-payment crediting: a second, distinct invoice credits again.
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _drip_sub())
    appmod._course_membership_renew({"subscription": "sub_drip_c", "id": "in_drip_5", "amount_paid": 9900})
    first = _balance(appmod, "doc@x.com")
    appmod._course_membership_renew({"subscription": "sub_drip_c", "id": "in_drip_6", "amount_paid": 9900})
    assert _balance(appmod, "doc@x.com") == first * 2
    assert first > 0


# ---------------------------------------------------------------------------
# Site 4: _course_plan_charge — $297x12 plan, per-payment.
# ---------------------------------------------------------------------------

def _plan_sub(email="plan@x.com", ref="doc", cpe=9_999_999_999):
    md = {"kind": "course_plan", "email": email}
    if ref is not None:
        md["ref"] = ref
    return {"id": "sub_plan_c", "status": "active", "current_period_end": cpe,
            "customer": "cus_p", "metadata": md}


def test_plan_invoice_credits_referring_ambassador(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _plan_sub())
    invoice = {"subscription": "sub_plan_c", "id": "in_plan_1", "amount_paid": 29700}
    expected = _expected_reward(appmod, "doc@x.com", 29700)
    result = appmod._course_plan_charge(invoice)
    assert result == "ok"
    assert _balance(appmod, "doc@x.com") == expected
    assert expected > 0


def test_plan_invoice_replay_does_not_double_credit(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _plan_sub())
    invoice = {"subscription": "sub_plan_c", "id": "in_plan_2", "amount_paid": 29700}
    appmod._course_plan_charge(invoice)
    first = _balance(appmod, "doc@x.com")
    appmod._course_plan_charge(invoice)
    assert _balance(appmod, "doc@x.com") == first


def test_plan_invoice_rewards_disabled_credits_nothing(appmod, monkeypatch):
    monkeypatch.delenv("REWARDS_TIERS_ENABLED", raising=False)
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _plan_sub())
    invoice = {"subscription": "sub_plan_c", "id": "in_plan_3", "amount_paid": 29700}
    appmod._course_plan_charge(invoice)
    assert _balance(appmod, "doc@x.com") == 0


def test_plan_invoice_no_ref_credits_nothing(appmod, monkeypatch):
    monkeypatch.setenv("REWARDS_TIERS_ENABLED", "true")
    _seed_affiliate(appmod)
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _plan_sub(ref=None))
    invoice = {"subscription": "sub_plan_c", "id": "in_plan_4", "amount_paid": 29700}
    appmod._course_plan_charge(invoice)
    assert _balance(appmod, "doc@x.com") == 0
