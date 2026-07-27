import hashlib
import hmac
import json
import sqlite3
import time

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    # Do NOT importlib.reload(app): a reload with DATA_DIR set re-runs app.py's
    # module bootstrap, which starts a BackgroundScheduler + prewarm daemon that
    # are never shut down and leak into the rest of the suite (timing-
    # nondeterministic bystander failures on CI). Redirect the module globals we
    # need directly instead — the proven pattern used by tests/test_support_program_*.
    import app as m
    monkeypatch.setattr(m, "LOG_DB", tmp_path / "chat_log.db")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")  # read live in the route
    monkeypatch.setattr(m, "send_mentorship_setup_link", lambda *a, **k: ("test", None))
    m.app.config["TESTING"] = True
    return m


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


def _paid_level(appmod, email, now=None):
    from dashboard import course_entitlements as ce
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ce.init_course_entitlements_table(cx)
        return ce.paid_level_for(cx, email, now=now)


def test_onetime_purchase_grants_cert_and_mints_token(appmod, monkeypatch):
    sent = {}
    monkeypatch.setattr(appmod, "send_mentorship_setup_link",
                        lambda email, name, url: sent.update(email=email, url=url) or ("test", None))
    event = {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_100", "mode": "payment", "payment_status": "paid",
        "customer": "cus_1", "metadata": {"kind": "course_purchase", "email": "buy@x.com", "product": "onetime"},
        "customer_details": {"email": "buy@x.com"}}}}
    r = _post_event(appmod, event)
    assert r.status_code == 200
    assert _paid_level(appmod, "buy@x.com", now=9_999_999_999) == 2
    assert sent["email"] == "buy@x.com" and "token=" in sent["url"]


def test_subscription_purchase_grants_membership(appmod, monkeypatch):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": "sub_1", "status": "active", "current_period_end": 2000,
        "customer": "cus_2", "metadata": {"kind": "course_membership", "email": "m@x.com"}})
    event = {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_200", "mode": "subscription", "subscription": "sub_1", "customer": "cus_2",
        "metadata": {"kind": "course_purchase", "email": "m@x.com", "product": "membership"},
        "customer_details": {"email": "m@x.com"}}}}
    r = _post_event(appmod, event)
    assert r.status_code == 200
    assert _paid_level(appmod, "m@x.com", now=500) == 2
    assert _paid_level(appmod, "m@x.com", now=2500) == 0


def test_replayed_event_does_not_double_grant(appmod):
    event = {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_300", "mode": "payment", "customer": "c",
        "metadata": {"kind": "course_purchase", "email": "r@x.com", "product": "onetime"},
        "customer_details": {"email": "r@x.com"}}}}
    _post_event(appmod, event)
    _post_event(appmod, event)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        n = cx.execute("SELECT COUNT(*) FROM course_entitlements WHERE stripe_ref='cs_300'").fetchone()[0]
    assert n == 1


def test_bad_signature_does_not_grant(appmod):
    body = json.dumps({"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_400", "mode": "payment",
        "metadata": {"kind": "course_purchase", "email": "no@x.com", "product": "onetime"},
        "customer_details": {"email": "no@x.com"}}}}).encode()
    r = appmod.app.test_client().post("/webhook/stripe", data=body,
                                      headers={"Stripe-Signature": "t=1,v1=deadbeef"},
                                      content_type="application/json")
    assert r.status_code == 400
    assert _paid_level(appmod, "no@x.com", now=9_999_999_999) == 0


def test_subscription_deleted_relocks(appmod, monkeypatch):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": "sub_9", "status": "active", "current_period_end": 9_999_999_999,
        "customer": "c", "metadata": {"kind": "course_membership", "email": "z@x.com"}})
    _post_event(appmod, {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_500", "mode": "subscription", "subscription": "sub_9", "customer": "c",
        "metadata": {"kind": "course_purchase", "email": "z@x.com", "product": "membership"},
        "customer_details": {"email": "z@x.com"}}}})
    assert _paid_level(appmod, "z@x.com", now=1) == 2
    _post_event(appmod, {"type": "customer.subscription.deleted", "data": {"object": {
        "id": "sub_9", "metadata": {"kind": "course_membership", "email": "z@x.com"}}}})
    assert _paid_level(appmod, "z@x.com", now=1) == 0


def test_renewal_without_metadata_email_extends_existing(appmod, monkeypatch):
    from dashboard import stripe_pay
    # Anonymous membership buyer: subscription metadata has kind but NO email.
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": "sub_ne", "status": "active", "current_period_end": 1000,
        "customer": "c", "metadata": {"kind": "course_membership"}})
    _post_event(appmod, {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_ne", "mode": "subscription", "subscription": "sub_ne", "customer": "c",
        "metadata": {"kind": "course_purchase", "product": "membership"},
        "customer_details": {"email": "anon@x.com"}}}})
    assert _paid_level(appmod, "anon@x.com", now=500) == 2      # initial grant via customer_details
    assert _paid_level(appmod, "anon@x.com", now=1500) == 0     # expires at 1000
    # invoice.paid renewal extends to 3000 even though sub metadata has no email
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": "sub_ne", "status": "active", "current_period_end": 3000,
        "customer": "c", "metadata": {"kind": "course_membership"}})
    _post_event(appmod, {"type": "invoice.paid", "data": {"object": {"subscription": "sub_ne"}}})
    assert _paid_level(appmod, "anon@x.com", now=2500) == 2     # renewed via stored-row email fallback


def _plan_sub(email="p@x.com", cpe=1000):
    return {"id": "sub_plan", "status": "active", "current_period_end": cpe,
            "customer": "cus_p", "metadata": {"kind": "course_plan", "email": email}}


def test_plan_charges_extend_then_convert_to_lifetime(appmod, monkeypatch):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _plan_sub(cpe=1000))
    cancelled = {}
    monkeypatch.setattr(stripe_pay, "cancel_subscription",
                        lambda sid: cancelled.setdefault("sid", sid) or {"id": sid, "status": "canceled"})
    # charges 1..11 → membership window (level 2 only within the window)
    for i in range(1, 12):
        _post_event(appmod, {"type": "invoice.paid",
                             "data": {"object": {"subscription": "sub_plan", "id": f"in_{i}"}}})
    assert _paid_level(appmod, "p@x.com", now=500) == 2            # inside window
    assert _paid_level(appmod, "p@x.com", now=9_999_999_999) == 0  # window expired, no cert yet
    # 12th charge → lifetime cert + subscription cancelled
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_plan", "id": "in_12"}}})
    assert _paid_level(appmod, "p@x.com", now=9_999_999_999) == 2  # lifetime cert
    assert cancelled.get("sid") == "sub_plan"


def test_plan_charge_replay_does_not_advance_count(appmod, monkeypatch):
    import sqlite3
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _plan_sub())
    monkeypatch.setattr(stripe_pay, "cancel_subscription", lambda sid: {"id": sid, "status": "canceled"})
    for i in (1, 1, 2):  # in_1 replayed
        _post_event(appmod, {"type": "invoice.paid",
                             "data": {"object": {"subscription": "sub_plan", "id": f"in_{i}"}}})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        n = cx.execute("SELECT COUNT(*) FROM course_plan_charges WHERE sub_id='sub_plan'").fetchone()[0]
    assert n == 2  # not 3


def test_plan_cancel_before_12_locks_but_after_conversion_keeps_access(appmod, monkeypatch):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _plan_sub(email="q@x.com", cpe=1000))
    monkeypatch.setattr(stripe_pay, "cancel_subscription", lambda sid: {"id": sid, "status": "canceled"})
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_plan", "id": "in_a"}}})
    assert _paid_level(appmod, "q@x.com", now=500) == 2
    _post_event(appmod, {"type": "customer.subscription.deleted",
                         "data": {"object": {"id": "sub_plan", "metadata": {"kind": "course_plan"}}}})
    assert _paid_level(appmod, "q@x.com", now=500) == 0  # locked (no cert yet)


def test_plan_handler_skips_membership_kind(appmod, monkeypatch):
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription",
                        lambda sid: {"id": "s", "status": "active", "current_period_end": 1000,
                                     "customer": "c", "metadata": {"kind": "course_membership", "email": "m@x.com"}})
    monkeypatch.setattr(stripe_pay, "cancel_subscription", lambda sid: {"id": sid, "status": "canceled"})
    _post_event(appmod, {"type": "invoice.paid", "data": {"object": {"subscription": "s", "id": "in_x"}}})
    import sqlite3
    with sqlite3.connect(appmod.LOG_DB) as cx:
        try:
            n = cx.execute("SELECT COUNT(*) FROM course_plan_charges WHERE sub_id='s'").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
    assert n == 0  # plan handler did not record a course_membership invoice
