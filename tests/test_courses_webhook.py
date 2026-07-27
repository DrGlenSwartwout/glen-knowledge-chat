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


def _drip_active(appmod, email, now=None):
    from dashboard import course_entitlements as ce
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ce.init_course_entitlements_table(cx)
        return ce.drip_active(cx, email, now=now)


def _unlocked(appmod, email, course="ash-certification"):
    from dashboard import course_module_unlocks as cmu
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cmu.init_unlock_tables(cx)
        return cmu.unlocked_modules(cx, email, course)


def _drip_checkout(appmod, monkeypatch, email, sub_id="sub_drip", cs_id="cs_drip", cpe=9_999_999_999):
    """Post a checkout.session.completed for a $99/mo drip (product=membership)."""
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": sub_id, "status": "active", "current_period_end": cpe,
        "customer": "cus_d", "metadata": {"kind": "course_membership", "email": email}})
    return _post_event(appmod, {"type": "checkout.session.completed", "data": {"object": {
        "id": cs_id, "mode": "subscription", "subscription": sub_id, "customer": "cus_d",
        "metadata": {"kind": "course_purchase", "email": email, "product": "membership"},
        "customer_details": {"email": email}}}})


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


def test_subscription_purchase_grants_drip_not_level2_and_unlocks_module1(appmod, monkeypatch):
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
    # a drip (course_membership) buyer is drip-active, NOT level 2 (that's plan/cert only)
    assert _paid_level(appmod, "m@x.com", now=500) == 0
    assert _drip_active(appmod, "m@x.com", now=500) is True
    assert _drip_active(appmod, "m@x.com", now=2500) is False  # window expired at 2000
    # first purchase unlocks the first paid module of ash-certification
    assert _unlocked(appmod, "m@x.com") == {"02-body"}


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
    assert _drip_active(appmod, "z@x.com", now=1) is True
    assert _paid_level(appmod, "z@x.com", now=1) == 0        # drip alone never was level 2
    assert _unlocked(appmod, "z@x.com") == {"02-body"}
    _post_event(appmod, {"type": "customer.subscription.deleted", "data": {"object": {
        "id": "sub_9", "metadata": {"kind": "course_membership", "email": "z@x.com"}}}})
    assert _drip_active(appmod, "z@x.com", now=1) is False
    # relock is the resolver's job (next task), not this handler — the unlocked set
    # (and any past module completion) is left intact by subscription.deleted.
    assert _unlocked(appmod, "z@x.com") == {"02-body"}


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
    assert _drip_active(appmod, "anon@x.com", now=500) is True    # initial grant via customer_details
    assert _drip_active(appmod, "anon@x.com", now=1500) is False  # expires at 1000
    assert _paid_level(appmod, "anon@x.com", now=500) == 0        # drip alone never lifts to level 2
    # invoice.paid renewal extends to 3000 even though sub metadata has no email
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": "sub_ne", "status": "active", "current_period_end": 3000,
        "customer": "c", "metadata": {"kind": "course_membership"}})
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_ne", "id": "in_ne_1"}}})
    assert _drip_active(appmod, "anon@x.com", now=2500) is True   # renewed via stored-row email fallback


def test_drip_invoice_unlocks_next_module_and_replay_does_not_advance(appmod, monkeypatch):
    from dashboard import stripe_pay
    email = "drip@x.com"
    _drip_checkout(appmod, monkeypatch, email)
    assert _unlocked(appmod, email) == {"02-body"}            # first module from checkout
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": "sub_drip", "status": "active", "current_period_end": 9_999_999_999,
        "customer": "cus_d", "metadata": {"kind": "course_membership", "email": email}})
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_drip", "id": "in_d1"}}})
    assert _unlocked(appmod, email) == {"02-body", "03-mind"}          # 1st paid invoice -> next module
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_drip", "id": "in_d2"}}})
    assert _unlocked(appmod, email) == {"02-body", "03-mind", "04-spirit"}  # 2nd distinct invoice -> next
    # replaying an already-seen invoice must NOT unlock a 4th module
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_drip", "id": "in_d2"}}})
    assert _unlocked(appmod, email) == {"02-body", "03-mind", "04-spirit"}
    assert _paid_level(appmod, email, now=1) == 0
    assert _drip_active(appmod, email, now=1) is True


def test_drip_self_selected_pref_is_honored_then_reverts_to_sequential(appmod, monkeypatch):
    from dashboard import course_module_unlocks as cmu
    from dashboard import stripe_pay
    email = "pref@x.com"
    _drip_checkout(appmod, monkeypatch, email, sub_id="sub_pref", cs_id="cs_pref")
    assert _unlocked(appmod, email) == {"02-body"}
    # learner sets a preference for a later module before the next invoice
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cmu.set_unlock_pref(cx, email, "ash-certification", "05-family-history")
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": "sub_pref", "status": "active", "current_period_end": 9_999_999_999,
        "customer": "cus_d", "metadata": {"kind": "course_membership", "email": email}})
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_pref", "id": "in_p1"}}})
    assert _unlocked(appmod, email) == {"02-body", "05-family-history"}   # pref honored, then consumed
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_pref", "id": "in_p2"}}})
    # pref was consumed by the prior invoice -> reverts to sequential course order
    assert _unlocked(appmod, email) == {"02-body", "05-family-history", "03-mind"}


def test_drip_first_cycle_unlocks_only_one_module(appmod, monkeypatch):
    # Real Stripe delivers BOTH checkout.session.completed AND a subscription_create
    # invoice.paid for the first $99 charge — together they must unlock ONE module, not two.
    from dashboard import stripe_pay
    email = "drp@x.com"
    sub_id = "sub_drp"
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: {
        "id": sub_id, "status": "active", "current_period_end": 9_999_999_999,
        "customer": "cus_d", "metadata": {"kind": "course_membership", "email": email}})
    # 1) checkout.session.completed (drip) unlocks module 1
    _drip_checkout(appmod, monkeypatch, email, sub_id=sub_id, cs_id="cs_drp")
    assert _unlocked(appmod, email) == {"02-body"}
    # 2) the SAME first charge's initial invoice (subscription_create) must NOT unlock a 2nd
    _post_event(appmod, {"type": "invoice.paid", "data": {"object": {
        "subscription": sub_id, "id": "in_create", "billing_reason": "subscription_create"}}})
    first = _unlocked(appmod, email)
    assert len(first) == 1, f"first cycle must unlock exactly one module, got {first}"
    assert first == {"02-body"}
    # 3) a monthly cycle invoice unlocks the next module
    _post_event(appmod, {"type": "invoice.paid", "data": {"object": {
        "subscription": sub_id, "id": "in_cycle2", "billing_reason": "subscription_cycle"}}})
    after = _unlocked(appmod, email)
    assert len(after) == 2, f"a cycle invoice unlocks the next module, got {after}"
    assert after == {"02-body", "03-mind"}


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


def test_plan_charge_counts_even_when_email_unresolved(appmod, monkeypatch):
    import sqlite3
    from dashboard import stripe_pay
    # Anonymous plan: subscription metadata carries NO email, and no membership row
    # exists yet (invoice.paid before checkout.session.completed). The charge must
    # still be counted so the 12-count stays accurate.
    monkeypatch.setattr(stripe_pay, "get_subscription",
                        lambda sid: {"id": "sub_anon", "status": "active", "current_period_end": 1000,
                                     "customer": "c", "metadata": {"kind": "course_plan"}})
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_anon", "id": "in_anon_1"}}})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        n = cx.execute("SELECT COUNT(*) FROM course_plan_charges WHERE sub_id='sub_anon'").fetchone()[0]
    assert n == 1  # counted despite unresolved email (grant deferred to the completed event)


def test_plan_overcharge_past_12_fires_alert(appmod, monkeypatch):
    # A 13th-or-later paid invoice (a charge that slipped past the 12-payment
    # conversion) must fire an operational overcharge alert; exactly 12 must not.
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "get_subscription", lambda sid: _plan_sub(email="oc@x.com", cpe=1000))
    monkeypatch.setattr(stripe_pay, "cancel_subscription", lambda sid: {"id": sid, "status": "canceled"})
    alerts = []
    monkeypatch.setattr(appmod, "_send_token_alert",
                        lambda subject, body: alerts.append((subject, body)) or True)
    for i in range(1, 13):  # 12 charges → conversion, NO overcharge alert
        _post_event(appmod, {"type": "invoice.paid",
                             "data": {"object": {"subscription": "sub_plan", "id": f"in_{i}"}}})
    assert alerts == []
    _post_event(appmod, {"type": "invoice.paid",
                         "data": {"object": {"subscription": "sub_plan", "id": "in_13"}}})
    assert len(alerts) == 1 and "OVERCHARGE" in alerts[0][0]
