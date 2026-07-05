import sqlite3
from unittest import mock
import app as appmod
from dashboard import coach_subscriptions as _cs


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _tok(email="m@x.com"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import evox as _ev, client_portal as _cp
        _ev.init_evox_tables(cx); _cp.init_client_portal_table(cx); _cs.init_sub_tables(cx)
        t = _ev.ensure_portal_token(cx, email, "Mel"); cx.commit()
    return t


def test_subscribe_returns_checkout_url():
    c = _client(); tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True), \
         mock.patch("dashboard.stripe_pay.create_checkout_session",
                    return_value={"id": "cs_1", "url": "https://stripe/cs_1"}):
        r = c.post(f"/api/community/coach-subscribe?token={tok}", json={"tier": "rae"})
    assert r.get_json()["url"] == "https://stripe/cs_1"


def test_subscribe_bad_tier_400():
    c = _client(); tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True):
        r = c.post(f"/api/community/coach-subscribe?token={tok}", json={"tier": "nope"})
    assert r.status_code == 400


def test_fulfill_creates_sub_grants_once():
    _tok()
    fake_session = {"metadata": {"kind": "coach_sub", "tier": "rae", "email": "m@x.com"},
                    "payment_intent": "pi_1"}
    fake_pi = {"status": "succeeded", "customer": "cus_1", "payment_method": "pm_1"}
    with mock.patch("dashboard.stripe_pay.get_session", return_value=fake_session), \
         mock.patch("dashboard.stripe_pay.get_payment_intent", return_value=fake_pi), \
         mock.patch("dashboard.evox.add_session_credits", return_value=1) as grant, \
         mock.patch.object(appmod, "send_evox_email"):
        appmod._fulfill_coach_sub("cs_evt_1")
        appmod._fulfill_coach_sub("cs_evt_1")   # webhook + return double-delivery
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        s = _cs.get_sub(cx, "m@x.com")
        assert s["tier"] == "rae" and s["status"] == "active"
        n = cx.execute("SELECT COUNT(*) FROM coach_sub_charges WHERE member_email='m@x.com'").fetchone()[0]
    assert grant.call_count == 1 and n == 1        # granted + charged exactly once


def test_fulfill_ignores_unpaid():
    with mock.patch("dashboard.stripe_pay.get_session",
                    return_value={"metadata": {"kind": "coach_sub", "tier": "glen", "email": "u@x.com"},
                                  "payment_intent": "pi_x"}), \
         mock.patch("dashboard.stripe_pay.get_payment_intent",
                    return_value={"status": "requires_payment_method", "customer": None, "payment_method": None}):
        appmod._fulfill_coach_sub("cs_evt_2")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        assert _cs.get_sub(cx, "u@x.com") is None   # no sub on an unpaid session


def test_cancel_sets_canceled():
    c = _client(); tok = _tok("cxl@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        _cs.create_sub(cx, email="cxl@x.com", tier="rae", customer_id="c",
                       payment_method_id="p", next_charge_at="2026-08-01"); cx.commit()
    r = c.post(f"/api/community/coach-subscribe/cancel?token={tok}")
    assert r.get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        assert _cs.get_sub(cx, "cxl@x.com")["status"] == "canceled"
