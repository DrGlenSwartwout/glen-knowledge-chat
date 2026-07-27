import sqlite3
from unittest import mock
import pytest
import app as appmod
from dashboard import family_plan as fp


@pytest.fixture(autouse=True)
def _clean_family_plan_state():
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        fp.init_family_plan_table(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS family_sub_grants "
                   "(session_id TEXT PRIMARY KEY, email TEXT, created_at TEXT)")
        cx.execute("DELETE FROM family_subscriptions")
        cx.execute("DELETE FROM family_sub_charges")
        cx.execute("DELETE FROM family_sub_grants")
        cx.commit()
    yield


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _tok(email="care@x.com", name="Care"):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx); fp.init_family_plan_table(cx)
        cp.upsert_portal(cx, email, name, {})     # ensure the client_portals row exists
        token = cp.reissue_token(cx, email)        # always returns a fresh raw token for an existing row
        cx.commit()
    return token


def test_subscribe_returns_checkout_url():
    tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True), \
         mock.patch.object(appmod, "_family_plan_enabled", return_value=True), \
         mock.patch("dashboard.stripe_pay.create_checkout_session",
                    return_value={"id": "cs_1", "url": "https://stripe/cs_1"}):
        r = _client().post(f"/api/portal/{tok}/family-plan/subscribe")
    assert r.status_code == 200 and r.get_json()["url"] == "https://stripe/cs_1"


def test_subscribe_flag_off_404():
    tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True), \
         mock.patch.object(appmod, "_family_plan_enabled", return_value=False):
        r = _client().post(f"/api/portal/{tok}/family-plan/subscribe")
    assert r.status_code == 404


def test_subscribe_stripe_inactive_503():
    tok = _tok()
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", False), \
         mock.patch.object(appmod, "_family_plan_enabled", return_value=True):
        r = _client().post(f"/api/portal/{tok}/family-plan/subscribe")
    assert r.status_code == 503


def test_public_annual_checkout_uses_1497_and_annual_tier():
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True), \
         mock.patch.object(appmod, "_family_plan_enabled", return_value=True), \
         mock.patch.object(appmod, "_family_upgrade_credit", return_value=0), \
         mock.patch("dashboard.stripe_pay.create_checkout_session",
                    return_value={"id": "cs_year", "url": "https://stripe/cs_year"}) as create:
        r = _client().post("/family-plan/checkout", json={
            "email": "family@x.com", "tier": "family_annual"})
    assert r.status_code == 200
    assert r.get_json()["amount_cents"] == 149700
    args, kwargs = create.call_args
    assert args[0] == 149700
    assert kwargs["metadata"]["tier"] == "family_annual"
    assert kwargs["save_card"] is True


def test_public_household_upgrade_applies_99_credit():
    with mock.patch.object(appmod, "_STRIPE_ACTIVE", True), \
         mock.patch.object(appmod, "_family_plan_enabled", return_value=True), \
         mock.patch.object(appmod, "_family_upgrade_credit", return_value=9900), \
         mock.patch("dashboard.stripe_pay.create_checkout_session",
                    return_value={"id": "cs_up", "url": "https://stripe/cs_up"}) as create:
        r = _client().post("/family-plan/checkout", json={
            "email": "member@x.com", "tier": "family_monthly"})
    assert r.get_json()["amount_cents"] == 4800
    args, kwargs = create.call_args
    assert args[0] == 4800
    assert kwargs["metadata"]["upgrade_credit_cents"] == "9900"


def test_fulfill_activates_and_charges_once():
    fake_session = {"metadata": {"kind": "family_plan", "email": "f@x.com"},
                    "payment_intent": "pi_1"}
    fake_pi = {"status": "succeeded", "customer": "cus_1", "payment_method": "pm_1"}
    with mock.patch("dashboard.stripe_pay.get_session", return_value=fake_session), \
         mock.patch("dashboard.stripe_pay.get_payment_intent", return_value=fake_pi), \
         mock.patch.object(appmod, "send_evox_email"):
        appmod._fulfill_family_plan("cs_evt_1")
        appmod._fulfill_family_plan("cs_evt_1")   # webhook + return double-delivery
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "f@x.com")
        assert s["status"] == "active" and s["stripe_customer_id"] == "cus_1"
        assert s["next_charge_at"] and s["source"] == "stripe"
        n = cx.execute("SELECT COUNT(*) FROM family_sub_charges "
                       "WHERE caregiver_email='f@x.com'").fetchone()[0]
    assert n == 1                                  # charged exactly once


def test_fulfill_annual_sets_twelve_month_cadence():
    fake_session = {
        "metadata": {"kind": "family_plan", "email": "year@x.com",
                     "tier": "family_annual", "upgrade_credit_cents": "0"},
        "payment_intent": "pi_year", "amount_total": 149700}
    fake_pi = {"status": "succeeded", "customer": "cus_y", "payment_method": "pm_y"}
    with mock.patch("dashboard.stripe_pay.get_session", return_value=fake_session), \
         mock.patch("dashboard.stripe_pay.get_payment_intent", return_value=fake_pi), \
         mock.patch.object(appmod, "send_evox_email"):
        assert appmod._fulfill_family_plan("cs_year_fulfill")["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        fp.init_family_plan_table(cx)
        sub = fp.get(cx, "year@x.com")
    assert sub["amount_cents"] == 149700
    assert sub["cadence_months"] == 12


def test_fulfill_ignores_unpaid():
    with mock.patch("dashboard.stripe_pay.get_session",
                    return_value={"metadata": {"kind": "family_plan", "email": "u@x.com"},
                                  "payment_intent": "pi_x"}), \
         mock.patch("dashboard.stripe_pay.get_payment_intent",
                    return_value={"status": "requires_payment_method",
                                  "customer": None, "payment_method": None}):
        appmod._fulfill_family_plan("cs_evt_2")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        assert fp.get(cx, "u@x.com") is None


def test_cancel_sets_cancelled_and_stops_cover():
    tok = _tok("cxl2@x.com")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        fp.activate(cx, "cxl2@x.com", next_charge_at="2026-08-01",
                    customer_id="c", payment_method_id="p"); cx.commit()
    r = _client().post(f"/api/portal/{tok}/family-plan/cancel")
    assert r.get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        assert fp.get(cx, "cxl2@x.com")["status"] == "cancelled"
        assert fp.is_active(cx, "cxl2@x.com") is False
