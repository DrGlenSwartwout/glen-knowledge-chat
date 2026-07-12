import sqlite3
from unittest import mock
import pytest
import requests
import app as appmod
from dashboard import family_plan as fp


@pytest.fixture(autouse=True)
def _clean_family_subs():
    """family_plan.due() (by design) returns BOTH active and past_due rows so
    bounded dunning can progress across cron runs. That means a past_due row
    left behind by one test is still 'due' and would leak into the next test
    against the shared appmod.LOG_DB. Reset the two family-plan tables before
    each test so each of the 8 cases below starts from a clean slate."""
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        fp.init_family_plan_table(cx)
        cx.execute("DELETE FROM family_subscriptions")
        cx.execute("DELETE FROM family_sub_charges")
        cx.commit()
    yield


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email, next_at, *, source="stripe", fail_count=0, status="active"):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        fp.activate(cx, email, next_charge_at=next_at, customer_id="cus",
                    payment_method_id="pm", source=source)
        if fail_count or status != "active":
            cx.execute("UPDATE family_subscriptions SET fail_count=?, status=? "
                       "WHERE caregiver_email=?", (fail_count, status, email.lower()))
        cx.commit()


def _hdr():
    return {"X-Console-Key": appmod.CONSOLE_SECRET}


def test_cron_requires_key():
    assert _client().post("/api/cron/family-plan/charge").status_code == 401


def test_cron_charges_due_and_advances():
    _seed("due@x.com", "2026-01-01")
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": "pi_ok", "status": "succeeded"}):
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["charged"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "due@x.com")
        assert s["next_charge_at"] > "2026-01-01" and s["last_charged_at"]


def test_cron_skips_future():
    _seed("future@x.com", "2099-01-01")
    with mock.patch("dashboard.stripe_pay.charge_off_session") as charge:
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["charged"] == 0 and not charge.called


def test_cron_never_charges_a_comp():
    _seed("comp@x.com", None, source="comp")
    with mock.patch("dashboard.stripe_pay.charge_off_session") as charge:
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["charged"] == 0 and not charge.called


def test_cron_failed_charge_goes_past_due_no_advance():
    _seed("fail@x.com", "2026-01-01")
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": None, "status": "failed"}), \
         mock.patch.object(appmod, "send_evox_email"):
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["failed"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "fail@x.com")
        assert s["status"] == "past_due" and s["fail_count"] == 1
        assert s["next_charge_at"] == "2026-01-01"


def test_cron_third_failure_cancels_and_stops_cover():
    _seed("dead@x.com", "2026-01-01", fail_count=2)   # already failed twice
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": None, "status": "failed"}), \
         mock.patch.object(appmod, "send_evox_email"):
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["failed"] == 1 and d["cancelled"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "dead@x.com")
        assert s["status"] == "cancelled" and fp.is_active(cx, "dead@x.com") is False


def test_cron_retries_a_past_due_sub_and_recovers_on_success():
    _seed("grace@x.com", "2026-01-01", fail_count=1, status="past_due")  # prior failure, in grace
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": "pi_ok", "status": "succeeded"}):
        d = _client().post("/api/cron/family-plan/charge", headers=_hdr()).get_json()
    assert d["charged"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        s = fp.get(cx, "grace@x.com")
        assert s["status"] == "active" and s["fail_count"] == 0      # recovered
        assert s["next_charge_at"] > "2026-01-01"                    # advanced


def test_cron_one_exception_does_not_abort_batch():
    _seed("boom@x.com", "2026-01-01")     # due first, raises
    _seed("ok@x.com", "2026-01-02")       # due second, succeeds
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    side_effect=[requests.Timeout("timed out"),
                                 {"id": "pi_ok2", "status": "succeeded"}]), \
         mock.patch.object(appmod, "send_evox_email"):
        resp = _client().post("/api/cron/family-plan/charge", headers=_hdr())
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["charged"] == 1 and d["failed"] == 1
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; fp.init_family_plan_table(cx)
        assert fp.get(cx, "boom@x.com")["next_charge_at"] == "2026-01-01"   # not advanced
        assert fp.get(cx, "ok@x.com")["next_charge_at"] > "2026-01-02"      # advanced
