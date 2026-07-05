import sqlite3
from unittest import mock
import requests
import app as appmod
from dashboard import coach_subscriptions as _cs


def _client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _seed(email, tier, next_at):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        _cs.create_sub(cx, email=email, tier=tier, customer_id="cus", payment_method_id="pm",
                       next_charge_at=next_at); cx.commit()


def _hdr():
    return {"X-Console-Key": appmod.CONSOLE_SECRET}


def test_cron_requires_key():
    assert _client().post("/api/cron/coach-subscriptions/charge").status_code == 401


def test_cron_charges_due_grants_and_advances():
    c = _client(); _seed("due@x.com", "rae", "2026-01-01")   # far-past → due
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": "pi_ok", "status": "succeeded"}), \
         mock.patch("dashboard.evox.add_session_credits", return_value=1) as grant:
        d = c.post("/api/cron/coach-subscriptions/charge", headers=_hdr()).get_json()
    assert d["charged"] == 1 and grant.called
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        s = _cs.get_sub(cx, "due@x.com")
        assert s["next_charge_at"] > "2026-01-01" and s["last_charged_at"]   # advanced


def test_cron_skips_future():
    c = _client(); _seed("future@x.com", "glen", "2099-01-01")   # future → not due
    with mock.patch("dashboard.stripe_pay.charge_off_session") as charge:
        d = c.post("/api/cron/coach-subscriptions/charge", headers=_hdr()).get_json()
    assert d["charged"] == 0 and not charge.called                # never charged early


def test_cron_one_sub_exception_does_not_abort_batch():
    c = _client()
    _seed("boom@x.com", "rae", "2026-01-01")     # due first (earlier date), raises
    _seed("ok@x.com", "rae", "2026-01-02")       # due second, succeeds
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    side_effect=[requests.Timeout("connect timed out"),
                                 {"id": "pi_ok2", "status": "succeeded"}]), \
         mock.patch("dashboard.evox.add_session_credits", return_value=1) as grant, \
         mock.patch.object(appmod, "send_evox_email"):
        resp = c.post("/api/cron/coach-subscriptions/charge", headers=_hdr())
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["charged"] == 1 and d["failed"] == 1
    assert grant.called
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        boom = _cs.get_sub(cx, "boom@x.com")
        ok = _cs.get_sub(cx, "ok@x.com")
        # first sub: exception recorded as a failure, not silently skipped
        assert boom["status"] == "past_due" and boom["fail_count"] == 1
        assert boom["next_charge_at"] == "2026-01-01"          # not advanced
        # second sub: still processed and charged despite the first sub's exception
        assert ok["next_charge_at"] > "2026-01-02" and ok["last_charged_at"]


def test_cron_failed_charge_past_due_no_grant():
    c = _client(); _seed("fail@x.com", "rae", "2026-01-01")
    with mock.patch("dashboard.stripe_pay.charge_off_session",
                    return_value={"id": None, "status": "failed"}), \
         mock.patch("dashboard.evox.add_session_credits") as grant, \
         mock.patch.object(appmod, "send_evox_email"):
        d = c.post("/api/cron/coach-subscriptions/charge", headers=_hdr()).get_json()
    assert d["failed"] == 1 and not grant.called
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row; _cs.init_sub_tables(cx)
        s = _cs.get_sub(cx, "fail@x.com")
        assert s["status"] == "past_due" and s["fail_count"] == 1
        assert s["next_charge_at"] == "2026-01-01"                 # NOT advanced
