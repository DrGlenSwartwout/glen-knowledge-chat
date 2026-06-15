"""Tests for membership-aware reminder copy + cancel-stops-charge
(Mechanic 1 — program-bundled live-group coaching).

Mirrors the harness in test_membership_charge_cron.py: seeds against the real
LOG_DB, stubs all external calls, posts to the cron endpoint with the secret
header. Assertions filter by the test email so they are robust to the shared
LOG_DB being populated by other tests.
"""
import os
import sqlite3
from datetime import date, timedelta

import app as appmod
from dashboard import subscriptions as subs

TEST_EMAIL = "mbr-reminder-test@example.com"


def _cron_secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


def _enable(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())


def _seed_membership(cx, *, amount_cents=9900, next_charge_date):
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL,))
    cx.commit()
    sid = subs.create_membership(
        cx,
        email=TEST_EMAIL,
        stripe_customer_id="cus_mbr",
        stripe_payment_method_id="pm_mbr",
        amount_cents=amount_cents,
        next_charge_date=next_charge_date,
        cadence_months=1,
    )
    cx.commit()
    return sid


def _stub_externals(monkeypatch, charged_ids):
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: charged_ids.append((k.get("metadata") or {}).get("sub"))
        or {"status": "succeeded", "id": "pi_x"},
    )
    monkeypatch.setattr(appmod.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)


def test_cancelled_membership_not_charged(monkeypatch):
    _enable(monkeypatch)

    charged_ids = []
    _stub_externals(monkeypatch, charged_ids)
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_membership(cx, next_charge_date="2000-01-01")
    subs.set_status(cx, sid, "cancelled")
    cx.commit()

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data
    assert r.get_json()["ok"] is True

    # The cancelled sub was NOT charged (its sub id never reached stripe).
    assert str(sid) not in [str(x) for x in charged_ids if x is not None]

    # Still cancelled, date untouched.
    s = subs.get(cx, sid)
    assert s["status"] == "cancelled"
    assert s["next_charge_date"] == "2000-01-01"

    cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL,)); cx.commit()
    cx.close()


def test_membership_heads_up_uses_membership_copy(monkeypatch):
    _enable(monkeypatch)

    charged_ids = []
    _stub_externals(monkeypatch, charged_ids)

    sent = []
    monkeypatch.setattr(appmod, "_send_subscription_email",
                        lambda email, kind, data: sent.append((email, kind, data)) or ("smtp", None))

    # next_charge_date 2 days from today: inside the 3-day heads-up window,
    # and strictly > today so it is not also due for charging.
    two_days = (date.today() + timedelta(days=2)).isoformat()

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_membership(cx, next_charge_date=two_days)

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data
    assert r.get_json()["ok"] is True

    # Find the heads-up email for our test sub (filter by email — shared LOG_DB).
    ours = [s for s in sent if s[0] == TEST_EMAIL and s[1] == "heads_up"]
    assert len(ours) == 1, f"expected one heads_up for {TEST_EMAIL}, got {sent}"
    _email, _kind, data = ours[0]
    assert data.get("kind") == "membership"

    cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL,)); cx.commit()
    cx.close()
