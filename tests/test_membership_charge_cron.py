"""Tests for the membership branch of /api/cron/charge-subscriptions
(Mechanic 1 — program-bundled live-group coaching).

Mirrors the harness in test_subscriptions_cron.py: seeds against the real
LOG_DB, stubs all external calls, posts to the cron endpoint with the secret
header. Here we seed ONE due kind='membership' subscription and assert it is
charged the flat amount_cents off-session and recorded as a membership order.
"""
import os
import sqlite3

import app as appmod
from dashboard import subscriptions as subs

TEST_EMAIL = "mbr-test@example.com"


def _cron_secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


def _seed_membership(cx, *, amount_cents=9900, next_charge_date="2000-01-01"):
    """Insert one due membership subscription and return its id."""
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
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


def _enable(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())


def test_membership_charge_flat_amount_and_advances(monkeypatch):
    _enable(monkeypatch)

    charge_amounts = []
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: charge_amounts.append(a[2]) or {"status": "succeeded", "id": "pi_x"},
    )
    monkeypatch.setattr(appmod.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_send_subscription_email",
                        lambda *a, **k: ("smtp", None))

    ingested = []
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: ingested.append(kw))

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_membership(cx, amount_cents=9900, next_charge_date="2000-01-01")

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["charged"] >= 1

    # Charged the flat amount, not a product-priced total
    assert 9900 in charge_amounts

    # Order recorded as a membership for the flat amount
    mbr = [o for o in ingested if o.get("source") == "membership"]
    assert len(mbr) == 1
    assert mbr[0]["total_cents"] == 9900

    # next_charge_date advanced by 1 month
    s = subs.get(cx, sid)
    assert s["next_charge_date"] == subs.add_months("2000-01-01", 1)

    cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL,)); cx.commit()
    cx.close()


TEST_EMAIL_B = "mbr-backfill-covered@example.com"
TEST_EMAIL_C = "mbr-backfill-lapsed@example.com"


def _seed_membership_for(cx, email, *, amount_cents=9900, next_charge_date="2026-07-01"):
    """Seed an active membership subscription for a specific email."""
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (email,))
    cx.commit()
    sid = subs.create_membership(
        cx,
        email=email,
        stripe_customer_id=f"cus_bf_{email[:6]}",
        stripe_payment_method_id=f"pm_bf_{email[:6]}",
        amount_cents=amount_cents,
        next_charge_date=next_charge_date,
        cadence_months=1,
    )
    cx.commit()
    return sid


def test_backfill_extends_lapsed_leaves_covered_monotonic(monkeypatch):
    """Backfill cron: lapsed grant is extended; already-covered grant is NOT touched."""
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())

    appmod._init_membership_tables()
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row

    next_date = "2026-07-01"
    from datetime import datetime as _dt, timedelta as _td

    # Email B: grant already covers next_charge_date + grace (well into the future)
    _seed_membership_for(cx, TEST_EMAIL_B, next_charge_date=next_date)
    cx.execute("DELETE FROM memberships WHERE email=?", (TEST_EMAIL_B,))
    far_future = (_dt.fromisoformat(next_date) + _td(days=10)).isoformat() + "Z"
    cx.execute(
        "INSERT INTO memberships (id,email,granted_at,expires_at,granted_by,source,truly_vip_ref,notes)"
        " VALUES ('gbf1',?,?,?,'seed','seed','','')",
        (TEST_EMAIL_B, "2026-01-01T00:00:00Z", far_future),
    )
    cx.commit()
    covered_max_before = cx.execute(
        "SELECT MAX(expires_at) FROM memberships WHERE email=?", (TEST_EMAIL_B,)
    ).fetchone()[0]

    # Email C: grant is absent (lapsed)
    _seed_membership_for(cx, TEST_EMAIL_C, next_charge_date=next_date)
    cx.execute("DELETE FROM memberships WHERE email=?", (TEST_EMAIL_C,))
    cx.commit()

    c = appmod.app.test_client()
    r = c.post("/api/cron/backfill-membership-grants", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["dry_run"] is False
    # At least the lapsed member was fixed
    assert body["fixed"] >= 1

    # Lapsed member (C) now has an active grant reaching next_charge_date + grace
    m = appmod._active_membership_for_email(TEST_EMAIL_C)
    assert m is not None, "lapsed member should now have an active membership grant"

    # Covered member (B) was NOT given an additional grant row (monotonic)
    covered_max_after = cx.execute(
        "SELECT MAX(expires_at) FROM memberships WHERE email=?", (TEST_EMAIL_B,)
    ).fetchone()[0]
    assert covered_max_after == covered_max_before, (
        "already-covered member should not have received a redundant grant"
    )

    # Cleanup
    for email in (TEST_EMAIL_B, TEST_EMAIL_C):
        cx.execute("DELETE FROM subscriptions WHERE email=?", (email,))
        cx.execute("DELETE FROM memberships WHERE email=?", (email,))
    cx.commit()
    cx.close()


def test_membership_dry_run_charges_nothing(monkeypatch):
    _enable(monkeypatch)

    charge_amounts = []
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: charge_amounts.append(a[2]) or {"status": "succeeded", "id": "pi_x"},
    )
    monkeypatch.setattr(appmod.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_send_subscription_email",
                        lambda *a, **k: ("smtp", None))

    ingested = []
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: ingested.append(kw))

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_membership(cx, amount_cents=9900, next_charge_date="2000-01-01")

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions?dry_run=1", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["dry_run"] is True
    assert body["charged"] >= 1  # dry preview counts the due membership

    # No charge, no order ingested, no advance
    assert charge_amounts == []
    assert ingested == []
    s = subs.get(cx, sid)
    assert s["next_charge_date"] == "2000-01-01"

    cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL,)); cx.commit()
    cx.close()


def test_successful_membership_charge_extends_access_grant(monkeypatch):
    _enable(monkeypatch)
    _stub_externals(monkeypatch, [])
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    # membership due today; a stale 1-day grant
    from datetime import date, timedelta
    sid = _seed_membership(cx, next_charge_date=date.today().isoformat())
    appmod._init_membership_tables()
    cx.execute("DELETE FROM memberships WHERE email=?", (TEST_EMAIL,))
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,granted_by,source,truly_vip_ref,notes)"
               " VALUES ('g0',?,?,?, 'seed','seed','','')",
               (TEST_EMAIL, "2000-01-01T00:00:00Z", (date.today()+timedelta(days=1)).isoformat()+"T00:00:00Z"))
    cx.commit()
    appmod.app.test_client().post("/api/cron/charge-subscriptions", headers=_headers())
    # grant now reaches ~ next_charge_date(+1 month) + 3 days grace
    m = appmod._active_membership_for_email(TEST_EMAIL)
    assert m is not None and m["days_remaining"] >= 30
    cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL,))
    cx.execute("DELETE FROM memberships WHERE email=?", (TEST_EMAIL,)); cx.commit(); cx.close()
