"""/console/members API (PR3): Trial / Full / Paused buckets, trial rows carry the
accrued upgrade credit. Console-key gated (Glen's secret or Rae's owner token)."""
import json
import sqlite3

import app as appmod
from dashboard import subscriptions as subs
from dashboard import orders as orders_mod

SECRET = "members-test-secret"
TRIAL = "members-trial@example.com"
FULL = "members-full@example.com"
PAUSED = "members-paused@example.com"
CANC = "members-cancelled@example.com"
EXPECTED_TRIAL_CREDIT = 0  # trial-credit machinery retired; always 0


def _clean(cx):
    for em in (TRIAL, FULL, PAUSED, CANC):
        cx.execute("DELETE FROM subscriptions WHERE email=?", (em,))
        cx.execute("DELETE FROM orders WHERE lower(email)=?", (em.lower(),))
    cx.commit()


def _mk_membership(cx, email):
    return subs.create_membership(
        cx, email=email, stripe_customer_id="cus", stripe_payment_method_id="pm",
        amount_cents=9900, next_charge_date="2026-07-01")


def _seed(cx):
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    orders_mod.init_orders_table(cx)
    _clean(cx)
    from datetime import date
    today = date.today().isoformat()
    # Trial member with in-window orders -> non-zero accrued credit.
    _mk_membership(cx, TRIAL)
    cx.execute("INSERT INTO orders (created_at, source, external_ref, email, items_json, "
               "total_cents, status) VALUES (?, 'biofield_trial', ?, ?, ?, 100, 'new')",
               (today + "T00:00:00+00:00", f"bt-{TRIAL}", TRIAL,
                json.dumps([{"name": "Biofield Analysis", "qty": 1}])))
    cx.execute("INSERT INTO orders (created_at, source, external_ref, email, items_json, "
               "total_cents, status) VALUES (?, 'reorder', ?, ?, ?, 0, 'new')",
               (today + "T01:00:00+00:00", f"re-{TRIAL}", TRIAL,
                json.dumps([{"name": "Bone Builder", "qty": 3}])))
    # Full member (one $99 charge cleared).
    fsid = _mk_membership(cx, FULL)
    subs.advance_after_charge(cx, fsid)
    # Paused member.
    psid = _mk_membership(cx, PAUSED)
    subs.set_skip_next(cx, psid, True)
    # Cancelled -> must NOT appear.
    csid = _mk_membership(cx, CANC)
    subs.set_status(cx, csid, "cancelled")
    cx.commit()


def test_requires_console_key(monkeypatch):
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", SECRET)
    r = appmod.app.test_client().get("/api/console/members")
    assert r.status_code == 401


def test_buckets_members_into_columns_with_fields(monkeypatch):
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", SECRET)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _seed(cx)

    r = appmod.app.test_client().get(f"/api/console/members?key={SECRET}")
    assert r.status_code == 200, r.data
    body = r.get_json()
    b = body["buckets"]

    trial_emails = {row["email"] for row in b["trial"]}
    full_emails = {row["email"] for row in b["full"]}
    paused_emails = {row["email"] for row in b["paused"]}

    assert TRIAL in trial_emails
    assert FULL in full_emails
    assert PAUSED in paused_emails
    # Cancelled member is excluded from every bucket.
    all_emails = trial_emails | full_emails | paused_emails
    assert CANC not in all_emails

    trial_row = next(row for row in b["trial"] if row["email"] == TRIAL)
    assert trial_row["credit_cents"] == EXPECTED_TRIAL_CREDIT
    assert trial_row["plan_cents"] == 9900
    assert trial_row["category"] == "trial"

    full_row = next(row for row in b["full"] if row["email"] == FULL)
    assert "credit_cents" not in full_row

    paused_row = next(row for row in b["paused"] if row["email"] == PAUSED)
    assert paused_row.get("resume_date")  # paused carries a resume date

    # counts mirror the buckets.
    assert body["counts"]["trial"] == len(b["trial"])

    _clean(cx); cx.close()
