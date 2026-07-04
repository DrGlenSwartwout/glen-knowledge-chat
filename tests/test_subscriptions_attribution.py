import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    return cx


def test_migration_idempotent():
    cx = _cx()
    subs.migrate_add_attribution_column(cx)  # second call must not raise
    subs.migrate_add_consent_column(cx)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(subscriptions)")}
    assert "attributed_practitioner_id" in cols


def test_create_membership_stamps_practitioner():
    cx = _cx()
    sid = subs.create_membership(
        cx, email="p@x.com", stripe_customer_id="cus_1",
        stripe_payment_method_id="pm_1", amount_cents=9900,
        next_charge_date="2026-08-01", attributed_practitioner_id="prac-42")
    row = cx.execute("SELECT attributed_practitioner_id FROM subscriptions WHERE id=?",
                     (sid,)).fetchone()
    assert row["attributed_practitioner_id"] == "prac-42"


def test_create_membership_defaults_null():
    cx = _cx()
    sid = subs.create_membership(
        cx, email="p@x.com", stripe_customer_id="cus_1",
        stripe_payment_method_id="pm_1", amount_cents=9900,
        next_charge_date="2026-08-01")
    row = cx.execute("SELECT attributed_practitioner_id FROM subscriptions WHERE id=?",
                     (sid,)).fetchone()
    assert row["attributed_practitioner_id"] is None
