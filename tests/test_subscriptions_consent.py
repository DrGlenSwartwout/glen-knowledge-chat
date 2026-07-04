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
    subs.migrate_add_consent_column(cx)  # second call must not raise
    cols = {r[1] for r in cx.execute("PRAGMA table_info(subscriptions)")}
    assert "practitioner_share_consent" in cols


def test_consent_defaults_zero():
    cx = _cx()
    sid = subs.create_membership(cx, email="p@x.com", stripe_customer_id="c",
        stripe_payment_method_id="p", amount_cents=9900, next_charge_date="2026-08-01")
    assert cx.execute("SELECT practitioner_share_consent FROM subscriptions WHERE id=?", (sid,)).fetchone()[0] == 0


def test_consent_persisted_when_set():
    cx = _cx()
    sid = subs.create_membership(cx, email="p@x.com", stripe_customer_id="c",
        stripe_payment_method_id="p", amount_cents=9900, next_charge_date="2026-08-01",
        practitioner_share_consent=1)
    assert cx.execute("SELECT practitioner_share_consent FROM subscriptions WHERE id=?", (sid,)).fetchone()[0] == 1
