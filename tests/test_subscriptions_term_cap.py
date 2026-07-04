import sqlite3
import dashboard.subscriptions as subs

def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    return cx

def test_migration_idempotent():
    cx = _cx()
    subs.migrate_add_term_cap_column(cx)  # second call must not raise
    cols = {r[1] for r in cx.execute("PRAGMA table_info(subscriptions)")}
    assert "term_charges_total" in cols

def test_create_membership_writes_cap_and_initial_count():
    cx = _cx()
    sid = subs.create_membership(cx, email="a@x.com", stripe_customer_id="cus",
                                 stripe_payment_method_id="pm", amount_cents=9900,
                                 next_charge_date="2026-08-02", cadence_months=1,
                                 term_charges_total=6, initial_order_count=1)
    row = subs.get(cx, sid)
    assert row["term_charges_total"] == 6
    assert row["order_count"] == 1

def test_create_membership_defaults_uncapped():
    cx = _cx()
    sid = subs.create_membership(cx, email="b@x.com", stripe_customer_id="cus",
                                 stripe_payment_method_id="pm", amount_cents=9900,
                                 next_charge_date="2026-08-02")
    row = subs.get(cx, sid)
    assert row["term_charges_total"] is None
    assert row["order_count"] == 0
