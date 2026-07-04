import sqlite3
import app as appmod
from dashboard import subscriptions as subs

EMAIL = "fm-subs@example.com"


def _seed(cx, *, next_charge_date="2026-07-01", order_count=1):
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    subs.migrate_add_free_months(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (EMAIL,)); cx.commit()
    sid = subs.create_membership(
        cx, email=EMAIL, stripe_customer_id="cus", stripe_payment_method_id="pm",
        amount_cents=9900, next_charge_date=next_charge_date, cadence_months=1,
        initial_order_count=order_count)
    cx.commit()
    return sid


def test_migrate_add_free_months_idempotent():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_free_months(cx)
    subs.migrate_add_free_months(cx)  # second call must not raise
    cols = {r[1] for r in cx.execute("PRAGMA table_info(subscriptions)")}
    assert "free_months_remaining" in cols
    cx.close()


def test_comp_cycle_advances_date_without_touching_order_count():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    sid = _seed(cx, next_charge_date="2026-07-01", order_count=1)
    before = subs.get(cx, sid)
    subs.comp_cycle(cx, sid)
    after = subs.get(cx, sid)
    assert after["next_charge_date"] == subs.add_months("2026-07-01", 1)
    assert after["order_count"] == before["order_count"] == 1
    assert not after.get("skip_next")
    cx.execute("DELETE FROM subscriptions WHERE email=?", (EMAIL,)); cx.commit(); cx.close()
