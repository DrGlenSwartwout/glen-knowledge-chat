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


def test_create_membership_sets_kind_and_amount():
    cx = _cx()
    sid = subs.create_membership(cx, email="p@x.com", stripe_customer_id="cus_1",
                                 stripe_payment_method_id="pm_1", amount_cents=9900,
                                 next_charge_date="2026-09-15")
    row = subs.get(cx, sid)
    assert row["kind"] == "membership"
    assert row["amount_cents"] == 9900
    assert row["cadence_months"] == 1
    assert row["status"] == "active"
    assert row["next_charge_date"] == "2026-09-15"


def test_product_subs_default_kind_product():
    cx = _cx()
    sid = subs.create(cx, email="p@x.com", stripe_customer_id="c", stripe_payment_method_id="pm",
                      items=[{"slug": "a", "qty": 1}], cadence_months=1, ship_address={},
                      next_charge_date="2026-08-01")
    assert subs.get(cx, sid)["kind"] == "product"


def test_list_due_returns_membership():
    cx = _cx()
    subs.create_membership(cx, email="p@x.com", stripe_customer_id="c",
                           stripe_payment_method_id="pm", amount_cents=9900,
                           next_charge_date="2026-01-01")
    due = subs.list_due(cx, as_of="2026-02-01")
    assert len(due) == 1 and due[0]["kind"] == "membership" and due[0]["amount_cents"] == 9900


def test_active_membership_for_email():
    cx = _cx()
    subs.create_membership(cx, email="p@x.com", stripe_customer_id="c",
                           stripe_payment_method_id="pm", amount_cents=9900,
                           next_charge_date="2026-09-15")
    ms = subs.active_memberships_by_email(cx, "p@x.com")
    assert len(ms) == 1 and ms[0]["kind"] == "membership"


def test_pause_membership_sets_skip_and_returns_dates():
    cx = _cx()
    sid = subs.create_membership(cx, email="m@x.com", stripe_customer_id="c",
                                 stripe_payment_method_id="p", amount_cents=9900,
                                 next_charge_date="2026-07-15", cadence_months=1)
    r = subs.pause_membership_by_email(cx, "m@x.com")
    assert r["sub_id"] == sid
    assert r["paused_charge_date"] == "2026-07-15"
    assert r["resume_date"] == "2026-08-15"           # auto-resumes one cycle later
    assert subs.get(cx, sid)["skip_next"] == 1
    # loyalty/order_count untouched by a pause (unlike cancel which resets it)
    assert subs.get(cx, sid)["order_count"] == 0
    # idempotent
    assert subs.pause_membership_by_email(cx, "m@x.com")["sub_id"] == sid
    # no active membership -> None
    assert subs.pause_membership_by_email(cx, "nobody@x.com") is None


def test_set_membership_cadence_sets_months_and_clears_skip():
    cx = _cx()
    sid = subs.create_membership(cx, email="c@x.com", stripe_customer_id="c",
                                 stripe_payment_method_id="p", amount_cents=9900,
                                 next_charge_date="2026-07-10", cadence_months=1)
    subs.set_skip_next(cx, sid, True)
    r = subs.set_membership_cadence_by_email(cx, "c@x.com", 3)
    assert r["sub_id"] == sid and r["cadence_months"] == 3
    row = subs.get(cx, sid)
    assert row["cadence_months"] == 3 and row["skip_next"] == 0   # cadence supersedes a single skip
    assert subs.set_membership_cadence_by_email(cx, "nobody@x.com", 2) is None
