import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_founding_columns(cx)
    return cx


def test_create_founding_reservation_is_pending_far_dated():
    cx = _cx()
    sid = subs.create_founding_reservation(
        cx, email="f@x.com", stripe_customer_id="cus", stripe_payment_method_id="pm",
        items=[{"slug": "neuro-magnesium", "qty": 1}], ship_address={"state": "HI"},
        founding_slug="neuro-magnesium")
    row = subs.get(cx, sid)
    assert row["founding"] == 1
    assert row["founding_state"] == "pending"
    assert row["founding_slug"] == "neuro-magnesium"
    assert row["order_count"] == 0
    assert row["next_charge_date"] == "2999-01-01"   # never picked by list_due until shipped


def test_pending_reservation_not_in_list_due():
    cx = _cx()
    subs.create_founding_reservation(
        cx, email="f@x.com", stripe_customer_id="c", stripe_payment_method_id="pm",
        items=[{"slug": "neuro-magnesium", "qty": 1}], ship_address={}, founding_slug="neuro-magnesium")
    assert subs.list_due(cx, as_of="2030-01-01") == []   # far-dated sentinel excludes it


def test_mark_founding_active_sets_first_charge_cycle():
    cx = _cx()
    sid = subs.create_founding_reservation(
        cx, email="f@x.com", stripe_customer_id="c", stripe_payment_method_id="pm",
        items=[{"slug": "neuro-magnesium", "qty": 1}], ship_address={}, founding_slug="neuro-magnesium")
    subs.mark_founding_active(cx, sid, next_charge_date="2026-08-01")
    row = subs.get(cx, sid)
    assert row["founding_state"] == "active"
    assert row["next_charge_date"] == "2026-08-01"
    assert row["order_count"] == 1


def test_count_and_list_founding_pending():
    cx = _cx()
    a = subs.create_founding_reservation(cx, email="a@x.com", stripe_customer_id="c",
        stripe_payment_method_id="pm", items=[], ship_address={}, founding_slug="neuro-magnesium")
    subs.create_founding_reservation(cx, email="b@x.com", stripe_customer_id="c",
        stripe_payment_method_id="pm", items=[], ship_address={}, founding_slug="neuro-magnesium")
    subs.mark_founding_active(cx, a, next_charge_date="2026-08-01")
    assert subs.count_founding(cx, "neuro-magnesium") == 2        # pending + active both count
    pending = subs.list_founding_pending(cx, "neuro-magnesium")
    assert len(pending) == 1 and pending[0]["email"] == "b@x.com"
