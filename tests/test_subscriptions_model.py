# tests/test_subscriptions_model.py
import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx); return cx


def test_add_months_edge_cases():
    assert subs.add_months("2026-01-31", 1) == "2026-02-28"   # day clamp, non-leap
    assert subs.add_months("2024-01-31", 1) == "2024-02-29"   # leap-year clamp
    assert subs.add_months("2024-02-29", 12) == "2025-02-28"  # leap day into non-leap
    assert subs.add_months("2026-12-15", 1) == "2027-01-15"   # Dec -> Jan year rollover
    assert subs.add_months("2026-07-01", 2) == "2026-09-01"   # plain


def test_set_cadence_and_next_charge_date():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                      items=[], cadence_months=1, ship_address={}, next_charge_date="2026-07-01")
    subs.set_cadence(cx, sid, 3)
    subs.set_next_charge_date(cx, sid, "2026-10-01")
    s = subs.get(cx, sid)
    assert s["cadence_months"] == 3 and s["next_charge_date"] == "2026-10-01"


def test_tier_for_escalates_and_caps():
    assert subs.tier_for(0) == 5
    assert subs.tier_for(1) == 10
    assert subs.tier_for(2) == 15
    assert subs.tier_for(9) == 15


def test_create_and_get():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="cus_1",
                      stripe_payment_method_id="pm_1",
                      items=[{"slug":"x","qty":1}], cadence_months=1,
                      ship_address={"state":"CA"}, next_charge_date="2026-07-01")
    s = subs.get(cx, sid)
    assert s["status"] == "active" and s["order_count"] == 0
    assert subs.get_active_by_email(cx, "a@x.com")[0]["id"] == sid


def test_list_due_respects_status_skip_and_date():
    cx = _cx()
    a = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                    items=[], cadence_months=1, ship_address={}, next_charge_date="2026-07-01")
    subs.create(cx, email="b@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                items=[], cadence_months=1, ship_address={}, next_charge_date="2026-09-01")
    due = subs.list_due(cx, as_of="2026-07-15")
    assert [d["id"] for d in due] == [a]            # only the past-due active one
    subs.set_skip_next(cx, a, True)
    assert subs.list_due(cx, as_of="2026-07-15") == []   # skip hides it


def test_advance_after_charge_increments_and_moves_date():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                      items=[], cadence_months=2, ship_address={}, next_charge_date="2026-07-01")
    subs.advance_after_charge(cx, sid)
    s = subs.get(cx, sid)
    assert s["order_count"] == 1
    assert s["next_charge_date"] == "2026-09-01"     # +2 months


def test_skip_consumes_without_incrementing_tier():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                      items=[], cadence_months=1, ship_address={}, next_charge_date="2026-07-01")
    subs.set_skip_next(cx, sid, True)
    subs.consume_skip(cx, sid)                        # advances date, clears flag, no order_count++
    s = subs.get(cx, sid)
    assert s["order_count"] == 0 and s["skip_next"] == 0
    assert s["next_charge_date"] == "2026-08-01"


def test_cancel_resets_tier():
    cx = _cx()
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                      items=[], cadence_months=1, ship_address={}, next_charge_date="2026-07-01")
    subs.advance_after_charge(cx, sid)
    subs.set_status(cx, sid, "cancelled")
    s = subs.get(cx, sid)
    assert s["status"] == "cancelled" and s["order_count"] == 0   # reset on cancel
