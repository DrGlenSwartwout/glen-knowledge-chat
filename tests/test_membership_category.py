# tests/test_membership_category.py
"""Trial-membership category classifier (PR1).

category_for(cx, email) -> 'none' | 'trial' | 'full' | 'paused'
Source of truth = the member's active kind=membership subscription:
  paused  -> active membership sub with skip_next=1
  full    -> active membership sub, skip_next=0, order_count >= 1 (>=1 real $99 charge cleared)
  trial   -> active membership sub, skip_next=0, order_count == 0 (free first month)
  none    -> no active membership sub
"""
import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    return cx


def _make_membership(cx, email="m@x.com"):
    return subs.create_membership(
        cx, email=email, stripe_customer_id="cus", stripe_payment_method_id="pm",
        amount_cents=9900, next_charge_date="2026-07-01")


def test_none_when_no_membership():
    cx = _cx()
    assert subs.category_for(cx, "nobody@x.com") == "none"


def test_trial_is_membership_with_zero_order_count():
    cx = _cx()
    _make_membership(cx)
    assert subs.category_for(cx, "m@x.com") == "trial"


def test_full_after_first_charge_bumps_order_count():
    cx = _cx()
    sid = _make_membership(cx)
    subs.advance_after_charge(cx, sid)            # first $99 cleared -> order_count 0->1
    assert subs.category_for(cx, "m@x.com") == "full"


def test_full_stays_full_at_higher_order_counts():
    cx = _cx()
    sid = _make_membership(cx)
    for _ in range(5):
        subs.advance_after_charge(cx, sid)
    assert subs.category_for(cx, "m@x.com") == "full"


def test_paused_when_skip_next_set_on_trial():
    cx = _cx()
    sid = _make_membership(cx)
    subs.set_skip_next(cx, sid, True)
    assert subs.category_for(cx, "m@x.com") == "paused"


def test_paused_takes_precedence_over_full():
    cx = _cx()
    sid = _make_membership(cx)
    subs.advance_after_charge(cx, sid)            # order_count 1 (would be 'full')
    subs.set_skip_next(cx, sid, True)             # ...but paused
    assert subs.category_for(cx, "m@x.com") == "paused"


def test_cancelled_membership_is_none():
    cx = _cx()
    sid = _make_membership(cx)
    subs.set_status(cx, sid, "cancelled")
    assert subs.category_for(cx, "m@x.com") == "none"


def test_product_subscription_is_not_a_membership_category():
    cx = _cx()
    subs.create(cx, email="p@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                items=[{"slug": "x", "qty": 1}], cadence_months=1, ship_address={},
                next_charge_date="2026-07-01")
    assert subs.category_for(cx, "p@x.com") == "none"


def test_email_is_normalised_or_passed_through():
    # classifier should at least match the stored (lowercased) email
    cx = _cx()
    _make_membership(cx, email="trial@x.com")
    assert subs.category_for(cx, "trial@x.com") == "trial"
