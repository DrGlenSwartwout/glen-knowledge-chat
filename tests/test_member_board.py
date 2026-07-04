# tests/test_member_board.py
"""Members board helpers (PR3, pure): classify_sub, member_board_row,
list_active_memberships. These back the /console/members Trial/Full/Paused board.
"""
import sqlite3
from dashboard import subscriptions as subs


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    return cx


def _mk(cx, email="m@x.com"):
    return subs.create_membership(
        cx, email=email, stripe_customer_id="cus", stripe_payment_method_id="pm",
        amount_cents=9900, next_charge_date="2026-07-01")


# --- classify_sub -----------------------------------------------------------

def test_classify_trial_full_paused():
    assert subs.classify_sub({"order_count": 0, "skip_next": 0}) == "trial"
    assert subs.classify_sub({"order_count": 1, "skip_next": 0}) == "full"
    assert subs.classify_sub({"order_count": 3, "skip_next": 0}) == "full"
    assert subs.classify_sub({"order_count": 0, "skip_next": 1}) == "paused"
    # paused wins even when order_count would be 'full'
    assert subs.classify_sub({"order_count": 5, "skip_next": 1}) == "paused"


def test_category_for_delegates_to_classify_sub():
    cx = _cx()
    sid = _mk(cx)
    assert subs.category_for(cx, "m@x.com") == "trial"
    subs.advance_after_charge(cx, sid)
    assert subs.category_for(cx, "m@x.com") == "full"


# --- list_active_memberships ------------------------------------------------

def test_list_active_memberships_excludes_cancelled_and_products():
    cx = _cx()
    _mk(cx, "a@x.com")                 # trial membership
    sid = _mk(cx, "b@x.com")
    subs.set_status(cx, sid, "cancelled")   # cancelled -> excluded
    subs.create(cx, email="p@x.com", stripe_customer_id="c", stripe_payment_method_id="p",
                items=[{"slug": "x", "qty": 1}], cadence_months=1, ship_address={},
                next_charge_date="2026-07-01")  # product sub -> excluded
    rows = subs.list_active_memberships(cx)
    emails = {r["email"] for r in rows}
    assert emails == {"a@x.com"}


def test_list_active_memberships_dedupes_two_subs_per_email_to_oldest():
    # A buyer holding two active membership subs (e.g. a $1-trial sub + a later
    # separate join) must appear ONCE, classified by the oldest sub (lowest id) so
    # the board agrees with category_for / the pricing gate.
    cx = _cx()
    first = _mk(cx, "dup@x.com")          # oldest sub -> still trial (order_count 0)
    second = _mk(cx, "dup@x.com")         # newer sub
    subs.advance_after_charge(cx, second)  # newer is 'full', but oldest wins
    rows = subs.list_active_memberships(cx)
    dup_rows = [r for r in rows if r["email"] == "dup@x.com"]
    assert len(dup_rows) == 1
    assert dup_rows[0]["id"] == first
    assert subs.classify_sub(dup_rows[0]) == "trial"  # matches category_for(rows[0])
    assert subs.category_for(cx, "dup@x.com") == subs.classify_sub(dup_rows[0])


# --- member_board_row -------------------------------------------------------

def test_board_row_trial_carries_credit_cents():
    cx = _cx(); _mk(cx, "t@x.com")
    sub = subs.list_active_memberships(cx)[0]
    row = subs.member_board_row(sub, name="Tina Trial", credit_cents=3000)
    assert row["category"] == "trial"
    assert row["email"] == "t@x.com"
    assert row["name"] == "Tina Trial"
    assert row["credit_cents"] == 3000
    assert row["plan_cents"] == 9900
    assert row["tier"] == subs.tier_for(0)
    assert "resume_date" not in row


def test_board_row_full_has_no_credit_or_resume():
    cx = _cx(); sid = _mk(cx, "f@x.com")
    subs.advance_after_charge(cx, sid)
    sub = subs.list_active_memberships(cx)[0]
    row = subs.member_board_row(sub, name="Frank Full")
    assert row["category"] == "full"
    assert "credit_cents" not in row
    assert "resume_date" not in row
    assert row["tier"] == subs.tier_for(1)


def test_board_row_paused_has_resume_date():
    cx = _cx(); sid = _mk(cx, "p@x.com")
    subs.set_skip_next(cx, sid, True)
    sub = subs.list_active_memberships(cx)[0]
    row = subs.member_board_row(sub)
    assert row["category"] == "paused"
    # resume = next_charge_date advanced by cadence_months (1) -> 2026-08-01
    assert row["resume_date"] == subs.add_months("2026-07-01", 1)
    assert "credit_cents" not in row
