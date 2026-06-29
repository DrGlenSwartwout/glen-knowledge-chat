"""Invoice 'become a member and get $X back' figure (PR4).

_invoice_summary(order) adds member_credit_cents = the quantity discount the buyer
missed by not being a paid member (Σ over volume-eligible lines of
max(0, regular − member) × qty). Shown only to non-paid-members (none/trial);
paused/full already get volume pricing.
"""
import sqlite3
import pytest

# Bone Builder is qty-eligible ($69.97). At qty 3 the member tier is 5997 vs 6997
# regular -> (6997-5997)*3 = 3000.
EXPECTED = 3000


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    appmod._init_membership_tables()
    return appmod


def _order(email, lines):
    return {"email": email, "external_ref": "INV-1", "name": "X", "status": "proposed",
            "items": lines, "total_cents": 0}


def _eligible_lines(qty=3):
    return [{"slug": "bone-builder", "name": "Bone Builder", "qty": qty,
             "unit_cents": 6997, "line_cents": 6997 * qty}]


def test_non_member_sees_missed_discount(appmod):
    s = appmod._invoice_summary(_order("nobody@example.com", _eligible_lines()))
    assert s["member_credit_cents"] == EXPECTED


def test_full_member_sees_zero(appmod):
    email = "full@example.com"
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    appmod._grant_membership(cx, email, 31, "test")        # active grant
    from dashboard import subscriptions as subs
    subs.init_subscriptions_table(cx); subs.migrate_add_membership_columns(cx)
    sid = subs.create_membership(cx, email=email, stripe_customer_id="c",
                                 stripe_payment_method_id="p", amount_cents=9900,
                                 next_charge_date="2026-07-01")
    subs.advance_after_charge(cx, sid)                     # -> full
    cx.commit(); cx.close()
    assert appmod._is_paid_member(email) is True
    s = appmod._invoice_summary(_order(email, _eligible_lines()))
    assert s["member_credit_cents"] == 0


def test_non_eligible_lines_are_zero(appmod):
    lines = [{"slug": "some-booklet", "name": "Some Booklet", "qty": 5,
              "unit_cents": 1000, "line_cents": 5000}]
    s = appmod._invoice_summary(_order("nobody@example.com", lines))
    assert s["member_credit_cents"] == 0
