import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import points as P
    from dashboard import ship_credit as SC
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    P.init_points_table(cx)
    return P, SC, cx


# ── grant ────────────────────────────────────────────────────────────────────

def test_grant_makes_spendable_balance_and_is_idempotent():
    _P, SC, cx = _db()
    SC.grant(cx, "Des@X.com", 300, source_ref="ORD-1")
    assert SC.balance(cx, "des@x.com") == 300          # email normalized both sides
    SC.grant(cx, "des@x.com", 300, source_ref="ORD-1")  # re-run recalc -> no double
    assert SC.balance(cx, "des@x.com") == 300


def test_grant_ignores_zero_blank_email_or_no_ref():
    _P, SC, cx = _db()
    SC.grant(cx, "des@x.com", 0, source_ref="ORD-1")
    SC.grant(cx, "", 300, source_ref="ORD-2")
    SC.grant(cx, "des@x.com", 300, source_ref="")
    assert SC.balance(cx, "des@x.com") == 0


def test_ship_credit_isolated_from_loyalty_points():
    P, SC, cx = _db()
    P.credit(cx, "des@x.com", value_cents=1000, reason="earn", order_ref="R", scope="rm")
    SC.grant(cx, "des@x.com", 300, source_ref="ORD-1")
    assert SC.balance(cx, "des@x.com") == 300           # ship_credit scope only
    assert P.balance(cx, "des@x.com", scope="rm") == 1000  # loyalty untouched


# ── plan_application (pure) ───────────────────────────────────────────────────

def test_plan_application_clamps_to_balance_and_chargeable_and_floors_at_zero():
    from dashboard import ship_credit as SC
    assert SC.plan_application(300, 5000) == 300   # bounded by balance
    assert SC.plan_application(5000, 300) == 300   # bounded by order total (no negative)
    assert SC.plan_application(0, 5000) == 0
    assert SC.plan_application(-10, 5000) == 0     # never negative
    assert SC.plan_application(300, 0) == 0


# ── consume ───────────────────────────────────────────────────────────────────

def test_consume_debits_and_is_idempotent_on_applying_order():
    _P, SC, cx = _db()
    SC.grant(cx, "des@x.com", 300, source_ref="ORD-1")
    assert SC.consume(cx, "des@x.com", 300, applied_ref="NEW-9") == 300
    assert SC.balance(cx, "des@x.com") == 0
    # resubmitting the SAME new order must not double-spend (guard on applied_ref)
    assert SC.consume(cx, "des@x.com", 300, applied_ref="NEW-9") == 0
    assert SC.balance(cx, "des@x.com") == 0


def test_consume_clamps_to_available_balance():
    _P, SC, cx = _db()
    SC.grant(cx, "des@x.com", 300, source_ref="ORD-1")
    # asking for more than the balance spends only what's there
    assert SC.consume(cx, "des@x.com", 900, applied_ref="NEW-9") == 300
    assert SC.balance(cx, "des@x.com") == 0


def test_partial_consume_leaves_remainder_for_next_order():
    _P, SC, cx = _db()
    SC.grant(cx, "des@x.com", 300, source_ref="ORD-1")
    assert SC.consume(cx, "des@x.com", 200, applied_ref="NEW-1") == 200
    assert SC.balance(cx, "des@x.com") == 100
    assert SC.consume(cx, "des@x.com", 100, applied_ref="NEW-2") == 100
    assert SC.balance(cx, "des@x.com") == 0


# ── refund ────────────────────────────────────────────────────────────────────

def test_mark_refunded_removes_balance_and_guards_double_refund():
    _P, SC, cx = _db()
    SC.grant(cx, "des@x.com", 300, source_ref="ORD-1")
    assert SC.already_refunded(cx, source_ref="ORD-1") is False
    assert SC.mark_refunded(cx, "des@x.com", 300, source_ref="ORD-1") == 300
    assert SC.balance(cx, "des@x.com") == 0
    assert SC.already_refunded(cx, source_ref="ORD-1") is True
    # a second refund of the same source order is a no-op
    assert SC.mark_refunded(cx, "des@x.com", 300, source_ref="ORD-1") == 0


def test_refunded_credit_cannot_also_auto_apply():
    _P, SC, cx = _db()
    SC.grant(cx, "des@x.com", 300, source_ref="ORD-1")
    SC.mark_refunded(cx, "des@x.com", 300, source_ref="ORD-1")
    # balance is gone, so a later checkout applies nothing
    assert SC.plan_application(SC.balance(cx, "des@x.com"), 5000) == 0
    assert SC.consume(cx, "des@x.com", 300, applied_ref="NEW-9") == 0
