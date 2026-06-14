# tests/test_points_ledger.py
import sqlite3
import pytest
from dashboard import points


def _cx():
    cx = sqlite3.connect(":memory:")
    points.init_points_table(cx)
    return cx


# ── Plan tests (verbatim) ──────────────────────────────────────────────────

def test_earn_then_balance():
    cx = _cx()
    # earn 5% of a $70 full-price order = 350 cents of value
    bal = points.earn(cx, "a@x.com", full_price_cents=7000, earn_pct=0.05, order_ref="o1")
    assert bal == 350
    assert points.balance(cx, "a@x.com") == 350


def test_redeem_decrements_balance():
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=7000, earn_pct=0.05, order_ref="o1")
    bal = points.redeem(cx, "a@x.com", value_cents=200, order_ref="o2")
    assert bal == 150
    assert points.balance(cx, "a@x.com") == 150


def test_cannot_redeem_more_than_balance():
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=2000, earn_pct=0.05, order_ref="o1")  # 100
    with pytest.raises(ValueError):
        points.redeem(cx, "a@x.com", value_cents=500, order_ref="o2")


# ── Additional tests ───────────────────────────────────────────────────────

def test_redeem_exact_balance_ok():
    """Redeeming exactly the full balance succeeds and leaves balance 0."""
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=4000, earn_pct=0.05, order_ref="o1")  # 200
    bal = points.redeem(cx, "a@x.com", value_cents=200, order_ref="o2")
    assert bal == 0
    assert points.balance(cx, "a@x.com") == 0


def test_balance_after_column_tracks_running_total():
    """After earn then redeem, the last row's balance_after equals balance(cx, email)."""
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=7000, earn_pct=0.05, order_ref="o1")  # +350
    points.redeem(cx, "a@x.com", value_cents=100, order_ref="o2")                     # -100 -> 250
    last_row = cx.execute(
        "SELECT balance_after FROM points_ledger WHERE email=? ORDER BY id DESC LIMIT 1",
        ("a@x.com",)
    ).fetchone()
    assert last_row is not None
    assert last_row[0] == points.balance(cx, "a@x.com")


def test_two_accounts_isolated():
    """Earns for a@x.com don't affect balance for b@x.com."""
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=10000, earn_pct=0.05, order_ref="o1")  # 500
    assert points.balance(cx, "b@x.com") == 0
