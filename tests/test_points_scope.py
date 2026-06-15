import sqlite3
from dashboard import points


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    points.init_points_table(cx)
    return cx


def test_scopes_are_isolated():
    cx = _cx()
    points.credit(cx, "p@x.com", value_cents=500, reason="earn:dispensary",
                  order_ref="O1", scope="dispensary:42")
    points.credit(cx, "p@x.com", value_cents=200, reason="earn", order_ref="O2")  # default rm
    assert points.balance(cx, "p@x.com", scope="dispensary:42") == 500
    assert points.balance(cx, "p@x.com") == 200            # rm scope unaffected
    assert points.balance(cx, "p@x.com", scope="dispensary:99") == 0


def test_redeem_scoped():
    cx = _cx()
    points.credit(cx, "p@x.com", value_cents=500, reason="earn:dispensary",
                  order_ref="O1", scope="dispensary:42")
    points.redeem(cx, "p@x.com", value_cents=300, order_ref="O3", scope="dispensary:42")
    assert points.balance(cx, "p@x.com", scope="dispensary:42") == 200
    points.credit(cx, "p@x.com", value_cents=1000, reason="earn", order_ref="O4")  # rm
    import pytest
    with pytest.raises(ValueError):
        points.redeem(cx, "p@x.com", value_cents=999, order_ref="O5", scope="dispensary:42")


def test_has_entry_is_scoped():
    cx = _cx()
    points.credit(cx, "p@x.com", value_cents=500, reason="earn:dispensary",
                  order_ref="O1", scope="dispensary:42")
    assert points.has_entry(cx, order_ref="O1", reason="earn:dispensary", scope="dispensary:42")
    assert not points.has_entry(cx, order_ref="O1", reason="earn:dispensary", scope="rm")


def test_default_scope_is_rm_backward_compatible():
    cx = _cx()
    points.earn(cx, "p@x.com", full_price_cents=10000, earn_pct=0.05, order_ref="O1")
    assert points.balance(cx, "p@x.com") == 500            # default rm, old signature works
