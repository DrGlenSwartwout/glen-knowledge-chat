import sqlite3
import pytest
from dashboard import household_holds as H
from dashboard import orders as O
from dashboard import family_plan as FP
from dashboard import household as HH


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    FP.init_family_plan_table(cx)
    HH.init_household_tables(cx)
    H.init_hold_tables(cx)
    return cx


def _order(cx, email, *, channel="ship", status="proposed"):
    return O.upsert_order(cx, source="test", external_ref=email, email=email,
                          name=email.split("@")[0],
                          items=[{"slug": "x", "qty": 1}], total_cents=1000,
                          channel=channel, status=status)


def test_eligible_only_for_covered_shippable(tmp_path):
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
    covered = _order(cx, "kid@x.com")
    uncovered = _order(cx, "stranger@x.com")
    pickup = _order(cx, "cg@x.com", channel="pickup")
    assert H.eligible_for_hold(cx, O.get_order(cx, covered)) is True
    assert H.eligible_for_hold(cx, O.get_order(cx, uncovered)) is False
    assert H.eligible_for_hold(cx, O.get_order(cx, pickup)) is False
