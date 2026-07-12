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


def test_open_then_sibling_joins_same_group_deadline_from_first():
    from datetime import datetime, timezone
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
    o1 = _order(cx, "cg@x.com")
    o2 = _order(cx, "kid@x.com")
    t0 = datetime(2026, 7, 12, 9, 0, tzinfo=timezone.utc)
    r1 = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com",
                             household_key="cg@x.com", hold_days=4, now=t0)
    assert r1["opened"] is True and r1["joined"] is False
    t1 = datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc)  # 2 days later
    r2 = H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com",
                             household_key="cg@x.com", hold_days=4, now=t1)
    assert r2["opened"] is False and r2["joined"] is True
    assert r2["group_id"] == r1["group_id"]
    hold = H.get_hold(cx, r1["group_id"])
    assert hold["hold_until"].startswith("2026-07-16")  # t0 + 4d, NOT t1 + 4d
    assert {m["id"] for m in H.orders_in_hold(cx, r1["group_id"])} == {o1, o2}


def test_release_returns_order_ids_and_closes_group():
    from datetime import datetime, timezone
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
    o1 = _order(cx, "cg@x.com"); o2 = _order(cx, "kid@x.com")
    t0 = datetime(2026, 7, 12, 9, 0, tzinfo=timezone.utc)
    g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com", now=t0)["group_id"]
    H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com", household_key="cg@x.com", now=t0)
    res = H.release_hold(cx, g, by="caregiver")
    assert sorted(res["order_ids"]) == sorted([o1, o2])
    assert H.get_hold(cx, g)["status"] == "released"


def test_due_holds_only_past_deadline_open():
    from datetime import datetime, timezone
    cx = _cx()
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    o1 = _order(cx, "cg@x.com")
    t0 = datetime(2026, 7, 12, 9, 0, tzinfo=timezone.utc)
    g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com", hold_days=4, now=t0)["group_id"]
    before = datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)
    after = datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)
    assert H.due_holds(cx, now=before) == []
    assert [d["id"] for d in H.due_holds(cx, now=after)] == [g]
