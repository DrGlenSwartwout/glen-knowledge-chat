import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import orders as O
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return O, cx


def test_upsert_is_idempotent_and_preserves_status():
    O, cx = _db()
    oid = O.upsert_order(cx, source="funnel", external_ref="INV-1", email="a@b.com",
                         name="Ann", items=[{"name": "Mag", "qty": 2}], total_cents=7000,
                         channel="retail")
    assert oid > 0
    # advance status, then re-ingest the same order: status must NOT regress
    assert O.set_order_status(cx, oid, "packed") is True
    again = O.upsert_order(cx, source="funnel", external_ref="INV-1", email="a@b.com",
                           name="Ann Updated", total_cents=7000)
    assert again == oid
    row = O.get_order(cx, oid)
    assert row["status"] == "packed"
    assert row["name"] == "Ann Updated"
    assert row["items"] == [{"name": "Mag", "qty": 2}]


def test_upsert_requires_external_ref():
    O, cx = _db()
    try:
        O.upsert_order(cx, source="manual", external_ref="")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_list_orders_filter_and_set_tracking():
    O, cx = _db()
    a = O.upsert_order(cx, source="gk", external_ref="1")
    b = O.upsert_order(cx, source="gk", external_ref="2")
    O.set_order_status(cx, b, "shipped")
    assert len(O.list_orders(cx)) == 2
    assert len(O.list_orders(cx, status="new")) == 1
    O.set_order_tracking(cx, a, "9400111899", shipment_id=5)
    assert O.get_order(cx, a)["tracking_number"] == "9400111899"
    assert O.get_order(cx, a)["shipment_id"] == 5


def test_orders_signal_levels():
    from dashboard import orders as O, signals as S
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    now = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)
    # no orders -> green
    assert O.orders_signal(cx, None, now=now)["level"] == S.GREEN
    # a fresh open order -> amber
    O.upsert_order(cx, source="funnel", external_ref="A")
    sig = O.orders_signal(cx, None, now=now)
    assert sig["level"] == S.AMBER and sig["count"] == 1
    # an order created >24h ago -> red
    cx.execute("UPDATE orders SET created_at=? WHERE external_ref='A'",
               ((now - timedelta(hours=48)).isoformat(),))
    cx.commit()
    assert O.orders_signal(cx, None, now=now)["level"] == S.RED


def test_orders_signal_gray_when_table_missing():
    from dashboard import orders as O, signals as S
    cx = sqlite3.connect(":memory:")  # no orders table
    assert O.orders_signal(cx, None)["level"] == S.GRAY


def test_lifecycle_action_marks_packed_and_logs_event():
    from dashboard import orders as O, dispatch as D, events as E, rbac as R, actions as A
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="Z")
    assert A.get_action("orders.mark_packed") is not None
    res = D.dispatch_action(cx, "orders.mark_packed", {"order_id": oid}, R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    assert O.get_order(cx, oid)["status"] == "packed"
    ev = E.list_events(cx, module="orders")
    assert ev and ev[0]["action_key"] == "orders.mark_packed"


def test_set_tracking_action_records_and_ships():
    from dashboard import orders as O, dispatch as D, events as E, rbac as R, actions as A
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    E.init_event_tables(cx)
    O.init_orders_table(cx)
    oid = O.upsert_order(cx, source="funnel", external_ref="T1")
    assert A.get_action("orders.set_tracking") is not None
    res = D.dispatch_action(cx, "orders.set_tracking",
                            {"order_id": oid, "tracking_number": "9400111899223"},
                            R.Actor(role=R.OWNER))
    assert res["status"] == "done"
    row = O.get_order(cx, oid)
    assert row["status"] == "shipped"
    assert row["tracking_number"] == "9400111899223"
