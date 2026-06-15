"""Phase 2 per-line partial fulfillment + backorder tracking:
record_fulfillment clamping, partial→full clearing, the orders.fulfill_lines
action transitions, and backorder_rollup aggregation."""
import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    from dashboard import orders as O
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    O.init_fulfillments_table(cx)
    return O, cx


def _order(O, cx, ref, lines, status="new"):
    return O.upsert_order(cx, source="in-house", external_ref=ref, status=status,
                          items=lines, total_cents=0)


def test_record_fulfillment_clamps_to_ordered():
    O, cx = _db()
    oid = _order(O, cx, "INH-1", [{"slug": "bone-builder", "name": "Bone Builder", "qty": 3}])
    assert O.record_fulfillment(cx, oid, 0, "bone-builder", 2) == 2
    # Over-fulfill attempt: only 1 unit of room remains, so it clamps to 1.
    assert O.record_fulfillment(cx, oid, 0, "bone-builder", 5) == 1
    # Fully fulfilled now → further events record nothing.
    assert O.record_fulfillment(cx, oid, 0, "bone-builder", 1) == 0
    assert O.fulfilled_qty(cx, oid, 0) == 3
    assert O.order_backorder_units(cx, oid) == 0


def test_partial_then_full_clears_backorder():
    O, cx = _db()
    oid = _order(O, cx, "INH-2", [{"slug": "x", "name": "X", "qty": 4}])
    O.record_fulfillment(cx, oid, 0, "x", 1)
    summary = O.fulfillment_for_order(cx, oid)[0]
    assert summary["fulfilled"] == 1 and summary["backordered"] == 3
    assert len(summary["events"]) == 1
    O.record_fulfillment(cx, oid, 0, "x", 3)
    assert O.order_backorder_units(cx, oid) == 0
    assert len(O.fulfillment_for_order(cx, oid)[0]["events"]) == 2  # history retained


def test_fulfill_lines_action_partial_then_done():
    O, cx = _db()
    oid = _order(O, cx, "INH-3", [
        {"slug": "a", "name": "A", "qty": 2},
        {"slug": "b", "name": "B", "qty": 1}], status="packed")
    ctx = {"cx": cx}
    res = O._fulfill_lines_exec({"order_id": oid, "lines": [{"index": 0, "qty": 1}]}, ctx)
    assert res["status"] == "shipped"            # partial → stays shipped
    assert res["backorder_units"] == 2           # 1 of A + 1 of B remain
    assert O.get_order(cx, oid)["status"] == "shipped"

    res2 = O._fulfill_lines_exec(
        {"order_id": oid, "lines": [{"index": 0, "qty": 1}, {"index": 1, "qty": 1}]}, ctx)
    assert res2["backorder_units"] == 0
    assert res2["status"] == "done"              # all lines cleared → done
    assert O.get_order(cx, oid)["status"] == "done"


def test_fulfill_lines_requires_a_quantity():
    O, cx = _db()
    oid = _order(O, cx, "INH-4", [{"slug": "a", "name": "A", "qty": 2}], status="packed")
    try:
        O._fulfill_lines_exec({"order_id": oid, "lines": [{"index": 0, "qty": 0}]}, {"cx": cx})
        assert False, "expected ValueError on no quantities"
    except ValueError:
        pass


def test_backorder_rollup_aggregates_and_excludes_done_cancelled():
    O, cx = _db()
    # Two open orders both backordering 'bone-builder', one a different product.
    o1 = _order(O, cx, "INH-5", [{"slug": "bb", "name": "Bone Builder", "qty": 3}], status="new")
    o2 = _order(O, cx, "INH-6", [{"slug": "bb", "name": "Bone Builder", "qty": 2},
                                 {"slug": "af", "name": "AllerFree", "qty": 1}], status="shipped")
    O.record_fulfillment(cx, o2, 0, "bb", 1)  # 1 of 2 shipped → 1 still owed
    # A done order and a cancelled order must NOT contribute.
    od = _order(O, cx, "INH-7", [{"slug": "bb", "name": "Bone Builder", "qty": 9}], status="done")
    oc = _order(O, cx, "INH-8", [{"slug": "bb", "name": "Bone Builder", "qty": 9}], status="cancelled")
    # A proposed (unpaid) invoice is not committed demand → excluded too.
    op = _order(O, cx, "INH-9", [{"slug": "bb", "name": "Bone Builder", "qty": 9}], status="proposed")

    roll = {r["slug"]: r for r in O.backorder_rollup(cx)}
    assert roll["bb"]["units_backordered"] == 4   # 3 (o1) + 1 (o2), not the done/cancelled/proposed
    assert roll["bb"]["order_count"] == 2
    assert roll["af"]["units_backordered"] == 1
    # Sorted: bone-builder (4) ahead of allerfree (1).
    assert O.backorder_rollup(cx)[0]["slug"] == "bb"
