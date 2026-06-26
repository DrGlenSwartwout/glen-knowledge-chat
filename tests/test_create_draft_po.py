"""C1: create_draft_po inserts a draft PO + items; reorder.create_po action wraps it."""
import sqlite3
import pytest


@pytest.fixture
def cx():
    cx = sqlite3.connect(":memory:")
    from dashboard import purchase_orders as po
    po.init_purchase_orders_schema(cx)
    return cx


def test_create_draft_po_inserts_po_and_items(cx):
    from dashboard import purchase_orders as po
    lines = [
        {"ingredient_id": 5, "ingredient": "Ashwagandha", "suggested_qty": 100.0, "unit": "g",
         "price_per_unit": 12.5, "unit_size": 50.0, "packs": 2, "est_cost": 25.0},
        {"ingredient_id": 6, "ingredient": "NoPrice", "suggested_qty": 30.0, "unit": "g",
         "price_per_unit": None, "unit_size": None, "packs": None, "est_cost": None},
    ]
    res = po.create_draft_po(cx, 9, "Acme Botanicals", lines)
    assert res["line_count"] == 2 and res["po_id"]
    hdr = cx.execute("SELECT supplier_id, supplier_name, status, vendor_po_no FROM purchase_orders WHERE id=?",
                     (res["po_id"],)).fetchone()
    assert hdr[0] == 9 and hdr[1] == "Acme Botanicals" and hdr[2] == "draft" and hdr[3].startswith("DRAFT-")
    items = cx.execute("SELECT ingredient_id, qty, qty_unit, cost, item_kind FROM po_items WHERE po_id=? ORDER BY ingredient_id",
                       (res["po_id"],)).fetchall()
    assert items[0] == (5, 100.0, "g", 12.5, "ingredient")
    assert items[1][0] == 6 and items[1][3] is None          # no-price line: cost NULL, still inserted


def test_create_draft_po_skips_lines_missing_id_or_qty(cx):
    from dashboard import purchase_orders as po
    res = po.create_draft_po(cx, 1, "X", [{"ingredient": "bad", "suggested_qty": 5}, {"ingredient_id": 7, "suggested_qty": None}])
    assert res["line_count"] == 0


def test_exec_create_po_ok(cx):
    from dashboard import reorder_actions as ra
    res = ra._exec_create_po({"supplier_id": 9, "supplier_name": "Acme",
                              "lines": [{"ingredient_id": 5, "suggested_qty": 100.0, "unit": "g", "price_per_unit": 12.5}]},
                             {"cx": cx, "actor": None})
    assert res["ok"] is True and res["po_id"]


def test_exec_create_po_no_supplier(cx):
    from dashboard import reorder_actions as ra
    res = ra._exec_create_po({"supplier_id": None, "lines": []}, {"cx": cx})
    assert res["ok"] is False


def test_action_registered_metadata():
    from dashboard import reorder_actions as ra
    from dashboard.actions import get_action, LOW_WRITE
    from dashboard.rbac import OWNER, OPS
    ra.register()
    a = get_action("reorder.create_po")
    assert a is not None and a.risk_tier == LOW_WRITE and a.permission == (OWNER, OPS)
