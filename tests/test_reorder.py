# tests/test_reorder.py
import json, sqlite3
import pytest
from dashboard import reorder as ro
from dashboard import inventory as inv


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, name TEXT, extras TEXT)")
        cx.execute("""CREATE TABLE ingredient_sources (id INTEGER PRIMARY KEY, ingredient_id INTEGER,
            supplier_id INTEGER, supplier_name TEXT, price_per_unit REAL, unit_size REAL,
            unit_type TEXT, preferred INTEGER DEFAULT 0, minimum_order REAL, minimum_order_unit TEXT)""")
        cx.execute("CREATE TABLE suppliers (id INTEGER PRIMARY KEY, company TEXT)")
        cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, name TEXT)")
        cx.execute("""CREATE TABLE formulation_items (id INTEGER PRIMARY KEY, formulation_id INTEGER,
            ingredient_id INTEGER, ingredient_name TEXT, dose REAL, dose_unit TEXT)""")
        cx.execute("CREATE TABLE purchase_orders (id INTEGER PRIMARY KEY, status TEXT)")
        cx.execute("CREATE TABLE po_items (id INTEGER PRIMARY KEY, po_id INTEGER, ingredient_id INTEGER, qty REAL)")
        cx.execute("CREATE TABLE po_receiving (id INTEGER PRIMARY KEY, po_item_id INTEGER, qty_received REAL)")
        inv.init_inventory_schema(cx)
        # ingredient 1: par 3 kg, preferred source MOQ 2 / unit_size 0.5 / $10
        cx.execute("INSERT INTO ingredients VALUES (1,'Mag',?)", (json.dumps({"par_level": "3", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO ingredients VALUES (2,'Lipoic',?)", (json.dumps({"par_level": "1", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO suppliers VALUES (7,'NOW Foods')")
        cx.execute("INSERT INTO ingredient_sources (id,ingredient_id,supplier_id,supplier_name,price_per_unit,unit_size,preferred,minimum_order,minimum_order_unit) VALUES (1,1,7,'NOW Foods',10.0,0.5,1,2.0,'kg')")
        cx.execute("INSERT INTO formulations VALUES (1,'Brain Blend')")
        cx.execute("INSERT INTO formulation_items (id,formulation_id,ingredient_id,ingredient_name,dose,dose_unit) VALUES (1,1,1,'Mag',0.5,'kg')")
        # ingredient 1 on-hand 1.0 (baseline); ingredient 2 on-hand 5.0 (well above par)
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',1.0)")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (2,'baseline',5.0)")
        cx.commit()
    return db


def test_round_up_order():
    assert ro._round_up_order(1.7, 2.0, 0.5) == 2.0      # MOQ floor
    assert ro._round_up_order(2.1, 2.0, 0.5) == 2.5      # round up to 0.5 multiple
    assert ro._round_up_order(2.1, None, None) == 2.1    # no MOQ/unit_size
    assert ro._round_up_order(0.3, None, 1.0) == 1.0     # ceil to unit_size


def test_on_order_excludes_closed_material_received(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO purchase_orders VALUES (10,'open')")
        cx.execute("INSERT INTO purchase_orders VALUES (11,'closed')")
        cx.execute("INSERT INTO po_items VALUES (100,10,1,4.0)")     # open, ingredient 1, qty 4
        cx.execute("INSERT INTO po_items VALUES (101,10,NULL,9.0)")  # open, material-only (null ingredient)
        cx.execute("INSERT INTO po_items VALUES (102,11,1,8.0)")     # CLOSED po → excluded
        cx.execute("INSERT INTO po_receiving VALUES (1000,100,1.5)") # 1.5 of item 100 already received
        cx.commit()
    oo = ro.on_order_by_ingredient(db)
    assert round(oo[1]["on_order"], 3) == 2.5            # 4.0 − 1.5 received; closed + material excluded
    assert 2 not in oo


def test_bom_demand(tmp_path):
    db = _db(tmp_path)
    d = ro.bom_demand([{"formulation_id": 1, "qty": 4}], db)
    assert d[1]["demand"] == 2.0                          # 0.5 dose × 4
    assert ro.bom_demand([], db) == {}


def test_reorder_report_par_and_plan(tmp_path):
    db = _db(tmp_path)
    # No plan: ingredient 1 par 3 − on_hand 1 − on_order 0 = shortfall 2 → reorder; ingredient 2 (5 ≥ 1) no line
    rep = ro.reorder_report(db_path=db)
    lines = [ln for g in rep["groups"] for ln in g["lines"]]
    by_ing = {ln["ingredient_id"]: ln for ln in lines}
    assert 2 not in by_ing
    assert by_ing[1]["shortfall"] == 2.0
    assert by_ing[1]["suggested_qty"] == 2.0             # MOQ 2 ≥ shortfall 2, on a 0.5 grid
    assert by_ing[1]["est_cost"] == 20.0                 # 2.0 × $10
    assert rep["groups"][0]["supplier"] == "NOW Foods"
    assert rep["groups"][0]["subtotal"] == 20.0
    # With a plan (4 units → demand 2): shortfall = 3 + 2 − 1 = 4 → suggested 4 (MOQ ok, 0.5 grid)
    rep2 = ro.reorder_report(plan=[{"formulation_id": 1, "qty": 4}], db_path=db)
    ln1 = [l for g in rep2["groups"] for l in g["lines"] if l["ingredient_id"] == 1][0]
    assert ln1["demand"] == 2.0 and ln1["shortfall"] == 4.0 and ln1["suggested_qty"] == 4.0
