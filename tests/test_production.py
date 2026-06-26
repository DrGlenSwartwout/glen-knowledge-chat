import sqlite3
import pytest
from dashboard import production as prod
from dashboard import inventory as inv


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, name TEXT, extras TEXT)")
        cx.execute("CREATE TABLE materials (id INTEGER PRIMARY KEY, name TEXT)")
        cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
        cx.execute("""CREATE TABLE formulation_items (id INTEGER PRIMARY KEY, fmp_id TEXT,
            formulation_id INTEGER, ingredient_id INTEGER, ingredient_name TEXT,
            dose REAL, dose_unit TEXT, raw_text TEXT, extras TEXT, notes TEXT)""")
        inv.init_inventory_schema(cx)
        prod.init_production_schema(cx)
        cx.execute("INSERT INTO ingredients VALUES (1,'Mag',NULL)")
        cx.execute("INSERT INTO ingredients VALUES (2,'Lipoic',NULL)")
        cx.execute("INSERT INTO materials VALUES (5,'Capsule')")
        cx.execute("INSERT INTO formulations VALUES (1,'f1','Brain Blend')")
        cx.execute("INSERT INTO formulation_items (id,formulation_id,ingredient_id,ingredient_name,dose,dose_unit) VALUES (1,1,1,'Mag',2.0,'kg')")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',10.0)")
        cx.commit()
    return db


def test_log_run_posts_negative_consumption(tmp_path):
    db = _db(tmp_path)
    rid = prod.log_run(1, "2026-02-01", 100, [{"ingredient_id": 1, "qty_used": 3.0, "unit": "kg"}],
                       batch_number="B1", db_path=db)
    assert isinstance(rid, int) and rid > 0
    assert inv.on_hand(1, db) == 7.0                 # 10 baseline − 3 consumed
    items = prod.list_run_items(rid, db)
    assert items[0]["posted"] == 1 and items[0]["qty_used"] == 3.0
    with pytest.raises(ValueError):
        prod.log_run(999, "2026-02-01", 1, [{"ingredient_id": 1, "qty_used": 1}], db_path=db)
    with pytest.raises(ValueError):
        prod.log_run(1, "2026-02-01", 1, [], db_path=db)   # no items


def test_post_consumption_idempotent(tmp_path):
    db = _db(tmp_path)
    rid = prod.log_run(1, "2026-02-01", 100, [{"ingredient_id": 1, "qty_used": 3.0, "unit": "kg"}], db_path=db)
    inv.add_adjustment(1, -1.0, db_path=db)          # manual recount after the run
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        again = prod.post_consumption(cx, run_id=rid, mode="all")
        cx.commit()
    assert again == 0                                 # already posted → no double
    assert inv.on_hand(1, db) == 6.0                  # 7.0 − 1.0 recount, unchanged by re-post


def test_mode_record_only_and_from_date(tmp_path):
    db = _db(tmp_path)
    # two manual runs on different dates, but post nothing at creation by inserting rows directly
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        cx.execute("INSERT INTO production_runs (id,formulation_id,run_date,quantity_units,source_kind) VALUES (10,1,'2025-01-01',50,'manual')")
        cx.execute("INSERT INTO production_runs (id,formulation_id,run_date,quantity_units,source_kind) VALUES (11,1,'2026-05-01',50,'manual')")
        cx.execute("INSERT INTO production_run_items (id,production_run_id,item_type,ingredient_id,qty_used,unit) VALUES (100,10,'ingredient',1,2.0,'kg')")
        cx.execute("INSERT INTO production_run_items (id,production_run_id,item_type,ingredient_id,qty_used,unit) VALUES (101,11,'ingredient',1,4.0,'kg')")
        cx.execute("INSERT INTO production_run_items (id,production_run_id,item_type,material_id,qty_used,unit) VALUES (102,11,'material',5,9.0,'ea')")
        cx.commit()
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        assert prod.post_consumption(cx, mode="record_only") == 0
        cx.commit()
    assert inv.on_hand(1, db) == 10.0                 # nothing posted
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        n = prod.post_consumption(cx, mode="from_date", cutoff_date="2026-01-01")
        cx.commit()
    assert n == 1                                     # only run 11's ingredient line (run 10 pre-cutoff; material skipped)
    assert inv.on_hand(1, db) == 6.0                  # 10 − 4
