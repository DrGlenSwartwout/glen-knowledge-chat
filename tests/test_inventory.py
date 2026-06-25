import json, sqlite3
import pytest
from dashboard import inventory as inv


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("""CREATE TABLE ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fmp_id TEXT, name TEXT, extras TEXT)""")
        inv.init_inventory_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name,extras) VALUES (1,'Mag L-threonate',?)",
                   (json.dumps({"inventory_starting": "1.0", "par_level": "3", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO ingredients (id,name,extras) VALUES (2,'R-Lipoic',?)",
                   (json.dumps({"par_level": "0.25", "par_level_unit": "kg"}),))
        cx.commit()
    return db


def test_on_hand_sums_signed(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',1.0)")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'receipt',5.0)")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'consumption',-2.0)")
        cx.commit()
    assert inv.on_hand(1, db) == 4.0
    assert inv.on_hand(2, db) == 0.0


def test_levels_below_par(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',4.0)")
        cx.commit()
    rows = {r["id"]: r for r in inv.inventory_levels(db_path=db)}
    assert rows[1]["on_hand"] == 4.0 and rows[1]["below_par"] == 0      # 4 >= 3
    assert rows[2]["on_hand"] == 0.0 and rows[2]["below_par"] == 1      # 0 < 0.25
    assert rows[1]["par_level_unit"] == "kg"


def test_add_adjustment_shifts_on_hand(tmp_path):
    db = _db(tmp_path)
    tid = inv.add_adjustment(1, -0.5, unit="kg", notes="recount", db_path=db)
    assert isinstance(tid, int) and tid > 0
    assert inv.on_hand(1, db) == -0.5
    with pytest.raises(ValueError):
        inv.add_adjustment(999, 1.0, db_path=db)          # no such ingredient


def test_update_txn_curated_notes_only(tmp_path):
    db = _db(tmp_path)
    tid = inv.add_adjustment(1, 1.0, db_path=db)
    inv.update_txn_curated(tid, {"notes": "x", "qty": 999, "txn_type": "hack"}, db_path=db)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        r = cx.execute("SELECT * FROM inventory_txns WHERE id=?", (tid,)).fetchone()
    assert r["notes"] == "x" and r["qty"] == 1.0 and r["txn_type"] == "adjustment"
