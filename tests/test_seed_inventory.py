import json, sqlite3
from dashboard import inventory as inv


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT, extras TEXT)")
        cx.execute("CREATE TABLE purchase_orders (id INTEGER PRIMARY KEY, po_date TEXT, posted_date TEXT)")
        cx.execute("CREATE TABLE po_items (id INTEGER PRIMARY KEY, po_id INTEGER, ingredient_id INTEGER, material_id INTEGER)")
        cx.execute("CREATE TABLE po_receiving (id INTEGER PRIMARY KEY, po_id INTEGER, po_item_id INTEGER, qty_received REAL, received_size TEXT, created_at TEXT)")
        inv.init_inventory_schema(cx)
        cx.execute("INSERT INTO ingredients VALUES (1,'f1','Mag',?)",
                   (json.dumps({"inventory_starting": "1.0", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO ingredients VALUES (2,'f2','Lipoic',?)", (json.dumps({"par_level": "0.25"}),))  # no baseline
        cx.execute("INSERT INTO purchase_orders VALUES (10,'2026-01-01','2026-01-05')")
        cx.execute("INSERT INTO po_items VALUES (100,10,1,NULL)")        # ingredient line
        cx.execute("INSERT INTO po_items VALUES (101,10,NULL,7)")        # material-only line
        cx.execute("INSERT INTO po_receiving VALUES (1000,10,100,5.0,'kg','2026-01-06')")   # → ingredient 1
        cx.execute("INSERT INTO po_receiving VALUES (1001,10,101,9.0,'ea','2026-01-06')")   # material-only, skip
        cx.commit()
    return db


def test_seed_and_idempotent(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        nb = inv.seed_baselines(cx)
        nr = inv.seed_receipts(cx)
        cx.commit()
    assert nb == 1 and nr == 1                       # one baseline (ing 1), one receipt (ing 1 only)
    assert inv.on_hand(1, db) == 6.0                 # 1.0 baseline + 5.0 received
    assert inv.on_hand(2, db) == 0.0
    # a manual adjustment must survive a re-seed
    inv.add_adjustment(1, -0.5, db_path=db)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        nb2 = inv.seed_baselines(cx)
        nr2 = inv.seed_receipts(cx)
        cx.commit()
    assert nb2 == 0 and nr2 == 0                      # idempotent: nothing re-inserted
    assert inv.on_hand(1, db) == 5.5                  # 6.0 − 0.5, unchanged by re-seed


def test_receipt_date_from_po(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        inv.seed_receipts(cx); cx.commit()
    rows = inv.list_txns(1, db)
    rec = [t for t in rows if t["txn_type"] == "receipt"][0]
    assert rec["txn_date"] == "2026-01-05"            # posted_date preferred over po_date
    assert rec["source_ref"] == "po_receiving:1000" and rec["unit"] == "kg"
