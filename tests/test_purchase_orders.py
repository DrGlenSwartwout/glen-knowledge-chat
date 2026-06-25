import sqlite3, pytest

@pytest.fixture
def db(tmp_path):
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        init_purchase_orders_schema(cx); init_purchase_orders_schema(cx)
    return p

def test_schema_reads_curated(db):
    from dashboard.purchase_orders import (search_purchase_orders, get_purchase_order,
        list_po_items, list_po_receiving, update_po_curated)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')")
        sid = cx.execute("SELECT id FROM suppliers").fetchone()[0]
        cx.execute("INSERT INTO ingredients (fmp_id,name) VALUES ('r1','R-Lipoic Acid')")
        iid = cx.execute("SELECT id FROM ingredients").fetchone()[0]
        cx.execute("INSERT INTO purchase_orders (fmp_id,supplier_id,vendor_po_no,po_date) VALUES ('p1',?,'PO-100','2026-01-01')",(sid,))
        pid = cx.execute("SELECT id FROM purchase_orders").fetchone()[0]
        cx.execute("INSERT INTO po_items (fmp_id,po_id,ingredient_id,item_label,qty,cost) VALUES ('it1',?,?, 'R-Lipoic Acid', 2, 50)",(pid,iid))
        itid = cx.execute("SELECT id FROM po_items").fetchone()[0]
        cx.execute("INSERT INTO po_receiving (fmp_id,po_id,po_item_id,qty_received) VALUES ('rc1',?,?,2)",(pid,itid))
        cx.commit()
    r = search_purchase_orders("PO-100", db_path=db)
    assert r[0]["vendor_po_no"]=="PO-100" and r[0]["supplier_company"]=="Acme"
    items = list_po_items(pid, db_path=db)
    assert items[0]["qty"]==2 and items[0]["item_label"]=="R-Lipoic Acid" and items[0]["ingredient_canonical"]=="R-Lipoic Acid"
    assert list_po_receiving(pid, db_path=db)[0]["qty_received"]==2
    update_po_curated(pid, {"notes":"x","vendor_po_no":"HACK"}, db_path=db)
    g = get_purchase_order(pid, db_path=db)
    assert g["notes"]=="x" and g["vendor_po_no"]=="PO-100"
