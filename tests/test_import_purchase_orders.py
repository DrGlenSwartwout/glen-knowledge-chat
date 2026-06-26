import sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from dashboard.materials_catalog import init_materials_schema
from dashboard.purchase_orders import init_purchase_orders_schema
from scripts.import_purchase_orders_from_fmp import import_purchase_orders, import_po_items, import_po_receiving

def _db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx); init_purchase_orders_schema(cx)
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')")
        cx.execute("INSERT INTO ingredients (fmp_id,name) VALUES ('r1','R-Lipoic Acid')")
        cx.execute("INSERT INTO materials (fmp_id,name) VALUES ('m1','Caps')"); cx.commit()
    return p

def test_import(tmp_path):
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        npo = import_purchase_orders(cx, [{"id_pk":"p1","id_fk_supplier":"s1","vendor_po_no":"PO-1","po_date":"2026-01-01","closed":"No"}])
        ri = import_po_items(cx, [
            {"id_pk":"it1","id_fk_po":"p1","id_fk_raw":"r1","product_id":"SKU","qty":"2","cost":"50","qty_unit":"kg"},
            {"id_pk":"it2","id_fk_po":"p1","id_fk_material":"m1","qty":"10"},
            {"id_pk":"it3","id_fk_po":"p1","id_fk_product":"5161","fee_name":"Freight"},
        ])
        nr = import_po_receiving(cx, [{"id_pk":"rc1","id_fk_po":"p1","id_fk_po_item":"it1","qty_received":"2"}])
        cx.commit()
        po = cx.execute("SELECT * FROM purchase_orders WHERE fmp_id='p1'").fetchone()
        items = {r["fmp_id"]: dict(r) for r in cx.execute("SELECT * FROM po_items")}
        rec = cx.execute("SELECT * FROM po_receiving WHERE fmp_id='rc1'").fetchone()
    assert npo==1 and po["supplier_id"] is not None and po["vendor_po_no"]=="PO-1" and po["status"]=="open"
    assert items["it1"]["ingredient_id"] is not None and items["it1"]["item_kind"]=="ingredient" and items["it1"]["item_label"]=="R-Lipoic Acid" and items["it1"]["qty"]==2.0
    assert items["it2"]["material_id"] is not None and items["it2"]["item_kind"]=="material" and items["it2"]["item_label"]=="Caps"
    assert items["it3"]["fmp_product_id"]=="5161" and items["it3"]["item_kind"]=="product"
    assert ri["items"]==3
    assert nr==1 and rec["po_id"]==po["id"] and rec["po_item_id"]==items["it1"]["id"] and rec["qty_received"]==2.0

def test_reimport_preserves_curated(tmp_path):
    p = _db(tmp_path)
    from dashboard.purchase_orders import update_po_curated
    with sqlite3.connect(p) as cx:
        import_purchase_orders(cx, [{"id_pk":"p1","id_fk_supplier":"s1","vendor_po_no":"PO-1","closed":"No"}]); cx.commit()
        pid = cx.execute("SELECT id FROM purchase_orders WHERE fmp_id='p1'").fetchone()[0]
    update_po_curated(pid, {"notes":"keep"}, db_path=p)
    with sqlite3.connect(p) as cx:
        import_purchase_orders(cx, [{"id_pk":"p1","id_fk_supplier":"s1","vendor_po_no":"PO-1-REV","closed":"Yes"}]); cx.commit()
        cx.row_factory=sqlite3.Row
        po = cx.execute("SELECT * FROM purchase_orders WHERE fmp_id='p1'").fetchone()
    assert po["vendor_po_no"]=="PO-1-REV" and po["status"]=="closed" and po["notes"]=="keep"
