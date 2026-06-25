# tests/test_import_materials.py
import sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from dashboard.materials_catalog import init_materials_schema
from scripts.import_materials_from_fmp import import_materials, import_material_suppliers, import_product_suppliers

def _db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')"); cx.commit()
    return p

def test_import(tmp_path):
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        nm = import_materials(cx, [{"id_pk":"m1","material_name":"Pullulan Caps","type":"Capsule","active":"Yes","par_level":"500"}])
        nms = import_material_suppliers(cx, [{"id_pk":"ms1","id_fk_material":"m1","id_fk_supplier":"s1","product_id":"SKU1","price":"10","purchase_size":"1000","purchase_size_unit":"ea"}])
        nps = import_product_suppliers(cx, [{"id_pk":"ps1","id_fk_product":"5161","id_fk_supplier":"s1","product_id":"SKU2","price":"20"}])
        cx.commit()
        m = cx.execute("SELECT * FROM materials WHERE fmp_id='m1'").fetchone()
        ms = cx.execute("SELECT ms.*, sup.company FROM material_suppliers ms JOIN suppliers sup ON sup.id=ms.supplier_id").fetchone()
        ps = cx.execute("SELECT * FROM product_suppliers WHERE fmp_id='ps1'").fetchone()
    assert nm==1 and m["name"]=="Pullulan Caps" and m["status"]=="active" and '"par_level": "500"' in m["extras"]
    assert nms==1 and ms["material_id"]==m["id"] and ms["sku"]=="SKU1" and ms["price"]==10.0 and ms["company"]=="Acme"
    assert nps==1 and ps["fmp_product_id"]=="5161" and ps["price"]==20.0

def test_reimport_preserves_curated(tmp_path):
    p = _db(tmp_path)
    from dashboard.materials_catalog import update_material_curated
    with sqlite3.connect(p) as cx:
        import_materials(cx, [{"id_pk":"m1","material_name":"Caps","active":"Yes"}]); cx.commit()
        mid = cx.execute("SELECT id FROM materials WHERE fmp_id='m1'").fetchone()[0]
    update_material_curated(mid, {"notes":"keep"}, db_path=p)
    with sqlite3.connect(p) as cx:
        import_materials(cx, [{"id_pk":"m1","material_name":"Caps RENAMED","active":"No"}]); cx.commit()
        cx.row_factory=sqlite3.Row
        m = cx.execute("SELECT * FROM materials WHERE fmp_id='m1'").fetchone()
    assert m["name"]=="Caps RENAMED" and m["notes"]=="keep"
