# tests/test_import_ingredients.py
import sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from scripts.import_ingredients_from_fmp import (
    _active, _num, _clean, import_suppliers, import_ingredients, import_sources, apply_canonical)

def _db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx)
    return p

def test_helpers():
    assert _active("Yes") == 1 and _active("No") == 0 and _active("") is None
    assert _num("350") == 350.0 and _num("$1,000.5") == 1000.5 and _num("") is None
    assert _clean("Inositol\nFlush Free") == "Inositol Flush Free"

def test_import_and_join_and_canonical(tmp_path):
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        import_suppliers(cx, [{"id_pk":"s1","company":"Acme","active":"Yes","notes":"x","zc_junk":"drop"}])
        import_ingredients(cx, [
            {"id_pk":"5161","name_common":"CBD","active":"Yes","type":"Cannabinoid"},
            {"id_pk":"4138","name_compound":"Cannabidiol\nfull spectrum","active":"Yes"},
            {"id_pk":"i3","name_common":"","name_compound":""},   # unnamed fallback
        ])
        import_sources(cx, [{"id_pk":"src1","id_fk_raw":"5161","id_fk_supplier":"s1",
                             "product_id":"SKU9","price":"350","purchase_size":"1000","purchase_size_unit":"g"}])
        res = apply_canonical(cx, [{"head_fmp_id":"5161","member_fmp_id":"4138"}])
        cx.commit()
        ing = {r["fmp_id"]: dict(r) for r in cx.execute("SELECT * FROM ingredients")}
        src = cx.execute("SELECT s.*, sup.company FROM ingredient_sources s JOIN suppliers sup ON sup.id=s.supplier_id").fetchone()
    assert ing["i3"]["name"] == "(unnamed FMP ingredient i3)"
    assert ing["4138"]["name"] == "Cannabidiol full spectrum"
    assert ing["4138"]["canonical_id"] == ing["5161"]["id"]   # member -> head
    assert ing["5161"]["canonical_id"] is None                # head not clustered under anyone
    assert src["sku"] == "SKU9" and src["price_per_unit"] == 350.0 and src["unit_type"] == "g"
    assert src["company"] == "Acme"
    assert res["applied"] == 1

def test_reimport_preserves_curated(tmp_path):
    p = _db(tmp_path)
    from dashboard.ingredient_catalog import update_ingredient_curated, update_source_curated
    with sqlite3.connect(p) as cx:
        import_suppliers(cx, [{"id_pk":"s1","company":"Acme"}])
        import_ingredients(cx, [{"id_pk":"i1","name_common":"R-Lipoic Acid","type":"old"}])
        import_sources(cx, [{"id_pk":"src1","id_fk_raw":"i1","id_fk_supplier":"s1","price":"10"}])
        cx.commit()
        iid = cx.execute("SELECT id FROM ingredients WHERE fmp_id='i1'").fetchone()[0]
        sid = cx.execute("SELECT id FROM ingredient_sources WHERE fmp_id='src1'").fetchone()[0]
    update_ingredient_curated(iid, {"inci_name": "Thioctic Acid"}, db_path=p)
    update_source_curated(sid, {"lead_time_days": 21}, db_path=p)
    with sqlite3.connect(p) as cx:  # re-import with changed FMP data
        cx.row_factory = sqlite3.Row
        import_ingredients(cx, [{"id_pk":"i1","name_common":"R-Lipoic Acid","type":"new"}])
        import_sources(cx, [{"id_pk":"src1","id_fk_raw":"i1","id_fk_supplier":"s1","price":"99"}])
        cx.commit()
        ing = cx.execute("SELECT * FROM ingredients WHERE fmp_id='i1'").fetchone()
        src = cx.execute("SELECT * FROM ingredient_sources WHERE fmp_id='src1'").fetchone()
    assert ing["inci_name"] == "Thioctic Acid"      # curated preserved
    assert '"type": "new"' in ing["extras"]          # FMP refreshed
    assert src["lead_time_days"] == 21               # curated preserved
    assert src["price_per_unit"] == 99.0             # FMP refreshed
