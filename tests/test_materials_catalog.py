import sqlite3, pytest

@pytest.fixture
def db(tmp_path):
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx)   # suppliers table (material_suppliers FK)
        init_materials_schema(cx); init_materials_schema(cx)  # idempotent
    return p

def test_schema_reads_curated(db):
    from dashboard.materials_catalog import (search_materials, get_material,
        list_suppliers_for_material, update_material_curated, set_preferred_material_supplier)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')")
        sid = cx.execute("SELECT id FROM suppliers").fetchone()[0]
        cx.execute("INSERT INTO materials (fmp_id,name) VALUES ('m1','Pullulan Caps')")
        mid = cx.execute("SELECT id FROM materials").fetchone()[0]
        cx.execute("INSERT INTO material_suppliers (fmp_id,material_id,supplier_id,price) VALUES ('a',?,?,10)", (mid,sid))
        cx.execute("INSERT INTO material_suppliers (fmp_id,material_id,supplier_id,price) VALUES ('b',?,?,20)", (mid,sid))
        ids=[r[0] for r in cx.execute("SELECT id FROM material_suppliers ORDER BY id")]; cx.commit()
    assert search_materials("pullulan", db_path=db)[0]["name"] == "Pullulan Caps"
    sup = list_suppliers_for_material(mid, db_path=db)
    assert sup[0]["company"] == "Acme"
    update_material_curated(mid, {"notes":"x","name":"HACK"}, db_path=db)
    assert get_material(mid, db_path=db)["notes"]=="x" and get_material(mid, db_path=db)["name"]=="Pullulan Caps"
    set_preferred_material_supplier(ids[1], db_path=db)
    pref={r["id"]:r for r in list_suppliers_for_material(mid, db_path=db)}
    assert pref[ids[1]]["preferred"]==1 and pref[ids[0]]["preferred"]==0
