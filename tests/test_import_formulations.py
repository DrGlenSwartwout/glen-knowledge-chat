import sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from dashboard.formulations import init_formulations_schema
from scripts.import_formulations_from_fmp import import_formulations, import_formulation_items

def _db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx); init_formulations_schema(cx)
        cx.execute("INSERT INTO ingredients (fmp_id, name) VALUES ('r1','R-Lipoic Acid')")
        cx.commit()
    return p

def test_import_formulations_and_items(tmp_path):
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        nprod = import_formulations(cx, [
            {"id_pk":"f1","product_name":"Nerve Pulse","type":"Functional Formulation","active":"Yes"},
            {"id_pk":"x9","product_name":"Not A Formula","type":"Product","active":"Yes"},  # skipped
        ])
        res = import_formulation_items(cx, [
            {"id_pk":"it1","id_fk_product":"f1","id_fk_raw":"r1","zc_raw_display":"100mg - R-Lipoic Acid","zc_mg":"100","qty":"1","unit_measurement":"ea."},
            {"id_pk":"it2","id_fk_product":"f1","id_fk_raw":"UNKNOWN","zc_raw_display":"50mg - Mystery","zc_mg":"50"},
        ], ff_product_ids={"f1"})
        cx.commit()
        forms = cx.execute("SELECT * FROM formulations").fetchall()
        items = {r["fmp_id"]: dict(r) for r in cx.execute("SELECT * FROM formulation_items")}
    assert nprod == 1 and forms[0]["name"] == "Nerve Pulse"
    assert items["it1"]["ingredient_id"] is not None and items["it1"]["dose"] == 100.0 and items["it1"]["dose_unit"] == "mg"
    assert items["it1"]["ingredient_name"] == "R-Lipoic Acid"     # from zc_raw_display after " - "
    assert items["it2"]["ingredient_id"] is None                 # unresolved id_fk_raw kept, not dropped
    assert res["unresolved"] == 1

def test_reimport_preserves_curated(tmp_path):
    p = _db(tmp_path)
    from dashboard.formulations import update_formulation_curated
    with sqlite3.connect(p) as cx:
        import_formulations(cx, [{"id_pk":"f1","product_name":"Nerve Pulse","type":"Functional Formulation","active":"Yes"}]); cx.commit()
        fid = cx.execute("SELECT id FROM formulations WHERE fmp_id='f1'").fetchone()[0]
    update_formulation_curated(fid, {"notes":"keep me"}, db_path=p)
    with sqlite3.connect(p) as cx:
        import_formulations(cx, [{"id_pk":"f1","product_name":"Nerve Pulse RENAMED","type":"Functional Formulation","active":"No"}]); cx.commit()
        r = cx.execute("SELECT * FROM formulations WHERE fmp_id='f1'").fetchone()
    assert r[2] == "Nerve Pulse RENAMED"  # name (FMP) refreshed; col order: id,fmp_id,name
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        r = cx.execute("SELECT * FROM formulations WHERE fmp_id='f1'").fetchone()
    assert r["notes"] == "keep me"        # curated preserved


def test_reimport_protects_overridden_item_dose(tmp_path):
    """Re-import of formulation_items against a POPULATED table must (a) not NameError on the
    overrides preload (regression: import json was missing), and (b) preserve a console-overridden
    dose while still refreshing non-overridden fields on the same row."""
    import json
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        import_formulations(cx, [{"id_pk": "f1", "product_name": "Nerve Pulse", "type": "Functional Formulation", "active": "Yes"}])
        import_formulation_items(cx, [
            {"id_pk": "it1", "id_fk_product": "f1", "id_fk_raw": "r1", "zc_raw_display": "100mg - R-Lipoic Acid", "zc_mg": "100"},
        ], ff_product_ids={"f1"})
        cx.commit()
        # console overrides the dose
        cx.execute("UPDATE formulation_items SET dose=999, overrides=? WHERE fmp_id='it1'", (json.dumps(["dose"]),))
        cx.commit()
        # re-import against the now-POPULATED table (this path NameError'd before the json-import fix)
        import_formulation_items(cx, [
            {"id_pk": "it1", "id_fk_product": "f1", "id_fk_raw": "r1", "zc_raw_display": "100mg - R-Lipoic Acid (updated)", "zc_mg": "100"},
        ], ff_product_ids={"f1"})
        cx.commit()
        row = cx.execute("SELECT dose, ingredient_name FROM formulation_items WHERE fmp_id='it1'").fetchone()
    assert row["dose"] == 999.0                                   # overridden dose survived re-import
    assert row["ingredient_name"] == "R-Lipoic Acid (updated)"    # non-overridden field DID refresh
