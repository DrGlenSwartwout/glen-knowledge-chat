import json, sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from scripts.import_ingredients_from_fmp import import_ingredients, import_sources


def _rows_ing(name, par):
    return [{"id_pk": "i1", "name_common": name, "form": "powder", "active": "Yes",
             "par_level": par, "par_level_unit": "g"}]


def test_override_protects_field_per_row(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        import_ingredients(cx, _rows_ing("Mag", "100") + [{"id_pk": "i2", "name_common": "Lipoic", "active": "Yes", "par_level": "5", "par_level_unit": "g"}])
        cx.commit()
        # console edits ingredient i1's par_level → mark overridden
        cx.execute("UPDATE ingredients SET par_level=999, overrides=? WHERE fmp_id='i1'", (json.dumps(["par_level"]),))
        cx.commit()
        # re-import with a DIFFERENT par for both rows AND a changed name for i1
        # (the name change proves a non-overridden field on the partially-locked row still refreshes)
        import_ingredients(cx, [{"id_pk": "i1", "name_common": "Magnesium", "active": "Yes", "par_level": "100", "par_level_unit": "g"},
                                {"id_pk": "i2", "name_common": "Lipoic", "active": "Yes", "par_level": "50", "par_level_unit": "g"}])
        cx.commit()
        r1 = cx.execute("SELECT par_level, name FROM ingredients WHERE fmp_id='i1'").fetchone()
        r2 = cx.execute("SELECT par_level FROM ingredients WHERE fmp_id='i2'").fetchone()
    assert r1[0] == 999.0          # overridden par survived re-import
    assert r1[1] == "Magnesium"    # non-overridden field on the same (partially-locked) row DID refresh
    assert r2[0] == 50.0           # different (non-overridden) row refreshed normally
