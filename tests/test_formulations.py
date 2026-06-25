import sqlite3, pytest

@pytest.fixture
def db(tmp_path):
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.formulations import init_formulations_schema
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx)           # Phase-1 tables (formulation_items FKs ingredients)
        init_formulations_schema(cx)
        init_formulations_schema(cx)           # idempotent
    return p

def test_schema_and_reads(db):
    from dashboard.formulations import (search_formulations, get_formulation,
        list_items_for_formulation, update_formulation_curated, update_item_curated)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO ingredients (fmp_id, name) VALUES ('r1','R-Lipoic Acid')")
        iid = cx.execute("SELECT id FROM ingredients").fetchone()[0]
        cx.execute("INSERT INTO formulations (fmp_id, name) VALUES ('f1','Nerve Pulse')")
        fid = cx.execute("SELECT id FROM formulations").fetchone()[0]
        cx.execute("INSERT INTO formulation_items (fmp_id, formulation_id, ingredient_id, ingredient_name, dose, dose_unit) "
                   "VALUES ('it1',?,?, 'R-Lipoic Acid', 100, 'mg')", (fid, iid))
        cx.commit()
    assert search_formulations("nerve", db_path=db)[0]["name"] == "Nerve Pulse"
    items = list_items_for_formulation(fid, db_path=db)
    assert items[0]["dose"] == 100 and items[0]["ingredient_name"] == "R-Lipoic Acid"
    assert items[0]["ingredient_canonical"] == "R-Lipoic Acid"   # from the JOIN
    update_formulation_curated(fid, {"notes": "test", "name": "HACK"}, db_path=db)
    assert get_formulation(fid, db_path=db)["notes"] == "test"
    assert get_formulation(fid, db_path=db)["name"] == "Nerve Pulse"
