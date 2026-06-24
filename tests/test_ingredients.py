import sqlite3, pytest

@pytest.fixture
def db(tmp_path):
    from dashboard.ingredients import init_ingredients_schema
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx)
        init_ingredients_schema(cx)  # idempotent
    return p

def test_schema_creates_tables(db):
    with sqlite3.connect(db) as cx:
        tables = {r[0] for r in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"suppliers", "ingredients", "ingredient_sources"} <= tables

def test_search_and_get(db):
    from dashboard.ingredients import search_ingredients, get_ingredient, list_sources_for_ingredient
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO suppliers (fmp_id, company) VALUES ('s1','Acme')")
        sid = cx.execute("SELECT id FROM suppliers").fetchone()[0]
        cx.execute("INSERT INTO ingredients (fmp_id, name) VALUES ('i1','R-Lipoic Acid')")
        iid = cx.execute("SELECT id FROM ingredients").fetchone()[0]
        cx.execute("INSERT INTO ingredient_sources (fmp_id, ingredient_id, supplier_id, price_per_unit) VALUES ('x1',?,?,12.5)", (iid, sid))
        cx.commit()
    assert search_ingredients("lipoic", db_path=db)[0]["name"] == "R-Lipoic Acid"
    assert get_ingredient(iid, db_path=db)["fmp_id"] == "i1"
    srcs = list_sources_for_ingredient(iid, db_path=db)
    assert srcs[0]["price_per_unit"] == 12.5 and srcs[0]["company"] == "Acme"
