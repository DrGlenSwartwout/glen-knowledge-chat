import json, sqlite3
import pytest
from dashboard.ingredient_catalog import init_ingredients_schema, set_ingredient_core, unlock_ingredient_core, get_ingredient


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        cx.execute("INSERT INTO ingredients (id,fmp_id,name) VALUES (1,'i1','Mag')")
        cx.commit()
    return db


def test_set_and_unlock_core(tmp_path):
    db = _db(tmp_path)
    set_ingredient_core(1, "par_level", "12", db_path=db)
    set_ingredient_core(1, "common_names", "Magnesium, Mag glycinate", db_path=db)
    ing = get_ingredient(1, db_path=db)
    assert ing["par_level"] == 12.0
    assert json.loads(ing["common_names"]) == ["Magnesium", "Mag glycinate"]
    assert set(json.loads(ing["overrides"])) == {"par_level", "common_names"}
    # non-allowlisted field rejected
    with pytest.raises(ValueError):
        set_ingredient_core(1, "fmp_id", "hacked", db_path=db)
    # non-numeric par rejected
    with pytest.raises(ValueError):
        set_ingredient_core(1, "par_level", "abc", db_path=db)
    # unlock removes from overrides, leaves value
    unlock_ingredient_core(1, "par_level", db_path=db)
    ing = get_ingredient(1, db_path=db)
    assert ing["par_level"] == 12.0
    assert set(json.loads(ing["overrides"])) == {"common_names"}
