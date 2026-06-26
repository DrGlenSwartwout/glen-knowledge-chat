import sqlite3
import pytest
from dashboard.ingredient_catalog import init_ingredients_schema
from dashboard.formulations import (
    init_formulations_schema, add_formulation_item, remove_formulation_item, list_items_for_formulation,
)


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_formulations_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name) VALUES (1,'HydroCurc')")
        cx.execute("INSERT INTO formulations (id,name) VALUES (1,'Brain Blend')")
        cx.commit()
    return db


def test_add_and_remove_item(tmp_path):
    db = _db(tmp_path)
    item = add_formulation_item(1, 1, "500", "mg", db_path=db)
    items = list_items_for_formulation(1, db_path=db)
    assert len(items) == 1 and items[0]["id"] == item and items[0]["dose"] == 500.0 and items[0]["dose_unit"] == "mg"
    with pytest.raises(ValueError):
        add_formulation_item(999, 1, "1", "mg", db_path=db)     # formulation must exist
    with pytest.raises(ValueError):
        add_formulation_item(1, 999, "1", "mg", db_path=db)     # ingredient must exist
    with pytest.raises(ValueError):
        add_formulation_item(1, 1, "abc", "mg", db_path=db)     # dose numeric
    remove_formulation_item(item, db_path=db)
    assert list_items_for_formulation(1, db_path=db) == []
