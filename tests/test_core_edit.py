import json, sqlite3
import pytest
from dashboard.ingredient_catalog import (
    init_ingredients_schema, set_ingredient_core, unlock_ingredient_core,
    set_source_core, unlock_source_core, get_ingredient,
)
from dashboard.formulations import init_formulations_schema, set_item_core, unlock_item_core


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
    # unlock non-allowlisted field is also rejected (guard added in task-4)
    with pytest.raises(ValueError):
        unlock_ingredient_core(1, "fmp_id", db_path=db)


def _db_with_source_and_item(tmp_path):
    """Return (db_path) with an ingredient, a source, and a formulation_item seeded."""
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        init_formulations_schema(cx)
        cx.execute("INSERT INTO ingredients (id,fmp_id,name) VALUES (1,'i1','Mag')")
        cx.execute(
            "INSERT INTO ingredient_sources (id,fmp_id,ingredient_id,price_per_unit,unit_size,unit_type)"
            " VALUES (10,'s1',1,5.0,100.0,'g')"
        )
        cx.execute(
            "INSERT INTO formulations (id,fmp_id,name) VALUES (1,'f1','TestForm')"
        )
        cx.execute(
            "INSERT INTO formulation_items (id,fmp_id,formulation_id,dose,dose_unit)"
            " VALUES (20,'fi1',1,50.0,'mg')"
        )
        cx.commit()
    return db


def test_source_core_numeric_coercion_and_allowlist(tmp_path):
    """set_source_core coerces numeric strings; rejects out-of-allowlist fields."""
    db = _db_with_source_and_item(tmp_path)
    # Numeric coercion
    set_source_core(10, "price_per_unit", "7.5", db_path=db)
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT price_per_unit, overrides FROM ingredient_sources WHERE id=10").fetchone()
    assert row[0] == 7.5
    assert "price_per_unit" in json.loads(row[1] or "[]")
    # Non-allowlisted field rejected
    with pytest.raises(ValueError):
        set_source_core(10, "fmp_id", "hacked", db_path=db)
    # unlock of valid field works, removes from overrides
    unlock_source_core(10, "price_per_unit", db_path=db)
    with sqlite3.connect(db) as cx:
        row2 = cx.execute("SELECT price_per_unit, overrides FROM ingredient_sources WHERE id=10").fetchone()
    assert row2[0] == 7.5  # value unchanged
    assert "price_per_unit" not in json.loads(row2[1] or "[]")
    # unlock of non-allowlisted field rejected (guard symmetry)
    with pytest.raises(ValueError):
        unlock_source_core(10, "fmp_id", db_path=db)


def test_item_core_and_allowlist(tmp_path):
    """set_item_core / unlock_item_core honour the _ITEM_CORE allowlist."""
    db = _db_with_source_and_item(tmp_path)
    set_item_core(20, "dose", "75", db_path=db)
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT dose, overrides FROM formulation_items WHERE id=20").fetchone()
    assert row[0] == 75.0
    assert "dose" in json.loads(row[1] or "[]")
    # Non-allowlisted field rejected
    with pytest.raises(ValueError):
        set_item_core(20, "fmp_id", "hacked", db_path=db)
    # unlock non-allowlisted rejected too
    with pytest.raises(ValueError):
        unlock_item_core(20, "fmp_id", db_path=db)
