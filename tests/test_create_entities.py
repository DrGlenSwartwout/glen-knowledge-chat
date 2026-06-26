# tests/test_create_entities.py
import sqlite3
import pytest
from dashboard.ingredient_catalog import (
    init_ingredients_schema, create_ingredient, create_supplier, create_source, get_ingredient,
    list_sources_for_ingredient,
)
from scripts.import_ingredients_from_fmp import import_ingredients


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        cx.commit()
    return db


def test_create_ingredient(tmp_path):
    db = _db(tmp_path)
    iid = create_ingredient({"name": "HydroCurc", "form": "powder",
                             "common_names": "Curcumin, LipiSperse curcumin",
                             "par_level": "5", "par_level_unit": "kg"}, db_path=db)
    ing = get_ingredient(iid, db_path=db)
    assert ing["name"] == "HydroCurc" and ing["fmp_id"] is None and ing["par_level"] == 5.0
    import json
    assert json.loads(ing["common_names"]) == ["Curcumin", "LipiSperse curcumin"]
    with pytest.raises(ValueError):
        create_ingredient({"form": "powder"}, db_path=db)            # name required
    with pytest.raises(ValueError):
        create_ingredient({"name": "X", "fmp_id": "hack"}, db_path=db)  # non-creatable field


def test_create_supplier_and_source(tmp_path):
    db = _db(tmp_path)
    iid = create_ingredient({"name": "HydroCurc"}, db_path=db)
    sup = create_supplier({"company": "Pharmako Biotechnologies", "email": "enquiries@pharmako.com.au"}, db_path=db)
    sid = create_source(iid, {"supplier_id": sup, "price_per_unit": "334", "unit_size": "1",
                              "unit_type": "kg", "minimum_order": "25", "lead_time_days": "9",
                              "preferred": 1}, db_path=db)
    srcs = list_sources_for_ingredient(iid, db_path=db)
    assert len(srcs) == 1 and srcs[0]["id"] == sid
    assert srcs[0]["price_per_unit"] == 334.0 and srcs[0]["minimum_order"] == 25.0 and srcs[0]["preferred"] == 1
    with pytest.raises(ValueError):
        create_supplier({"email": "x@y.z"}, db_path=db)              # company required
    with pytest.raises(ValueError):
        create_source(99999, {"price_per_unit": "1"}, db_path=db)   # ingredient must exist


def test_created_ingredient_survives_reimport(tmp_path):
    """The core E2 invariant: an fmp_id=NULL console-created row is untouched + not duplicated by re-import."""
    db = _db(tmp_path)
    iid = create_ingredient({"name": "HydroCurc", "par_level": "5", "par_level_unit": "kg"}, db_path=db)
    with sqlite3.connect(db) as cx:
        # re-import a real FMP ingredient (different fmp_id) — must not touch the created row
        import_ingredients(cx, [{"id_pk": "f1", "name_common": "R-Lipoic Acid", "active": "Yes"}])
        cx.commit()
        rows = cx.execute("SELECT id, name, fmp_id, par_level FROM ingredients ORDER BY id").fetchall()
    assert len(rows) == 2                          # created + imported, no duplicate
    created = [r for r in rows if r[0] == iid][0]
    assert created[1] == "HydroCurc" and created[2] is None and created[3] == 5.0   # untouched


def test_create_source_preferred_normalized(tmp_path):
    """preferred is normalized to 0/1 regardless of odd truthy inputs (email collector may send these)."""
    db = _db(tmp_path)
    iid = create_ingredient({"name": "HydroCurc"}, db_path=db)
    sup = create_supplier({"company": "Pharmako"}, db_path=db)
    sid = create_source(iid, {"supplier_id": sup, "price_per_unit": "1", "preferred": 2}, db_path=db)
    srcs = list_sources_for_ingredient(iid, db_path=db)
    assert srcs[0]["preferred"] == 1          # 2 normalized to 1, and it's the preferred source
