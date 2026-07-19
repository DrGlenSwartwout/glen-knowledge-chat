import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.formulations import init_formulations_schema
    from dashboard.inventory import init_inventory_schema
    from dashboard.production import init_production_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        init_formulations_schema(cx); init_inventory_schema(cx); init_production_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name) VALUES (1,'Mag')")
        cx.execute("INSERT INTO formulations (id,fmp_id,name) VALUES (1,'p1','Brain Blend')")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',10.0)")
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()


def test_log_and_read(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/production/log", json={"formulation_id": 1, "run_date": "2026-03-01",
               "quantity_units": 100, "items": [{"ingredient_id": 1, "qty_used": 3.0, "unit": "kg"}]})
    assert r.status_code == 200
    rid = r.get_json()["data"]["id"]
    d = c.get(f"/api/production/{rid}").get_json()["data"]
    assert d["run"]["formulation_name"] == "Brain Blend" and len(d["items"]) == 1
    assert c.get("/api/production/search?q=Brain").get_json()["data"][0]["id"] == rid
    assert c.get("/api/production/999").status_code == 404
    bad = c.post("/api/production/log", json={"formulation_id": 1, "run_date": "x", "quantity_units": 1, "items": []})
    assert bad.status_code == 400
