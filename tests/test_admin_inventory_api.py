import importlib, json, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    from dashboard.inventory import init_inventory_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        init_purchase_orders_schema(cx); init_inventory_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name,extras) VALUES (1,'Mag',?)",
                   (json.dumps({"par_level": "3", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',1.0)")
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()


def test_levels_get_adjust_patch(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    lv = c.get("/api/inventory/levels?q=Mag").get_json()["data"]
    assert lv[0]["id"] == 1 and lv[0]["on_hand"] == 1.0 and lv[0]["below_par"] == 1
    d = c.get("/api/inventory/1").get_json()["data"]
    assert d["on_hand"] == 1.0 and "txns" in d
    r = c.post("/api/inventory/1/adjust", json={"qty": 2.5, "notes": "recount"}).get_json()
    assert r["data"]["on_hand"] == 3.5
    tid = c.get("/api/inventory/1").get_json()["data"]["txns"][0]["id"]
    assert c.patch(f"/api/inventory/txns/{tid}", json={"notes": "z"}).status_code == 200
    assert c.get("/api/inventory/999").status_code == 404
