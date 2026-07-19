import importlib, json, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.formulations import init_formulations_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    from dashboard.inventory import init_inventory_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_formulations_schema(cx)
        init_purchase_orders_schema(cx); init_inventory_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name,extras) VALUES (1,'Mag',?)",
                   (json.dumps({"par_level": "3", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO formulations (id,name) VALUES (1,'Brain Blend')")
        cx.execute("INSERT INTO formulation_items (formulation_id,ingredient_id,ingredient_name,dose,dose_unit) VALUES (1,1,'Mag',0.5,'kg')")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',1.0)")
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


def test_reorder_get_and_post(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    g = c.get("/api/reorder/report").get_json()["data"]
    line = [l for grp in g["groups"] for l in grp["lines"] if l["ingredient_id"] == 1][0]
    assert line["shortfall"] == 2.0                       # par 3 − on_hand 1
    p = c.post("/api/reorder/report", json={"plan": [{"formulation_id": 1, "qty": 4}], "include_below_par": True}).get_json()["data"]
    line2 = [l for grp in p["groups"] for l in grp["lines"] if l["ingredient_id"] == 1][0]
    assert line2["shortfall"] == 4.0                      # 3 + (0.5×4) − 1
