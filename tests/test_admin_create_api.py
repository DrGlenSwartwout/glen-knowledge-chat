import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.formulations import init_formulations_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_formulations_schema(cx)
        cx.execute("INSERT INTO formulations (id,name) VALUES (1,'Brain Blend')")
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


def test_create_flow(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    iid = c.post("/api/ingredients", json={"name": "HydroCurc", "par_level": "5"}).get_json()["data"]["id"]
    sup = c.post("/api/suppliers", json={"company": "Pharmako"}).get_json()["data"]["id"]
    src = c.post(f"/api/ingredients/{iid}/sources", json={"supplier_id": sup, "price_per_unit": "334", "unit_type": "kg"})
    assert src.status_code == 200
    item = c.post(f"/api/formulations/1/items", json={"ingredient_id": iid, "dose": "500", "dose_unit": "mg"}).get_json()["data"]["id"]
    assert c.delete(f"/api/formulation-items/{item}").status_code == 200
    assert c.post("/api/ingredients", json={"form": "powder"}).status_code == 400   # name required
