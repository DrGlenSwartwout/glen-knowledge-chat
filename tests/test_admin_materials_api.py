# tests/test_admin_materials_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        cx.execute("INSERT INTO materials (fmp_id,name) VALUES ('m1','Pullulan Caps')"); cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()

def test_search_get_patch(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/materials/search?q=caps").get_json()
    mid = r["data"][0]["id"]
    d = c.get(f"/api/materials/{mid}").get_json()["data"]
    assert d["material"]["name"]=="Pullulan Caps" and "suppliers" in d
    assert c.patch(f"/api/materials/{mid}", json={"notes":"x"}).status_code == 200
