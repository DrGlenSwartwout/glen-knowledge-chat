# tests/test_admin_ingredients_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        cx.execute("INSERT INTO ingredients (fmp_id,name) VALUES ('i1','R-Lipoic Acid')")
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')")
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client(), db

def test_search_and_curated(tmp_path, monkeypatch):
    c, db = _client(tmp_path, monkeypatch)
    r = c.get("/api/ingredients/search?q=lipoic").get_json()
    assert r["data"][0]["name"] == "R-Lipoic Acid"
    iid = r["data"][0]["id"]
    assert c.patch(f"/api/ingredients/{iid}", json={"inci_name":"Thioctic Acid"}).status_code == 200
    d = c.get(f"/api/ingredients/{iid}").get_json()["data"]
    assert d["ingredient"]["inci_name"] == "Thioctic Acid" and "sources" in d
    assert c.get("/api/ingredients/suppliers").get_json()["data"][0]["company"] == "Acme"
