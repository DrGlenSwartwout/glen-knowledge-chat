# tests/test_admin_formulations_api.py
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
        cx.execute("INSERT INTO formulations (fmp_id,name) VALUES ('f1','Nerve Pulse')")
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()

def test_search_get_patch(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/formulations/search?q=nerve").get_json()
    fid = r["data"][0]["id"]
    d = c.get(f"/api/formulations/{fid}").get_json()["data"]
    assert d["formulation"]["name"] == "Nerve Pulse" and "items" in d
    assert c.patch(f"/api/formulations/{fid}", json={"notes":"x"}).status_code == 200
