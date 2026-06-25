"""Route-level tests for PATCH /api/*/core and POST /api/*/unlock endpoints.

Skipped automatically if app fails to import (e.g. Pinecone not configured).
"""
import importlib, json, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        cx.execute("INSERT INTO ingredients (id,fmp_id,name) VALUES (1,'i1','Mag')")
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


def test_core_edit_and_unlock(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    # PATCH sets a numeric core field
    r = c.patch("/api/ingredients/1/core", json={"field": "par_level", "value": "9"})
    assert r.status_code == 200, r.get_data(as_text=True)
    # GET confirms value persisted and override recorded
    d = c.get("/api/ingredients/1").get_json()["data"]["ingredient"]
    assert d["par_level"] == 9.0
    assert "par_level" in json.loads(d["overrides"])
    # Non-allowlisted field returns 400
    assert c.patch("/api/ingredients/1/core", json={"field": "fmp_id", "value": "x"}).status_code == 400
    # Missing field key also returns 400
    assert c.patch("/api/ingredients/1/core", json={"value": "5"}).status_code == 400
    # Unlock succeeds
    r_unlock = c.post("/api/ingredients/1/unlock", json={"field": "par_level"})
    assert r_unlock.status_code == 200, r_unlock.get_data(as_text=True)
    # Unlock non-allowlisted field returns 400 (ValueError caught by the route handler)
    r_bad_unlock = c.post("/api/ingredients/1/unlock", json={"field": "fmp_id"})
    assert r_bad_unlock.status_code == 400
