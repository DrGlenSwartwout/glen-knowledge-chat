import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.sourcing import init_sourcing_schema, stage_quotes
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_sourcing_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name) VALUES (1,'Curcumin')")
        stage_quotes(cx, [{"gmail_msg_id": "m1", "ingredient_name": "Curcumin",
                           "price": 334.0, "price_unit": "kg", "moq": 25.0, "confidence": 0.9}])
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()


def test_sourcing_flow(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    q = c.get("/api/sourcing/quotes").get_json()["data"]
    qid = q[0]["id"]
    assert c.patch(f"/api/sourcing/quotes/{qid}", json={"ingredient_id": 1}).status_code == 200
    r = c.post(f"/api/sourcing/quotes/{qid}/approve")
    assert r.status_code == 200 and r.get_json()["data"]["source_id"]
