import sqlite3
import app as app_module
from dashboard import recommendation_events as re


def _seed(tmp_path, monkeypatch, *, email):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    re.init_recommendation_events(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS query_log (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, session_id TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS cta_clicks (ts TEXT, log_id INTEGER, cta_type TEXT)")
    cx.execute("INSERT INTO query_log (id, email) VALUES (1, ?)", (email,))
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    # catalog stub: neuro-magnesium is a real product, junk-slug is not
    monkeypatch.setattr(app_module, "_cta_valid_product", lambda s: s == "neuro-magnesium", raising=False)
    app_module.app.config["TESTING"] = True
    return db


def test_identified_chat_click_records_chat_event(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch, email="a@b.com")
    c = app_module.app.test_client()
    r = c.post("/api/cta-click", json={"log_id": 1, "cta_type": "page", "slug": "neuro-magnesium"})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "chat" and e["product_key"] == "neuro-magnesium"
               for e in re.list_events(cx, "a@b.com"))


def test_anonymous_session_records_nothing(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch, email="")     # anonymous
    c = app_module.app.test_client()
    c.post("/api/cta-click", json={"log_id": 1, "cta_type": "page", "slug": "neuro-magnesium"})
    cx = sqlite3.connect(db)
    n = cx.execute("SELECT COUNT(*) FROM recommendation_events WHERE source_key='chat'").fetchone()[0]
    assert n == 0


def test_invalid_slug_records_nothing(tmp_path, monkeypatch):
    db = _seed(tmp_path, monkeypatch, email="a@b.com")
    c = app_module.app.test_client()
    c.post("/api/cta-click", json={"log_id": 1, "cta_type": "page", "slug": "junk-slug"})
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []
