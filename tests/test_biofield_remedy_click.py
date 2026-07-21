import sqlite3
import app as app_module
from dashboard import recommendation_events as re


def test_remedy_click_records_biofield(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); re.init_recommendation_events(cx); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_biofield_verify_token", lambda th: (True, {"id": 1, "email": "a@b.com"}))
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/begin/biofield/anytok/remedy-click", json={"slug": "neuro-magnesium"})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "biofield" and e["product_key"] == "neuro-magnesium"
               for e in re.list_events(cx, "a@b.com"))


def test_remedy_click_bad_token(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db"); import sqlite3 as s; s.connect(db).close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_biofield_verify_token", lambda th: (False, None))
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/begin/biofield/bad/remedy-click", json={"slug": "x"})
    assert r.get_json()["ok"] is False
