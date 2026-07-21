import sqlite3
import app as app_module
from dashboard import recommendation_events as re, recommendation_prefs as rp, client_360


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    re.init_recommendation_events(cx); rp.init_recommendation_prefs(cx)
    return cx


def test_bundle_recommendations_include_notes():
    cx = _cx()
    re.record_event(cx, "a@b.com", "neuro-magnesium", "purchased", occurred_at="d", origin_ref="1")
    rp.set_operator_note(cx, "a@b.com", "neuro-magnesium", "night dose")
    b = client_360.bundle(cx, "a@b.com")
    rec = next(p for p in b["recommendations"] if p["product_key"] == "neuro-magnesium")
    assert rec["operator_note"] == "night dose"
    assert "client_note" in rec


def test_operator_note_endpoint_writes(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); re.init_recommendation_events(cx); rp.init_recommendation_prefs(cx); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_bos_actor", lambda: object())
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/api/console/client/recommendation/operator-note",
               json={"email": "a@b.com", "product_key": "neuro-magnesium", "note": "am"})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    assert rp.get_notes(cx, "a@b.com")["neuro-magnesium"]["operator_note"] == "am"


def test_operator_note_endpoint_requires_auth(monkeypatch):
    monkeypatch.setattr(app_module, "_bos_actor", lambda: None)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/api/console/client/recommendation/operator-note",
               json={"email": "a@b.com", "product_key": "x", "note": "y"})
    assert r.status_code == 401
