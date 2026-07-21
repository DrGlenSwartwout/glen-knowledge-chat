import sqlite3
import app as app_module
from dashboard import recommendation_events as re, client_portal as cp


def test_portal_click_records_scan(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); cp.init_client_portal_table(cx); re.init_recommendation_events(cx)
    token, _pid = cp.upsert_portal(cx, "a@b.com", "Al", {})
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post(f"/api/portal/{token}/recommendation/click", json={"slug": "neuro-magnesium", "source": "scan"})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    assert any(e["source_key"] == "scan" and e["product_key"] == "neuro-magnesium"
               for e in re.list_events(cx, "a@b.com"))


def test_portal_click_rejects_unknown_source_and_bad_token(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); cp.init_client_portal_table(cx); re.init_recommendation_events(cx)
    token, _pid = cp.upsert_portal(cx, "a@b.com", "Al", {})
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    # unknown source -> no event (but ok:true or 400, implementer's choice — assert NO event recorded)
    c.post(f"/api/portal/{token}/recommendation/click", json={"slug": "x", "source": "not-a-source"})
    r = c.post("/api/portal/badtoken/recommendation/click", json={"slug": "x", "source": "scan"})
    assert r.status_code == 404
    cx = sqlite3.connect(db)
    assert re.list_events(cx, "a@b.com") == []
