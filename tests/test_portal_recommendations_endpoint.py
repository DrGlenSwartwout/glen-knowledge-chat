import sqlite3
import app as app_module
from dashboard import recommendation_events as re, client_portal as cp


def test_portal_recommendations_endpoint(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx)
    re.init_recommendation_events(cx)
    # a portal token for the client — upsert_portal mints and returns the RAW token
    # on first create (only its sha256 hash is persisted in client_portals).
    token, _pid = cp.upsert_portal(cx, "a@b.com", "Al", {})
    assert token
    # seed a purchased event
    re.record_event(cx, "a@b.com", "neuro-magnesium", "purchased", occurred_at="2026-07-10", origin_ref="7")
    cx.commit()
    cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.get(f"/api/portal/{token}/recommendations")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert any(s["source"] == "purchased" and
               any(p["product_key"] == "neuro-magnesium" for p in s["products"]) for s in data["sections"])


def test_portal_recommendations_unknown_token_404(monkeypatch, tmp_path):
    db = str(tmp_path / "log2.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx)
    re.init_recommendation_events(cx)
    cx.commit()
    cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.get("/api/portal/not-a-real-token/recommendations")
    assert r.status_code == 404
