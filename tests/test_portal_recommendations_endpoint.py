import sqlite3
import app as app_module
from dashboard import recommendation_events as re, client_portal as cp
from dashboard import scan_recommendations as sr


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


def test_scan_section_contains_only_requested_scan(monkeypatch, tmp_path):
    db = str(tmp_path / "selected.db")
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    cp.init_client_portal_table(cx)
    re.init_recommendation_events(cx)
    sr.init_table(cx)
    token, _pid = cp.upsert_portal(cx, "scan@b.com", "Scan", {})
    sr.replace_scan(cx, "scan@b.com", "june", "2026-06-20", [
        {"item_code": "OLD", "priority_rank": 1, "section": "Infoceuticals"}])
    sr.replace_scan(cx, "scan@b.com", "aug", "2026-08-02", [
        {"item_code": "NEW", "priority_rank": 1, "section": "Infoceuticals"}])
    re.record_event(cx, "scan@b.com", "neuro-magnesium", "scan",
                    occurred_at="2026-06-20", origin_ref="scan:june:1")
    re.record_event(cx, "scan@b.com", "vitality", "scan",
                    occurred_at="2026-08-02", origin_ref="scan:aug:1")
    cx.close()

    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    data = app_module.app.test_client().get(
        f"/api/portal/{token}/recommendations?scan_date=2026-08-02").get_json()
    scan = next(s for s in data["sections"] if s["source"] == "scan")
    assert data["scan_date"] == "2026-08-02"
    assert [p["product_key"] for p in scan["products"]] == ["vitality"]
