import sqlite3

import app as app_module
from dashboard import client_portal as cp
from dashboard import supplement_reviews as sr
from dashboard import wishlist


def _seed(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx)
    sr.init_table(cx)
    token, _pid = cp.upsert_portal(cx, "a@b.com", "Al", {})
    cx.commit()
    cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_PORTAL_REMEDIES_ENABLED", True, raising=False)
    app_module.app.config["TESTING"] = True
    return db, token


def test_to_oasis_adds_slug_to_wishlist(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post(f"/api/portal/{token}/remedies/to-oasis", json={"slug": "vitamin-d3"})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}
    cx = sqlite3.connect(db)
    assert wishlist.slugs_for(cx, "email:a@b.com") == {"vitamin-d3"}


def test_to_oasis_is_idempotent_not_toggled_off(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    c.post(f"/api/portal/{token}/remedies/to-oasis", json={"slug": "vitamin-d3"})
    r2 = c.post(f"/api/portal/{token}/remedies/to-oasis", json={"slug": "vitamin-d3"})
    assert r2.status_code == 200
    assert r2.get_json() == {"ok": True}
    cx = sqlite3.connect(db)
    # A second post must NEVER remove it (toggle() would flip it off if called
    # unconditionally) -- it must still be present.
    assert wishlist.slugs_for(cx, "email:a@b.com") == {"vitamin-d3"}


def test_to_oasis_records_engagement_event(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    c.post(f"/api/portal/{token}/remedies/to-oasis", json={"slug": "vitamin-d3"})
    cx = sqlite3.connect(db)
    rows = cx.execute(
        "SELECT client_email, product_key, source_key FROM recommendation_events "
        "WHERE client_email=? AND product_key=?", ("a@b.com", "vitamin-d3")).fetchall()
    assert len(rows) == 1
    assert rows[0][2] == "my-remedies"


def test_to_oasis_unknown_token_404(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/api/portal/not-a-real-token/remedies/to-oasis", json={"slug": "vitamin-d3"})
    assert r.status_code == 404


def test_flag_off_404(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "_PORTAL_REMEDIES_ENABLED", False, raising=False)
    c = app_module.app.test_client()
    r = c.post(f"/api/portal/{token}/remedies/to-oasis", json={"slug": "vitamin-d3"})
    assert r.status_code == 404
