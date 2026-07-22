import sqlite3

import app as app_module
from dashboard import client_portal as cp
from dashboard import supplement_reviews as sr


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


def test_add_listed_appears_in_block(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post(f"/api/portal/{token}/remedies/add",
               json={"product_name": "Vitamin D3", "product_brand": "NOW", "reason": "sun", "importance": 7})
    body = r.get_json()
    assert r.status_code == 200
    assert body["enabled"] is True
    ext = body["external"]
    assert len(ext) == 1
    assert ext[0]["product_name"] == "Vitamin D3"
    assert ext[0]["status"] == "listed"
    assert ext[0]["reason"] == "sun"
    assert ext[0]["importance"] == 7


def _add(c, token, name="Vitamin D3", brand="NOW", reason="sun", importance=7):
    r = c.post(f"/api/portal/{token}/remedies/add",
               json={"product_name": name, "product_brand": brand, "reason": reason, "importance": importance})
    return r.get_json()["external"][0]["product_key"]


def test_meta_updates_importance_and_preserves_reason(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    pk = _add(c, token)
    r = c.post(f"/api/portal/{token}/remedies/meta", json={"product_key": pk, "importance": 9})
    body = r.get_json()
    ext = [x for x in body["external"] if x["product_key"] == pk][0]
    assert ext["importance"] == 9
    assert ext["reason"] == "sun"  # reason preserved, not nulled


def test_meta_updates_reason_and_preserves_importance(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    pk = _add(c, token)
    r = c.post(f"/api/portal/{token}/remedies/meta", json={"product_key": pk, "reason": "immune support"})
    body = r.get_json()
    ext = [x for x in body["external"] if x["product_key"] == pk][0]
    assert ext["reason"] == "immune support"
    assert ext["importance"] == 7  # importance preserved, not nulled


def test_request_review_promotes_listed_to_requested(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    pk = _add(c, token)
    r = c.post(f"/api/portal/{token}/remedies/request-review",
               json={"product_name": "Vitamin D3", "product_brand": "NOW"})
    body = r.get_json()
    ext = [x for x in body["external"] if x["product_key"] == pk][0]
    assert ext["status"] == "requested"


def test_remove_drops_listed_item(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    pk = _add(c, token)
    r = c.post(f"/api/portal/{token}/remedies/remove", json={"product_key": pk})
    body = r.get_json()
    assert all(x["product_key"] != pk for x in body["external"])


def test_bad_token_404(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    assert c.post("/api/portal/not-a-real-token/remedies/add",
                  json={"product_name": "X"}).status_code == 404
    assert c.post("/api/portal/not-a-real-token/remedies/meta",
                  json={"product_key": "x"}).status_code == 404
    assert c.post("/api/portal/not-a-real-token/remedies/remove",
                  json={"product_key": "x"}).status_code == 404
    assert c.post("/api/portal/not-a-real-token/remedies/request-review",
                  json={"product_name": "X"}).status_code == 404


def test_flag_off_404(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "_PORTAL_REMEDIES_ENABLED", False, raising=False)
    c = app_module.app.test_client()
    assert c.post(f"/api/portal/{token}/remedies/add",
                  json={"product_name": "X"}).status_code == 404
    assert c.post(f"/api/portal/{token}/remedies/meta",
                  json={"product_key": "x"}).status_code == 404
    assert c.post(f"/api/portal/{token}/remedies/remove",
                  json={"product_key": "x"}).status_code == 404
    assert c.post(f"/api/portal/{token}/remedies/request-review",
                  json={"product_name": "X"}).status_code == 404


def test_cross_client_isolation(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx)
    sr.init_table(cx)
    token_a, _ = cp.upsert_portal(cx, "a@b.com", "Al", {})
    token_b, _ = cp.upsert_portal(cx, "b@b.com", "Bea", {})
    cx.commit()
    cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "_PORTAL_REMEDIES_ENABLED", True, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()

    c.post(f"/api/portal/{token_a}/remedies/add", json={"product_name": "A's Supp"})

    body_b = c.post(f"/api/portal/{token_b}/remedies/add", json={"product_name": "B's Supp"}).get_json()
    assert len(body_b["external"]) == 1
    assert body_b["external"][0]["product_name"] == "B's Supp"

    cx = sqlite3.connect(db)
    rows_a = sr.list_for_email(cx, "a@b.com")
    rows_b = sr.list_for_email(cx, "b@b.com")
    cx.close()
    assert len(rows_a) == 1 and rows_a[0]["product_name"] == "A's Supp"
    assert len(rows_b) == 1 and rows_b[0]["product_name"] == "B's Supp"
