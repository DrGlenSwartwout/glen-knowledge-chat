# tests/test_console_biofield_portal.py
import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


_LAYER = {"n": 1, "title": "Calm", "meaning": "settle", "remedy": "Terrain Restore",
          "dosing": "10 drops 3x/day"}


def test_post_creates_portal_and_returns_url(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "x@y.com", "name": "X",
                     "content": {"greeting": "Aloha", "layers": [_LAYER]}})
    assert r.status_code == 200
    j = r.get_json()
    assert j["token"] and j["url"].endswith(j["token"])
    # content round-trips through the public portal API
    r2 = c.get(f"/api/portal/{j['token']}")
    assert r2.get_json()["layers"][0]["title"] == "Calm"


def test_post_requires_console_key(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal",
               json={"email": "x@y.com", "content": {"layers": [_LAYER]}})
    assert r.status_code == 401


def test_post_requires_email(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"content": {"layers": [_LAYER]}})
    assert r.status_code == 400


def test_post_requires_some_content(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "x@y.com", "name": "X", "content": {}})
    assert r.status_code == 400


def test_post_send_emails_link(client, monkeypatch):
    c, appmod = client
    sent = {}
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda to, name, subj, body, **k: sent.update(to=to, body=body))
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "e@y.com", "name": "E", "send": True,
                     "content": {"greeting": "hi", "layers": [_LAYER]}})
    tok = r.get_json()["token"]
    assert sent["to"] == "e@y.com" and tok in sent["body"]


def _seed(appmod, email="seed@y.com", name="Seed"):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    cp.upsert_portal(cx, email, name, {"greeting": "hello", "layers": [_LAYER]})
    cx.close()


def test_get_loads_existing_portal(client):
    c, appmod = client
    _seed(appmod, "seed@y.com", "Seed")
    j = c.get("/api/console/biofield-portal?key=test-secret&email=seed@y.com").get_json()
    assert j["found"] is True
    assert j["name"] == "Seed"
    assert j["content"]["layers"][0]["title"] == "Calm"


def test_get_unknown_returns_scaffold(client):
    c, _ = client
    j = c.get("/api/console/biofield-portal?key=test-secret&email=nobody@y.com").get_json()
    assert j["found"] is False
    assert j["content"] == {}


def test_get_requires_key(client):
    c, _ = client
    assert c.get("/api/console/biofield-portal?email=x@y.com").status_code == 401


def test_catalog_returns_products(client):
    c, _ = client
    j = c.get("/api/console/biofield-portal/catalog?key=test-secret").get_json()
    assert isinstance(j["products"], list) and j["products"]
    assert "slug" in j["products"][0] and "name" in j["products"][0]


def test_page_served(client):
    c, _ = client
    assert c.get("/console/biofield-portal").status_code == 200


# ── Import from FMP ──────────────────────────────────────────────────────────

def test_import_fmp_returns_content(client, monkeypatch):
    c, _ = client
    from dashboard import fmp_biofield
    monkeypatch.setattr(fmp_biofield, "import_content",
        lambda email, name="", tags=None: {"greeting": "G",
            "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "R", "dosing": "d"}],
            "video": {}, "reorder_items": [], "pricing_note": ""})
    r = c.post("/api/console/biofield-portal/import-fmp?key=test-secret",
               json={"email": "e@x.com", "name": "O"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["found"] is True
    assert j["content"]["layers"][0]["title"] == "Calm"


def test_import_fmp_not_found(client, monkeypatch):
    c, _ = client
    from dashboard import fmp_biofield
    monkeypatch.setattr(fmp_biofield, "import_content", lambda *a, **k: None)
    r = c.post("/api/console/biofield-portal/import-fmp?key=test-secret",
               json={"email": "nobody@x.com"})
    assert r.status_code == 200
    assert r.get_json()["found"] is False


def test_import_fmp_requires_key(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal/import-fmp", json={"email": "e@x.com"})
    assert r.status_code == 401
