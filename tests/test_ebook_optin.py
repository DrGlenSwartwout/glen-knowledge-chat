import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import app as appmod
    importlib.reload(appmod)
    return appmod

def test_optin_provisions_portal_and_grants(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/api/ebook/optin", json={
        "email": "Lead@x.com", "ebook_slug": "healing-glaucoma-starter",
        "source_site": "healingglaucoma.com", "name": "Lee"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["portal_url"].startswith(("http", "/portal/"))
    from dashboard import portal_library as lib, client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert lib.has(cx, "lead@x.com", "healing-glaucoma-starter") is True
    assert cx.execute("SELECT COUNT(*) FROM client_portals WHERE email='lead@x.com'").fetchone()[0] == 1

def test_optin_rejects_bad_email_and_unknown_slug(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    assert c.post("/api/ebook/optin", json={"email": "x", "ebook_slug": "healing-glaucoma-starter"}).status_code == 400
    assert c.post("/api/ebook/optin", json={"email": "a@b.com", "ebook_slug": "nope"}).status_code == 400

def test_optin_sets_cors_and_handles_preflight(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    c = appmod.app.test_client()
    assert c.options("/api/ebook/optin").headers.get("Access-Control-Allow-Origin") == "*"
    r = c.post("/api/ebook/optin", json={"email": "a@b.com", "ebook_slug": "healing-glaucoma-starter"})
    assert r.headers.get("Access-Control-Allow-Origin") == "*"
