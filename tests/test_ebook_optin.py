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
    # C1: the response must carry NO token and NO portal_url — the portal link
    # is delivered ONLY by email (proving ownership), never echoed back to an
    # unauthenticated caller who merely typed an email address.
    assert r.get_json() == {"ok": True}
    body_text = r.get_data(as_text=True).lower()
    assert "portal" not in body_text
    assert "token" not in body_text
    from dashboard import portal_library as lib, client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    assert lib.has(cx, "lead@x.com", "healing-glaucoma-starter") is True
    assert cx.execute("SELECT COUNT(*) FROM client_portals WHERE email='lead@x.com'").fetchone()[0] == 1
    # I1: the dedicated library-ready email path fired — its once-guard row is
    # inserted synchronously (before the threaded send), so it's visible here
    # even though the actual network send happens on a background thread.
    row = cx.execute(
        "SELECT 1 FROM ebook_library_welcome_sent WHERE email=? AND ebook_slug=?",
        ("lead@x.com", "healing-glaucoma-starter")).fetchone()
    assert row is not None

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
