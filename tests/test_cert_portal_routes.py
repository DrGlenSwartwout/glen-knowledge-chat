# tests/test_cert_portal_routes.py
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CERT_PORTAL_ENABLED", "true")
    import app as appmod
    # Hermetic sqlite: point LOG_DB at a tmp file so tests never touch the dev db.
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _mint_cert_token(appmod, email):
    """Insert a cert_portal auth token directly and return the raw token."""
    import secrets
    from datetime import timedelta
    tok = secrets.token_urlsafe(16)
    now = appmod._now_utc()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT PRIMARY KEY, "
                   "email TEXT, purpose TEXT NOT NULL, extra TEXT, created_at TEXT NOT NULL, "
                   "expires_at TEXT NOT NULL, consumed_at TEXT)")
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
            "VALUES (?,?,?,?,?)",
            (appmod._hash_token(tok), email, "cert_portal", now.isoformat(),
             (now + timedelta(minutes=appmod.AUTH_TOKEN_TTL_MIN)).isoformat()))
        cx.commit()
    return tok


def test_login_always_200(client):
    c, _ = client
    r = c.post("/cert/login", json={"email": "doc@x.com"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_auth_sets_cookie_and_redirects(client):
    c, appmod = client
    tok = _mint_cert_token(appmod, "doc@x.com")
    r = c.get(f"/cert/auth/{tok}")
    assert r.status_code == 302
    assert "rm_cert_email" in r.headers.get("Set-Cookie", "")


def test_auth_rejects_bad_token(client):
    c, _ = client
    r = c.get("/cert/auth/not-a-real-token")
    assert r.status_code == 400


def test_portal_page_served_when_enabled(client):
    c, _ = client
    r = c.get("/cert")
    assert r.status_code == 200


def test_portal_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    # The flag is read live (not at import), so force it off via the helper.
    monkeypatch.setattr(appmod, "_cert_portal_enabled", lambda: False)
    r = c.get("/cert")
    assert r.status_code == 404
