"""Console browser-login via signed cookie, so the master console key never has
to ride in the URL bar (address bar / history / referer / server logs).

Covers:
  - a browser ?key= on a /console page → 302 that strips the key and sets the cookie
  - the cookie then authenticates a gated route with NO key in the URL
  - script paths (header, ?key= on /api/console) are unchanged
  - /api/console ?key= is NOT redirected (scripts don't follow redirects)
  - rotating CONSOLE_SECRET invalidates every previously issued cookie
"""
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    # use_cookies=True (default) → the client's jar carries Set-Cookie forward
    return appmod.app.test_client(), appmod


def _set_cookie_header(resp, name):
    for h in resp.headers.getlist("Set-Cookie"):
        if h.startswith(name + "="):
            return h
    return None


def test_browser_key_redirects_strips_and_sets_cookie(client):
    c, _ = client
    r = c.get("/console/pages?key=test-secret", headers={"Accept": "text/html"})
    assert r.status_code == 302
    assert "key=" not in r.headers["Location"]
    assert r.headers["Location"].endswith("/console/pages")
    sc = _set_cookie_header(r, "rm_console_auth")
    assert sc and "HttpOnly" in sc


def test_authenticated_console_request_refreshes_cookie(client):
    """Rolling expiry: an authenticated console request (cookie, no ?key=) re-issues
    the cookie so the 12h clock resets on activity."""
    c, _ = client
    c.get("/console/pages?key=test-secret", headers={"Accept": "text/html"})  # jar holds cookie
    r = c.get("/api/console/next-actions")  # cookie-authenticated, no key
    assert r.status_code == 200
    assert _set_cookie_header(r, "rm_console_auth") is not None


def test_anonymous_request_is_not_refreshed(client):
    c, _ = client
    r = c.get("/api/console/next-actions")  # no cookie, no key → 401
    assert r.status_code == 401
    assert _set_cookie_header(r, "rm_console_auth") is None


def test_non_console_path_is_not_refreshed(client):
    c, _ = client
    c.get("/console/pages?key=test-secret", headers={"Accept": "text/html"})  # jar holds cookie
    r = c.get("/some-non-console-path-xyz")  # outside console/admin → no refresh header
    assert _set_cookie_header(r, "rm_console_auth") is None


def test_extra_query_params_survive_the_strip(client):
    c, _ = client
    r = c.get("/console/pages?tab=drafts&key=test-secret",
              headers={"Accept": "text/html"})
    assert r.status_code == 302
    loc = r.headers["Location"]
    assert "key=" not in loc
    assert "tab=drafts" in loc


def test_cookie_authenticates_without_key(client):
    c, _ = client
    # Log in (302 sets the cookie in the client's jar)
    r = c.get("/console/pages?key=test-secret", headers={"Accept": "text/html"})
    assert r.status_code == 302
    # A gated API route with NO key in the URL is now authorized via the cookie
    r2 = c.get("/api/console/next-actions")
    assert r2.status_code == 200


def test_query_key_on_api_still_works(client):
    c, _ = client
    r = c.get("/api/console/next-actions?key=test-secret")
    assert r.status_code == 200


def test_header_key_still_works(client):
    c, _ = client
    r = c.get("/api/console/next-actions",
              headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200


def test_api_key_is_not_redirected(client):
    # A script hitting /api/console with ?key= must be served, not 302'd —
    # scripts don't follow redirects and would otherwise break.
    c, _ = client
    r = c.get("/api/console/next-actions?key=test-secret",
              headers={"Accept": "text/html"})
    assert r.status_code == 200


def test_no_cookie_no_key_is_unauthorized(client):
    c, _ = client
    r = c.get("/api/console/next-actions")
    assert r.status_code == 401


def test_invalid_key_sets_no_cookie(client):
    c, _ = client
    r = c.get("/console/pages?key=wrong", headers={"Accept": "text/html"})
    assert _set_cookie_header(r, "rm_console_auth") is None


def test_rotation_invalidates_old_cookie(monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "old-secret")
    old_cookie = appmod._console_cookie_value()
    assert appmod._console_cookie_valid(old_cookie)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "new-secret")
    assert not appmod._console_cookie_valid(old_cookie)


def test_dashboard_require_console_key_accepts_cookie(monkeypatch):
    """The legacy dashboard.require_console_key gate (used by several /admin/*
    data endpoints) must honor the login cookie once app.py registers the hook,
    so those same-origin JS fetches keep working after the key is stripped."""
    import app as appmod
    import dashboard
    from flask import request
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "test-secret")

    @dashboard.require_console_key
    def protected():
        return "ok", 200

    cookie = appmod._console_cookie_value()
    # No key anywhere, only the cookie → authorized
    with appmod.app.test_request_context(
            "/admin/x", headers={"Cookie": appmod.CONSOLE_COOKIE + "=" + cookie}):
        body, status = protected()
        assert status == 200 and body == "ok"
    # No key and no cookie → 401
    with appmod.app.test_request_context("/admin/x"):
        resp = protected()
        assert resp[1] == 401


def test_no_console_secret_means_no_valid_cookie(monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "")
    assert appmod._console_cookie_value() == ""
    assert not appmod._console_cookie_valid("")
    assert not appmod._console_cookie_valid("anything")


# ── Owner-token cookie login (Rae): same clean URL-strip as the master secret,
#    but the cookie carries the revocable token and never escalates to master. ──

_RAE = "rae-owner-token-abc123"


def _seed_owner_token(appmod, token=_RAE, scope="workspace:rae"):
    appmod._init_workspace_schema()
    with appmod.db.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "INSERT INTO workspace_users (name, display_name, scope) VALUES ('rae','Rae',?) "
            "ON CONFLICT(name) DO UPDATE SET scope=excluded.scope", (scope,))
        uid = cx.execute("SELECT id FROM workspace_users WHERE name='rae'").fetchone()[0]
        cx.execute("INSERT INTO access_tokens (token, user_id) VALUES (?,?)", (token, uid))
        cx.commit()


def test_owner_token_precondition(client):
    # Sanity: the seeded token resolves to the OWNER role (else the rest is moot).
    c, appmod = client
    _seed_owner_token(appmod)
    assert appmod._owner_token_ok(_RAE) is True


def test_owner_token_browser_login_strips_and_sets_cookie(client):
    c, appmod = client
    _seed_owner_token(appmod)
    r = c.get("/console/pages?key=" + _RAE, headers={"Accept": "text/html"})
    assert r.status_code == 302
    assert "key=" not in r.headers["Location"]
    sc = _set_cookie_header(r, "rm_console_auth")
    assert sc and "HttpOnly" in sc
    # cookie carries the TOKEN itself (not the master HMAC)
    assert _RAE in sc


def test_owner_cookie_authenticates_without_key(client):
    c, appmod = client
    _seed_owner_token(appmod)
    c.get("/console/pages?key=" + _RAE, headers={"Accept": "text/html"})  # jar holds cookie
    r = c.get("/api/console/next-actions")  # no key in URL — cookie authenticates
    assert r.status_code == 200


def test_owner_cookie_never_escalates_to_master(client):
    # An owner-token cookie must resolve to the token, never the master secret.
    c, appmod = client
    _seed_owner_token(appmod)
    from flask import request
    with appmod.app.test_request_context(
            "/console", headers={"Cookie": "rm_console_auth=" + _RAE}):
        assert appmod._present_console_key() == _RAE
        assert appmod._present_console_key() != appmod.CONSOLE_SECRET


def test_owner_cookie_is_refreshed_rolling(client):
    c, appmod = client
    _seed_owner_token(appmod)
    c.get("/console/pages?key=" + _RAE, headers={"Accept": "text/html"})
    r = c.get("/api/console/next-actions")
    assert r.status_code == 200
    assert _set_cookie_header(r, "rm_console_auth") is not None


def test_revoked_owner_token_cookie_is_rejected(client):
    c, appmod = client
    _seed_owner_token(appmod)
    with appmod.db.connect(appmod.LOG_DB) as cx:
        cx.execute("UPDATE access_tokens SET revoked_at=datetime('now') WHERE token=?", (_RAE,))
        cx.commit()
    from flask import request
    with appmod.app.test_request_context(
            "/console", headers={"Cookie": "rm_console_auth=" + _RAE}):
        assert appmod._present_console_key() == ""


def test_scoped_va_token_is_not_cookie_logged_in(client):
    # A VA-scoped token (Shaira) is not an OWNER token → no strip, no cookie.
    c, appmod = client
    _seed_owner_token(appmod, token="va-token-xyz", scope="workspace:shaira")
    r = c.get("/console/pages?key=va-token-xyz", headers={"Accept": "text/html"})
    assert _set_cookie_header(r, "rm_console_auth") is None
