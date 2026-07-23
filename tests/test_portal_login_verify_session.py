# tests/test_portal_login_verify_session.py
"""client_login_verify must fall back to a valid rm_portal_session cookie when
the magic-link token is dead (already consumed / expired) instead of always
redirecting to /portal/login?error=link -- a valid session is not a dead link.
See dashboard/portal_identity.identity_from_session, the same check that
authenticates /portal/me and the portal APIs."""
import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "_client_login_enabled", lambda: True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_person(appmod, email, name="C", roles='["client"]'):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(appmod.LOG_DB)
    pi._ensure_people_table(cx)
    cx.execute(
        "INSERT OR IGNORE INTO people (email, name, roles, created_at, updated_at) "
        "VALUES (?,?,?,?,?)", (email, name, roles, "t", "t"))
    cx.commit()
    cx.close()


def _dead_token_and_session(appmod, email="ss@example.com"):
    """Seed a person, mint a session, mint+consume a magic link (dead token).
    Returns (dead_token, session_token)."""
    from dashboard import portal_identity as pi
    _seed_person(appmod, email, "SS")
    cx = sqlite3.connect(appmod.LOG_DB)
    pid = cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]
    session = pi.create_client_session(cx, pid, email)
    magic = pi.create_client_magic_link(cx, pid, email)
    cx.commit()
    consumed = pi.consume_client_magic_link(cx, magic)
    cx.commit()
    cx.close()
    assert consumed == pid  # sanity: the link really was live once
    return magic, session


def test_post_dead_token_with_valid_session_falls_back_to_portal_me(client):
    c, appmod = client
    dead_token, session = _dead_token_and_session(appmod)
    c.set_cookie("rm_portal_session", session)
    r = c.post("/portal/login-verify", data={"token": dead_token},
               follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/portal/me")


def test_post_dead_token_without_session_still_errors(client):
    c, appmod = client
    dead_token, _session = _dead_token_and_session(appmod, "ss2@example.com")
    r = c.post("/portal/login-verify", data={"token": dead_token},
               follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["Location"] == "/portal/login?error=link"


def test_get_dead_token_with_valid_session_falls_back_to_portal_me(client):
    c, appmod = client
    dead_token, session = _dead_token_and_session(appmod, "ss3@example.com")
    c.set_cookie("rm_portal_session", session)
    r = c.get(f"/portal/login-verify?token={dead_token}", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/portal/me")


def test_get_live_token_confirm_page_is_not_cached(client):
    c, appmod = client
    from dashboard import portal_identity as pi
    _seed_person(appmod, "ss4@example.com", "SS4")
    cx = sqlite3.connect(appmod.LOG_DB)
    pid = cx.execute("SELECT id FROM people WHERE email=?",
                      ("ss4@example.com",)).fetchone()[0]
    live_token = pi.create_client_magic_link(cx, pid, "ss4@example.com")
    cx.commit()
    cx.close()

    r = c.get(f"/portal/login-verify?token={live_token}")
    assert r.status_code == 200
    assert r.headers.get("Cache-Control") == "no-store"
