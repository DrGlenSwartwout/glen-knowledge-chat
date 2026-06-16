"""Tests for the console-gated POST /api/cert/portal-invite route.

Sends a practitioner-portal magic-link email to a cert participant.
"""

import pytest


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    import app as appmod
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


# ── 1. missing email → 400 ────────────────────────────────────────────────────

def test_missing_email(client):
    c, appmod = client
    key = appmod.CONSOLE_SECRET or ""
    r = c.post("/api/cert/portal-invite?key=" + key, json={})
    assert r.status_code == 400
    assert r.get_json()["error"] == "email required"


# ── 2. unknown email (no practitioner record) → 200 {ok:true, sent:false} ────

def test_unknown_email(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod._pp, "id_for_email", lambda email: None)
    key = appmod.CONSOLE_SECRET or ""
    r = c.post("/api/cert/portal-invite?key=" + key,
               json={"email": "nobody@example.com"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["sent"] is False


# ── 3. known email → 200 {ok:true, sent:true}, mailer called with magic URL ──

def test_known_email_sends_link(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod._pp, "id_for_email", lambda email: "pid1")
    monkeypatch.setattr(appmod._pp, "name_for_email", lambda email: "Dr X")
    monkeypatch.setattr(appmod._pp, "create_magic_link_token", lambda pid, email: "tok")

    calls = []

    def _fake_send(to_email, name, magic_url):
        calls.append((to_email, name, magic_url))
        return ("smtp", None)

    monkeypatch.setattr(appmod, "_send_practitioner_magic_link", _fake_send)

    key = appmod.CONSOLE_SECRET or ""
    r = c.post("/api/cert/portal-invite?key=" + key,
               json={"email": "dr@example.com"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["sent"] is True
    assert body["email"] == "dr@example.com"

    assert len(calls) == 1
    _to, _name, magic_url = calls[0]
    assert "/practitioner/login-verify?token=tok" in magic_url


# ── 4. unauthorized (no console key, when CONSOLE_SECRET set) → 401 ──────────

def test_unauthorized(client):
    c, appmod = client
    if not appmod.CONSOLE_SECRET:
        pytest.skip("CONSOLE_SECRET not set")
    r = c.post("/api/cert/portal-invite", json={"email": "dr@example.com"})
    assert r.status_code == 401
