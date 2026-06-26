"""GET /api/me — resolves the caller's key/token to {role, nav} for op-nav."""
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod.dashboard, "CONSOLE_SECRET", "test-secret")
    appmod._init_auth_tables()
    appmod._init_workspace_schema()
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_token(appmod, token, owner):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO workspace_users (name, display_name, scope) VALUES (?,?,?)",
                   (owner, owner.title(), f"workspace:{owner}"))
        uid = cx.execute("SELECT id FROM workspace_users WHERE name=?", (owner,)).fetchone()[0]
        cx.execute("INSERT INTO access_tokens (token, user_id, note) VALUES (?,?,?)",
                   (token, uid, "test"))
        cx.commit()


def test_me_admin_is_glen(client):
    c, _ = client
    j = c.get("/api/me", headers={"X-Console-Key": "test-secret"}).get_json()
    assert j["role"] == "owner" and j["nav"] == "glen" and j["name"] == "Glen"


def test_me_rae_token_is_owner_nav_rae(client):
    c, appmod = client
    _seed_token(appmod, "rae-tok", "rae")
    j = c.get("/api/me", headers={"X-Console-Key": "rae-tok"}).get_json()
    assert j["role"] == "owner" and j["nav"] == "rae" and j["name"] == "Rae"


def test_me_shaira_token_is_va(client):
    c, appmod = client
    _seed_token(appmod, "sha-tok", "shaira")
    j = c.get("/api/me", headers={"X-Console-Key": "sha-tok"}).get_json()
    assert j["role"] == "va" and j["nav"] == "va"


def test_me_no_key_is_all_null_200(client):
    c, _ = client
    r = c.get("/api/me")
    assert r.status_code == 200
    j = r.get_json()
    assert j["role"] is None and j["nav"] is None
