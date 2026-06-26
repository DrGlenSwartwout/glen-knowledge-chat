"""require_console_key accepts an OWNER-role per-user token (Rae) in addition to
CONSOLE_SECRET, so the console's read endpoints work for her token. A VA token
(Shaira) and a bad key are still rejected. Importing app registers the real
owner-token check via dashboard.set_owner_token_check()."""
import sqlite3
import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod.dashboard, "CONSOLE_SECRET", "test-secret")
    appmod._init_auth_tables()
    appmod._init_workspace_schema()
    return appmod


def _seed(appmod, token, owner):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO workspace_users (name, display_name, scope) VALUES (?,?,?)",
                   (owner, owner.title(), f"workspace:{owner}"))
        uid = cx.execute("SELECT id FROM workspace_users WHERE name=?", (owner,)).fetchone()[0]
        cx.execute("INSERT INTO access_tokens (token, user_id, note) VALUES (?,?,?)",
                   (token, uid, "t"))
        cx.commit()


# A throwaway endpoint wrapped by the real decorator.
def _protected():
    import dashboard
    @dashboard.require_console_key
    def _inner():
        return ("OK", 200)
    return _inner()


def test_console_secret_passes(appmod):
    with appmod.app.test_request_context(headers={"X-Console-Key": "test-secret"}):
        assert _protected() == ("OK", 200)


def test_owner_token_passes(appmod):
    _seed(appmod, "rae-tok", "rae")
    with appmod.app.test_request_context(headers={"X-Console-Key": "rae-tok"}):
        assert _protected() == ("OK", 200)


def test_va_token_rejected(appmod):
    _seed(appmod, "sha-tok", "shaira")
    with appmod.app.test_request_context(headers={"X-Console-Key": "sha-tok"}):
        body, code = _protected()
        assert code == 401


def test_bad_key_rejected(appmod):
    with appmod.app.test_request_context(headers={"X-Console-Key": "nope"}):
        body, code = _protected()
        assert code == 401


# _owner_token_ok backs the ~57 inline `key != CONSOLE_SECRET` gates that don't
# use the decorator. It must accept an OWNER token and reject a VA token / bad key.
def test_owner_token_ok_helper(appmod):
    _seed(appmod, "rae-tok", "rae")
    _seed(appmod, "sha-tok", "shaira")
    assert appmod._owner_token_ok("rae-tok") is True
    assert appmod._owner_token_ok("sha-tok") is False
    assert appmod._owner_token_ok("nope") is False
    assert appmod._owner_token_ok("") is False
