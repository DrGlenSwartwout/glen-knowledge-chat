"""_role_for_token maps an access token -> rbac role, so BOS actions accept
scoped tokens (Rae = owner). Shaira stays VA, bound by the policy matrix."""
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


def test_role_for_token_rae_is_owner(appmod):
    _seed(appmod, "rae-tok", "rae")
    assert appmod._role_for_token("rae-tok") == "owner"


def test_role_for_token_shaira_is_va(appmod):
    _seed(appmod, "sha-tok", "shaira")
    assert appmod._role_for_token("sha-tok") == "va"


def test_role_for_token_unknown_is_none(appmod):
    assert appmod._role_for_token("nope") is None
