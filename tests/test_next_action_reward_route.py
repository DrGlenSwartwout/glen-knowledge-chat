import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setenv("DATA_SHARING_REWARD_ENABLED", "1")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_next_actions_route_ok_with_reward_flag_and_fresh_db(client):
    """Regression: member_reward_grants must be defensively init'd in the route,
    same as every other lister's table, else a fresh/monkeypatched LOG_DB 500s
    the ENTIRE /api/console/next-actions queue (not just the reward item)."""
    c, appmod = client
    # Sanity: table genuinely does not exist yet in this fresh DB.
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = cx.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='member_reward_grants'").fetchone()
        assert row is None

    r = c.get("/api/console/next-actions?key=test-secret")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert "items" in body
