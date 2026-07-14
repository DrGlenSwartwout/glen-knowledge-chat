import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_next_actions_route_returns_items(client):
    c, appmod = client
    from dashboard import biofield_reveals, ff_match_drafts
    with sqlite3.connect(appmod.LOG_DB) as cx:
        biofield_reveals.init_table(cx)
        ff_match_drafts.init_table(cx)
        cx.execute(
            "INSERT INTO ff_match_drafts (email,scan_date,items_json,status,"
            "created_at,updated_at) VALUES ('f@b.co','s','[]','draft','t','t')")
        cx.commit()

    r = c.get("/api/console/next-actions?key=test-secret")
    assert r.status_code == 200
    body = r.get_json()
    assert "items" in body
    assert any(it["label"] == "Publish" for it in body["items"])


def test_next_actions_route_requires_console_key(client):
    c, _ = client
    r = c.get("/api/console/next-actions")
    assert r.status_code == 401
