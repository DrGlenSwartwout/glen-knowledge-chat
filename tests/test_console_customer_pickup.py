import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_people_table()
    appmod.app.config["TESTING"] = True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("INSERT INTO people (email) VALUES ('d@x.com')")
        cx.commit()
    monkeypatch.setattr(appmod, "_bos_actor",
                        lambda: type("A", (), {"role": appmod._bos_rbac.OWNER})())
    return appmod.app.test_client(), appmod


def test_sets_and_clears_the_flag(client):
    c, appmod = client
    from dashboard import customers as C
    r = c.post("/api/console/customers/pickup", json={"email": "d@x.com", "pickup": True})
    assert r.get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert C.pickup_default_for_email(cx, "d@x.com") is True
    c.post("/api/console/customers/pickup", json={"email": "d@x.com", "pickup": False})
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert C.pickup_default_for_email(cx, "d@x.com") is False


def test_unknown_email_is_400_not_silent_noop(client):
    c, _ = client
    r = c.post("/api/console/customers/pickup", json={"email": "nope@x.com", "pickup": True})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_blank_email_is_400(client):
    c, _ = client
    r = c.post("/api/console/customers/pickup", json={"email": "", "pickup": True})
    assert r.status_code == 400


def test_non_owner_is_401(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_bos_actor", lambda: None)
    r = c.post("/api/console/customers/pickup", json={"email": "d@x.com", "pickup": True})
    assert r.status_code == 401
