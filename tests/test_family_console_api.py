import sqlite3
import app as appmod


def _client(tmp_db, monkeypatch):
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "K")
    return appmod.app.test_client()


def test_console_family_crud(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    h = {"X-Console-Key": "K"}
    assert c.post("/api/console/family", json={"primary_email": "p@x.com", "member_email": "p@x.com",
                  "member_label": "P", "member_type": "human", "display_order": 0}, headers=h).status_code == 200
    c.post("/api/console/family", json={"primary_email": "p@x.com", "member_email": "sasha@f.com",
           "member_label": "Sasha", "member_type": "pet", "display_order": 1}, headers=h)
    got = c.get("/api/console/family-members/p@x.com", headers=h).get_json()
    assert [m["member_email"] for m in got["members"]] == ["p@x.com", "sasha@f.com"]
    c.delete("/api/console/family", json={"primary_email": "p@x.com", "member_email": "sasha@f.com"}, headers=h)
    assert len(c.get("/api/console/family-members/p@x.com", headers=h).get_json()["members"]) == 1


def test_console_family_requires_key(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    assert c.get("/api/console/family-members/p@x.com").status_code == 403
