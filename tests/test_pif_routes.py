import sqlite3
import app as appmod
import dashboard
from dashboard import points


def _client(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db)
    points.init_points_table(cx)
    cx.close()
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "testsecret")
    return appmod.app.test_client()


def test_milestone_route_requires_console_key(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/admin/pif/milestone", json={"email": "m@x.com", "milestone_key": "k"})
    assert r.status_code == 401


def test_milestone_route_credits(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/admin/pif/milestone?key=testsecret",
               json={"email": "m@x.com", "milestone_key": "program_complete_1"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["balance_cents"] == 500


def test_milestone_route_validates_input(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/admin/pif/milestone?key=testsecret", json={"email": "m@x.com"})
    assert r.status_code == 400
