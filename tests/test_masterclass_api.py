import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    monkeypatch.setattr("dashboard.zoom.get_token", lambda *a, **k: "tok")
    monkeypatch.setattr("dashboard.zoom.create_meeting",
                        lambda *a, **k: {"join_url": "https://zoom.us/j/mc", "meeting_id": "mc1", "start_url": "x"})
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_console_create_requires_auth(client):
    r = client.post("/api/console/masterclass", json={"topic": "T", "start_ts": "2026-07-10T18:00:00"})
    assert r.status_code == 401

def test_console_create_makes_event_and_zoom(client):
    r = client.post("/api/console/masterclass",
                    json={"topic": "Terrain 101", "description": "d", "start_ts": "2026-07-10T18:00:00",
                          "duration_min": 60, "price_cents": 5000, "member_price_cents": 0}, headers=ADMIN)
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["zoom_ok"] is True and "/masterclass/" in d["event_url"]
    import sqlite3
    from dashboard import masterclass as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        ev = mc.get_event(cx, d["event_id"])
        assert ev["zoom_join_url"] == "https://zoom.us/j/mc"
