# tests/test_evox_api.py  (needs doppler — imports app)
import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

def test_send_evox_email_builds_mixed_with_ics(monkeypatch):
    captured = {}
    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def sendmail(self, frm, to, msg): captured["msg"] = msg
    monkeypatch.setattr(appmod, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(appmod, "SMTP_USER", "u"); monkeypatch.setattr(appmod, "SMTP_PASS", "p")
    monkeypatch.setattr(appmod.smtplib, "SMTP", FakeSMTP)
    mode, err = appmod.send_evox_email("c@x.com", "C", "EVOX confirmed",
                                       "<p>hi</p>", "hi", b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n")
    assert mode == "smtp" and err is None
    assert "text/calendar" in captured["msg"] and "multipart/mixed" in captured["msg"]


# ── Route tests (Task 7): start / state / readiness / availability / book ──────
@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "EVOX_HOURS", "1-4:09:00-16:00")
    # neutralize outbound email
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

def _start(client, email="c@x.com"):
    r = client.post("/api/evox/start", json={"email": email, "name": "C"})
    return r.get_json()["token"]

def test_availability_blocked_until_ready(client):
    tok = _start(client)
    r = client.get(f"/api/evox/availability?token={tok}&range=week")
    assert r.status_code == 403 and r.get_json()["error"] == "not_ready"

def test_full_flow_book(client, monkeypatch):
    tok = _start(client)
    for item in ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok"):
        client.post(f"/api/evox/readiness?token={tok}", json={"item": item, "value": True})
    slots = client.get(f"/api/evox/availability?token={tok}&range=week").get_json()["slots"]
    assert slots, "expected at least one open slot in Mon-Thu window"
    r = client.post(f"/api/evox/book?token={tok}", json={"start_ts": slots[0]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    # second identical booking -> slot taken
    r2 = client.post(f"/api/evox/book?token={tok}", json={"start_ts": slots[0]})
    assert r2.status_code == 409
