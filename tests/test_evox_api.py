# tests/test_evox_api.py  (needs doppler — imports app)
import os, sqlite3, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod
from dashboard import evox as _evmod

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


def test_book_slot_taken_preserves_credit(client, monkeypatch):
    email = "credit@x.com"
    tok = _start(client, email=email)
    for item in ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok"):
        client.post(f"/api/evox/readiness?token={tok}", json={"item": item, "value": True})
    slots = client.get(f"/api/evox/availability?token={tok}&range=week").get_json()["slots"]
    assert slots, "expected at least one open slot in Mon-Thu window"
    slot = slots[0]

    with sqlite3.connect(appmod.LOG_DB) as cx:
        _evmod.add_session_credits(cx, email, 1)
        cx.execute(
            "INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
            "prepaid,ics_uid,created_at) VALUES (?,?,?,?,'booked',0,?,?)",
            ("other@x.com", "rae", slot,
             slot, "evox-conflict@illtowell.com", "2026-01-01T00:00:00+00:00"))
        cx.commit()

    # Blind the route's server-side re-validation to the conflict so execution
    # reaches create_booking, which relies on the UNIQUE index (not booked_starts).
    monkeypatch.setattr(_evmod, "booked_starts", lambda cx, practitioner="rae": set())

    r = client.post(f"/api/evox/book?token={tok}", json={"start_ts": slot})
    assert r.status_code == 409
    assert r.get_json()["error"] == "slot_taken"

    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert _evmod.session_credit_balance(cx, email) == 1


# ── Task 8: confirmation emails (client + Rae) with ICS ────────────────────────
def test_confirmations_send_client_and_rae(monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda to, name, subj, html, text, ics: calls.append((to, ics)))
    monkeypatch.setattr(appmod, "EVOX_RAE_PHONE", "808-555-1212")
    monkeypatch.setattr(appmod, "EVOX_RAE_EMAIL", "rae@illtowell.com", raising=False)
    appmod._evox_send_confirmations("c@x.com", {
        "id": 1, "email": "c@x.com", "start_ts": "2026-07-06T11:00:00",
        "end_ts": "2026-07-06T12:00:00", "ics_uid": "u1@illtowell.com", "prepaid": False})
    assert len(calls) == 2
    tos = {c[0] for c in calls}
    assert "c@x.com" in tos and "rae@illtowell.com" in tos
    assert all(b"BEGIN:VCALENDAR" in c[1] for c in calls)



# ── Task 9: static/evox.html page served at GET /evox ──────────────────────────
def test_evox_page_served(client):
    r = client.get("/evox")
    assert r.status_code == 200
    assert b"EVOX Setup" in r.data


def test_book_route_sends_client_and_rae_confirmations(client, monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda to, name, subj, html, text, ics: calls.append((to, ics)))
    monkeypatch.setattr(appmod, "EVOX_RAE_EMAIL", "rae@illtowell.com", raising=False)
    tok = _start(client)
    for item in ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok"):
        client.post(f"/api/evox/readiness?token={tok}", json={"item": item, "value": True})
    slots = client.get(f"/api/evox/availability?token={tok}&range=week").get_json()["slots"]
    assert slots, "expected at least one open slot in Mon-Thu window"
    r = client.post(f"/api/evox/book?token={tok}", json={"start_ts": slots[0]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert len(calls) == 2
    tos = {c[0] for c in calls}
    assert "c@x.com" in tos and "rae@illtowell.com" in tos
    assert all(b"BEGIN:VCALENDAR" in c[1] for c in calls)
