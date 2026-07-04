import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "GLEN_CONSULT_HOURS", "1-7:09:00-17:00", raising=False)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_consult_ready_flip_requires_auth(client):
    r = client.post("/api/console/consult-ready", json={"email": "c@x.com", "ready": True})
    assert r.status_code == 401

def test_consult_ready_flip_sets_flag(client):
    r = client.post("/api/console/consult-ready",
                    json={"email": "c@x.com", "ready": True}, headers=ADMIN)
    assert r.status_code == 200 and r.get_json()["ready"] is True
    import sqlite3
    from dashboard import consult
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert consult.consult_is_ready(cx, "c@x.com") is True


def _mk_portal(email="p@x.com"):
    import sqlite3
    from dashboard import client_portal as cp
    # _init_people_table() normally runs once at app-module import time (against
    # whatever LOG_DB was live then); tests swap LOG_DB to a fresh tmp file per
    # test, so re-run it here against the swapped path — same fix the EVOX test
    # suite gets for free by going through /api/evox/start first.
    appmod._init_people_table()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx)
        token, _ = cp.upsert_portal(cx, email, "P", {"source": "test"})
    return token

def test_availability_blocked_until_ready(client, monkeypatch):
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    tok = _mk_portal("p1@x.com")
    r = client.get(f"/api/consult/availability?token={tok}&range=week")
    assert r.status_code == 403 and r.get_json()["error"] == "not_ready"

def test_full_consult_flow(client, monkeypatch):
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    monkeypatch.setattr("dashboard.zoom.get_token", lambda *a, **k: "tok")
    monkeypatch.setattr("dashboard.zoom.create_meeting",
                        lambda *a, **k: {"join_url": "https://zoom.us/j/1", "meeting_id": "1", "start_url": "x"})
    tok = _mk_portal("p2@x.com")
    client.post("/api/console/consult-ready", json={"email": "p2@x.com", "ready": True}, headers=ADMIN)
    slots = client.get(f"/api/consult/availability?token={tok}&range=week").get_json()["slots"]
    assert slots
    r = client.post(f"/api/consult/book?token={tok}", json={"start_ts": slots[0]})
    assert r.status_code == 200 and r.get_json()["join_url"] == "https://zoom.us/j/1"
    r2 = client.post(f"/api/consult/book?token={tok}", json={"start_ts": slots[0]})
    assert r2.status_code == 409


def test_consult_slot_taken_via_race(client, monkeypatch):
    # The full-flow test's second booking of the SAME slot hits the availability
    # re-check (slot_unavailable) before ever reaching create_booking, so the
    # SlotTaken/409 handler is never actually exercised. Force execution past
    # the re-validation by faking booked_starts, so create_booking's own
    # UNIQUE-index race check is what fires.
    import sqlite3
    from dashboard import evox as evox_mod

    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    monkeypatch.setattr("dashboard.zoom.get_token", lambda *a, **k: "tok")
    monkeypatch.setattr("dashboard.zoom.create_meeting",
                        lambda *a, **k: {"join_url": "https://zoom.us/j/1", "meeting_id": "1", "start_url": "x"})

    tok = _mk_portal("race@x.com")
    client.post("/api/console/consult-ready", json={"email": "race@x.com", "ready": True}, headers=ADMIN)
    slots = client.get(f"/api/consult/availability?token={tok}&range=week").get_json()["slots"]
    assert slots
    slot = slots[0]

    # Pre-insert a conflicting active booking for the same practitioner/slot,
    # simulating another request winning the race between the availability
    # re-check and create_booking's INSERT.
    with sqlite3.connect(appmod.LOG_DB) as cx:
        evox_mod.init_evox_tables(cx)
        end_ts = slot  # value irrelevant to this test
        cx.execute(
            "INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
            "created_at,session_type,medium) "
            "VALUES (?,?,?,?,'booked',?,?,?)",
            ("other@x.com", "glen", slot, end_ts, "2026-01-01T00:00:00", "biofield-consult", "video"))
        cx.commit()

    # Make the route's re-validation pass (as if the slot still looked free)
    # so it proceeds into create_booking, whose UNIQUE index then raises SlotTaken.
    monkeypatch.setattr(evox_mod, "booked_starts", lambda cx, practitioner="rae": set())

    r = client.post(f"/api/consult/book?token={tok}", json={"start_ts": slot})
    assert r.status_code == 409
    assert r.get_json()["error"] == "slot_taken"
