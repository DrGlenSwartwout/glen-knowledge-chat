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


def test_consult_confirmations_include_join_link(monkeypatch):
    calls = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda to, name, subj, html, text, ics: calls.append((to, html, ics)), raising=False)
    monkeypatch.setattr(appmod, "GLEN_CONSULT_EMAIL", "glen@illtowell.com", raising=False)
    appmod._consult_send_confirmations("c@x.com", {
        "id": 1, "email": "c@x.com", "start_ts": "2026-07-06T13:00:00",
        "end_ts": "2026-07-06T13:30:00", "ics_uid": "u1@illtowell.com",
        "join_url": "https://zoom.us/j/9", "session_type": "biofield-consult", "medium": "video"})
    assert len(calls) == 2
    assert any("zoom.us/j/9" in c[1] for c in calls)          # client email carries the link
    assert all(b"BEGIN:VCALENDAR" in c[2] for c in calls)


def test_portal_view_carries_consult_block(client):
    tok = _mk_portal("pv@x.com")
    client.post("/api/console/consult-ready", json={"email": "pv@x.com", "ready": True}, headers=ADMIN)
    d = client.get(f"/api/portal/{tok}/view").get_json()
    assert "consult" in d and d["consult"]["ready"] is True
    assert "stages" in d["consult"]


def test_run_reminders_uses_consult_copy_for_consult_bookings(client, monkeypatch):
    import sqlite3
    from datetime import timedelta
    from dashboard import evox
    captured = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda to, name, subj, html, text, ics: captured.append((to, subj, html)),
                        raising=False)
    start = (appmod._hst_now() + timedelta(hours=30)).replace(microsecond=0).isoformat()
    end = (appmod._hst_now() + timedelta(hours=30, minutes=30)).replace(microsecond=0).isoformat()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        evox.init_evox_tables(cx)
        cx.execute("INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
                   "session_type,medium,zoom_join_url) VALUES (?,?,?,?, 'booked', ?,?,?)",
                   ("consult@x.com", "glen", start, end, "biofield-consult", "video",
                    "https://zoom.us/j/consulttest"))
        cx.commit()
    r = client.post("/api/evox/run-reminders", headers=ADMIN)
    assert r.status_code == 200 and r.get_json()["sent"] == 1
    to, subj, html = captured[-1]
    assert to == "consult@x.com"
    assert subj == "Reminder: your Biofield Consult tomorrow"
    assert "zoom.us/j/consulttest" in html and "call Rae" not in html
