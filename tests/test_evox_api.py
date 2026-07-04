# tests/test_evox_api.py  (needs doppler — imports app)
import os, sqlite3, pytest
from datetime import timedelta
from urllib.parse import urlparse, parse_qs
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
    # /api/evox/start no longer returns the portal token — it emails the setup
    # link. Capture that link so _start (and tests) can recover the token.
    setup_links = []
    def _capture_setup_link(to_email, name, setup_url):
        setup_links.append({"to": to_email, "name": name, "url": setup_url})
        return ("console-log", None)
    monkeypatch.setattr(appmod, "send_evox_setup_link", _capture_setup_link)
    appmod.app.config["TESTING"] = True
    c = appmod.app.test_client()
    c.evox_setup_links = setup_links  # tests read the last emailed setup URL here
    return c

def _token_from_setup_url(url):
    return parse_qs(urlparse(url).query).get("token", [""])[0]

def _start(client, email="c@x.com"):
    r = client.post("/api/evox/start", json={"email": email, "name": "C"})
    assert r.get_json().get("ok") is True
    # The token is delivered only via the emailed setup link, never in the body.
    return _token_from_setup_url(client.evox_setup_links[-1]["url"])

def test_start_does_not_leak_portal_token(client):
    """SECURITY regression: /api/evox/start must never return the portal token
    (the client's MASTER portal bearer credential) to the unauthenticated
    caller. It emails the setup link instead."""
    from dashboard import evox as _ev, customers as _cu, client_portal as _cp
    email = "leaktest@x.com"
    # Pre-create the portal so a stable token already exists for this email.
    appmod._init_people_table()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _ev.init_evox_tables(cx)
        _cp.init_client_portal_table(cx)
        _cu.find_or_create_by_email(cx, email=email, name="Leak")
        stored_token = _ev.ensure_portal_token(cx, email, "Leak")
    assert stored_token  # sanity: a real bearer token exists

    r = client.post("/api/evox/start", json={"email": email, "name": "Leak"})
    assert r.status_code == 200
    data = r.get_json()
    # No token, no url — and no value anywhere in the body equal to the token.
    assert "token" not in data
    assert "url" not in data
    assert stored_token not in data.values()
    assert data == {"ok": True, "emailed": True}

    # The setup link WAS emailed and carries the token via /evox?token=…
    link = client.evox_setup_links[-1]
    assert link["to"] == email
    assert "/evox?token=" in link["url"]
    assert _token_from_setup_url(link["url"]) == stored_token


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


# ── Task 10: 24-48h reminder pass (console-gated cron) ─────────────────────────
def test_reminders_send_once_deterministic(client, monkeypatch):
    """Deterministic replacement for the plan's vacuous sample test: proves the
    24-48h window (in-window reminded, out-of-window skipped), idempotency via
    reminded_at, and the console-key auth gate — instead of asserting sent >= 0
    against a nondeterministic real-calendar slot."""
    sent = []
    monkeypatch.setattr(appmod, "send_evox_email",
                        lambda *a, **k: sent.append(a[0]) or ("console-log", None))
    assert appmod.CONSOLE_SECRET == "test-secret"  # confirm the client fixture sets this

    now = appmod._hst_now()
    in_window_ts = (now + timedelta(hours=30)).isoformat()   # inside 24-48h -> reminded
    out_window_ts = (now + timedelta(hours=2)).isoformat()   # outside window -> skipped

    with sqlite3.connect(appmod.LOG_DB) as cx:
        _evmod.init_evox_tables(cx)  # creates evox_bookings before we insert directly
        cx.execute(
            "INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
            "prepaid,ics_uid,created_at) VALUES (?,?,?,?,'booked',0,?,?)",
            ("r@x.com", "rae", in_window_ts, in_window_ts,
             "evox-inwindow@illtowell.com", now.isoformat()))
        cx.execute(
            "INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
            "prepaid,ics_uid,created_at) VALUES (?,?,?,?,'booked',0,?,?)",
            ("control@x.com", "rae", out_window_ts, out_window_ts,
             "evox-outwindow@illtowell.com", now.isoformat()))
        cx.commit()

    hdr = {"X-Console-Key": "test-secret"}

    # No / wrong console key -> 401, nothing sent.
    r_noauth = client.post("/api/evox/run-reminders")
    assert r_noauth.status_code == 401
    r_wrongauth = client.post("/api/evox/run-reminders", headers={"X-Console-Key": "nope"})
    assert r_wrongauth.status_code == 401
    assert sent == []

    # First run: only the in-window booking is reminded; the out-of-window
    # control is left alone.
    r1 = client.post("/api/evox/run-reminders", headers=hdr).get_json()
    assert r1["sent"] == 1
    assert sent == ["r@x.com"]

    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = {row["email"]: row["reminded_at"]
                for row in cx.execute("SELECT email, reminded_at FROM evox_bookings")}
        assert rows["r@x.com"] is not None
        assert rows["control@x.com"] is None

    # Second run: idempotent — already-reminded booking is not re-sent.
    r2 = client.post("/api/evox/run-reminders", headers=hdr).get_json()
    assert r2["sent"] == 0
    assert sent == ["r@x.com"]  # unchanged
