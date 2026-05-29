import io
import json
import sqlite3
import pytest
import smtplib

# Reuse the _load_app helper from the route tests
from tests.test_begin_routes import _load_app


class FakeSMTP:
    """Drop-in replacement for smtplib.SMTP that records every send."""
    instances = []
    def __init__(self, host, port=0, *a, **k):
        self.host = host; self.port = port
        self.starttls_called = False
        self.login_args = None
        self.sent = []          # list of (from_addr, to_addrs, msg_bytes)
        self.quit_called = False
        FakeSMTP.instances.append(self)
    def starttls(self, *a, **k): self.starttls_called = True
    def login(self, u, p): self.login_args = (u, p)
    def sendmail(self, frm, to, msg): self.sent.append((frm, to, msg))
    def quit(self): self.quit_called = True
    # support context-manager use just in case the helper uses with-statement
    def __enter__(self): return self
    def __exit__(self, *a): self.quit(); return False


@pytest.fixture(autouse=True)
def reset_fakesmtp():
    FakeSMTP.instances = []
    yield


def test_init_inquiry_tables_creates_six_tables_idempotent(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    import sqlite3
    cx = sqlite3.connect(db)
    # call twice; must not raise; must create the tables
    app_module.init_inquiry_tables(cx)
    app_module.init_inquiry_tables(cx)
    names = {r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    expected = {"inquiries","inquiry_practitioners","inquiry_reply_tokens",
                "inquiry_replies","inquiry_reply_impressions",
                "practitioner_inquiry_opt_outs"}
    assert expected.issubset(names)


def test_send_inquiry_email_honors_reply_to(monkeypatch):
    app_module = _load_app()
    # set minimal SMTP env so the helper does real flow with FakeSMTP
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASS", "pw")
    monkeypatch.setenv("SMTP_FROM", "hello@remedymatch.com")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    ok = app_module._send_inquiry_email(
        "pract@example.com", "Test subject", "Test body line.",
        reply_to="client@example.com")
    assert ok is True
    assert len(FakeSMTP.instances) == 1
    inst = FakeSMTP.instances[0]
    assert inst.starttls_called and inst.quit_called
    assert inst.sent
    frm, to, raw = inst.sent[0]
    assert "pract@example.com" in to
    # Reply-To header must be set to the client
    # raw may be str (as_string()) or bytes (as_bytes()) depending on the send path
    raw_lower = raw.lower() if isinstance(raw, str) else raw.lower().decode("utf-8", errors="replace")
    raw_norm  = raw        if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    assert "reply-to: client@example.com" in raw_lower or "Reply-To: client@example.com" in raw_norm


def test_send_inquiry_email_returns_false_on_smtp_failure(monkeypatch):
    app_module = _load_app()
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASS", "pw")
    class BrokenSMTP(FakeSMTP):
        def sendmail(self, *a, **k):
            raise smtplib.SMTPException("boom")
    monkeypatch.setattr(smtplib, "SMTP", BrokenSMTP)
    ok = app_module._send_inquiry_email("x@e.com","s","b",reply_to="c@e.com")
    assert ok is False  # never raises


def test_send_inquiry_email_no_smtp_env_returns_true(monkeypatch):
    """Dev fallback: with no SMTP_HOST/USER/PASS, the helper logs to stdout and returns True."""
    app_module = _load_app()
    for k in ("SMTP_HOST","SMTP_USER","SMTP_PASS"):
        monkeypatch.delenv(k, raising=False)
    ok = app_module._send_inquiry_email("x@e.com","s","b",reply_to=None)
    assert ok is True


# ── Slice 2: POST /api/practitioner-finder/inquiry ───────────────────────────

# Five fake practitioners:
#   p1 – accepts_inquiries=True,  email present
#   p2 – accepts_inquiries=None,  email present (scraped-but-uncontacted, ALLOWED)
#   p3 – accepts_inquiries=False, email present (opted out at listing)
#   p4 – accepts_inquiries=True,  email empty   (no email)
#   p5 – accepts_inquiries=None,  email present (scraped-but-uncontacted, ALLOWED)
_FAKE_DB = {
    "p1": {"id": "p1", "email": "a@e.com", "name": "A", "accepts_inquiries": True},
    "p2": {"id": "p2", "email": "b@e.com", "name": "B", "accepts_inquiries": None},
    "p3": {"id": "p3", "email": "c@e.com", "name": "C", "accepts_inquiries": False},
    "p4": {"id": "p4", "email": "",        "name": "D", "accepts_inquiries": True},
    "p5": {"id": "p5", "email": "e@e.com", "name": "E", "accepts_inquiries": None},
}


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    # init all tables (auth + inquiry + journey) against the tmp db
    import begin_funnel
    cx = sqlite3.connect(db)
    app_module.init_inquiry_tables(cx)
    begin_funnel.init_journey_tables(cx)
    cx.close()
    app_module._init_auth_tables()   # creates auth_tokens, users, sessions in tmp db
    # stub the practitioner fetcher
    monkeypatch.setattr(
        app_module, "_fetch_practitioners_by_ids",
        lambda ids: [_FAKE_DB[i] for i in ids if i in _FAKE_DB],
    )
    # fake SMTP
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "u@e.com")
    monkeypatch.setenv("SMTP_PASS", "p")
    monkeypatch.setenv("SMTP_FROM", "hello@remedymatch.com")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    client = app_module.app.test_client()
    return client, app_module, db


def _inquiry_body(**overrides):
    base = {
        "client_name": "Jane Doe",
        "client_email": "jane@example.com",
        "main_challenge": "chronic fatigue",
        "main_goal": "regain energy",
        "practitioner_ids": ["p1"],
    }
    base.update(overrides)
    return base


def test_inquiry_happy_path_sends_to_three_records_skips_opted_out_and_no_email(app_client):
    client, app_module, db = app_client
    r = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(practitioner_ids=["p1", "p2", "p3", "p4", "p5"]),
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["sent_count"] == 3, f"expected 3, got {body['sent_count']}"
    skipped_ids = {s["practitioner_id"] for s in body["skipped"]}
    assert "p3" in skipped_ids
    assert "p4" in skipped_ids
    reasons = {s["practitioner_id"]: s["reason"] for s in body["skipped"]}
    assert reasons["p3"] == "opted_out_at_listing"
    assert reasons["p4"] == "no_email"
    # FakeSMTP: 3 sends, all with Reply-To = jane@example.com
    all_sends = [s for inst in FakeSMTP.instances for s in inst.sent]
    assert len(all_sends) == 3
    for _, _, raw in all_sends:
        raw_str = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
        assert "jane@example.com" in raw_str
    # DB checks
    cx = sqlite3.connect(db)
    inq_rows = cx.execute("SELECT * FROM inquiries").fetchall()
    assert len(inq_rows) == 1
    ip_rows = cx.execute("SELECT status FROM inquiry_practitioners").fetchall()
    sent_rows = [r for r in ip_rows if r[0] == "sent"]
    assert len(sent_rows) == 3
    rt_rows = cx.execute("SELECT * FROM inquiry_reply_tokens").fetchall()
    assert len(rt_rows) == 3
    je_rows = cx.execute(
        "SELECT detail FROM journey_events WHERE trigger='practitioner_inquiry'"
    ).fetchall()
    assert len(je_rows) == 1
    detail = json.loads(je_rows[0][0])
    assert detail["count"] == 3
    cx.close()


def test_inquiry_amg_session_minted_when_absent(app_client):
    client, app_module, db = app_client
    r = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(),
        content_type="application/json",
    )
    assert r.status_code == 200
    set_cookie = r.headers.get("Set-Cookie", "")
    assert "amg_session=" in set_cookie
    assert "Max-Age=31536000" in set_cookie or "max-age=31536000" in set_cookie.lower()


def test_inquiry_amg_session_reused_when_present(app_client):
    client, app_module, db = app_client
    # first POST mints the session
    r1 = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(),
        content_type="application/json",
    )
    assert r1.status_code == 200
    # extract session_id from DB
    cx = sqlite3.connect(db)
    session_id = cx.execute("SELECT session_id FROM inquiries").fetchone()[0]
    cx.close()
    # second POST with cookie
    r2 = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(client_email="other@example.com"),
        content_type="application/json",
        headers={"Cookie": f"amg_session={session_id}"},
    )
    # This will 429 (same session, different email) -- that's fine for this test;
    # we only care that if we use a brand-new different session it doesn't re-mint.
    # Actually: let's just confirm the session in DB matches the cookie we sent.
    # The 429 path doesn't set a new cookie, so check status_code != 200 means no new cookie.
    # Simpler: POST a deduped request (same session + same email + same set) -> 200 deduped, no new cookie.
    r3 = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(),   # same email + same set -> deduped
        content_type="application/json",
        headers={"Cookie": f"amg_session={session_id}"},
    )
    assert r3.status_code == 200
    body = r3.get_json()
    assert body.get("deduped") is True
    # No new Set-Cookie on a deduped response
    assert "amg_session=" not in (r3.headers.get("Set-Cookie", ""))
    # inquiries.session_id must match the cookie we sent
    cx = sqlite3.connect(db)
    sid_in_db = cx.execute("SELECT session_id FROM inquiries LIMIT 1").fetchone()[0]
    cx.close()
    assert sid_in_db == session_id


def test_inquiry_rate_limit_one_per_session_per_24h(app_client):
    client, app_module, db = app_client
    # First POST succeeds
    r1 = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(),
        content_type="application/json",
    )
    assert r1.status_code == 200
    cx = sqlite3.connect(db)
    session_id = cx.execute("SELECT session_id FROM inquiries").fetchone()[0]
    cx.close()
    # Second POST same session, different email -> 429
    r2 = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(client_email="different@example.com"),
        content_type="application/json",
        headers={"Cookie": f"amg_session={session_id}"},
    )
    assert r2.status_code == 429
    body = r2.get_json()
    assert "one inquiry per day" in body.get("error", "").lower()


def test_inquiry_dedupe_same_session_same_email_same_set(app_client):
    client, app_module, db = app_client
    payload = _inquiry_body(practitioner_ids=["p1", "p2"])
    # First POST
    r1 = client.post(
        "/api/practitioner-finder/inquiry",
        json=payload,
        content_type="application/json",
    )
    assert r1.status_code == 200
    original_id = r1.get_json()["inquiry_id"]
    cx = sqlite3.connect(db)
    session_id = cx.execute("SELECT session_id FROM inquiries").fetchone()[0]
    cx.close()
    # Count sends after first POST
    sends_after_first = sum(len(inst.sent) for inst in FakeSMTP.instances)
    assert sends_after_first == 2   # p1 + p2
    # Replay with same session + same payload
    r2 = client.post(
        "/api/practitioner-finder/inquiry",
        json=payload,
        content_type="application/json",
        headers={"Cookie": f"amg_session={session_id}"},
    )
    assert r2.status_code == 200
    body2 = r2.get_json()
    assert body2["inquiry_id"] == original_id
    assert body2.get("deduped") is True
    assert body2["sent_count"] == 0
    # No new emails sent during replay
    sends_after_second = sum(len(inst.sent) for inst in FakeSMTP.instances)
    assert sends_after_second == sends_after_first


def test_inquiry_rate_limit_three_per_ip_per_24h(app_client):
    client, app_module, db = app_client
    # Three distinct sessions from the same IP (X-Forwarded-For: 1.2.3.4).
    # Use client.set_cookie before each request so the session id we supply
    # is what the server reads; the server only mints a cookie when the
    # request carries none, so set_cookie + distinct session values each time.
    for i in range(3):
        client.set_cookie("amg_session", f"iptest_session{i}xxxxxx")
        r = client.post(
            "/api/practitioner-finder/inquiry",
            json=_inquiry_body(
                client_email=f"user{i}@example.com",
                practitioner_ids=["p1"],
            ),
            content_type="application/json",
            headers={"X-Forwarded-For": "1.2.3.4"},
        )
        assert r.status_code == 200, f"request {i} should succeed, got {r.status_code}: {r.get_json()}"
    # 4th request from same IP -> 429
    client.set_cookie("amg_session", "iptest_session4xxxxxx")
    r4 = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(
            client_email="user4@example.com",
            practitioner_ids=["p1"],
        ),
        content_type="application/json",
        headers={"X-Forwarded-For": "1.2.3.4"},
    )
    assert r4.status_code == 429
    body = r4.get_json()
    assert "network" in body.get("error", "").lower() or "too many" in body.get("error", "").lower()


def test_inquiry_max_twenty_practitioners(app_client):
    client, app_module, db = app_client
    r = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(practitioner_ids=[f"x{i}" for i in range(21)]),
        content_type="application/json",
    )
    assert r.status_code == 400
    body = r.get_json()
    assert "20" in body.get("error", "") or "max" in body.get("error", "").lower()


def test_inquiry_smtp_failure_marks_status_failed_but_others_send(app_client, monkeypatch):
    client, app_module, db = app_client

    call_count = {"n": 0}

    class PartialFakeSMTP(FakeSMTP):
        def sendmail(self, frm, to, msg):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise smtplib.SMTPException("simulated failure on 2nd send")
            super().sendmail(frm, to, msg)

    monkeypatch.setattr(smtplib, "SMTP", PartialFakeSMTP)

    r = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(practitioner_ids=["p1", "p2", "p5"]),
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    # 2 sent, 1 failed
    assert body["sent_count"] == 2
    # DB: one inquiry_practitioners row with status='failed'
    cx = sqlite3.connect(db)
    statuses = [row[0] for row in cx.execute(
        "SELECT status FROM inquiry_practitioners"
    ).fetchall()]
    cx.close()
    assert statuses.count("failed") == 1
    assert statuses.count("sent") == 2


def test_inquiry_required_fields_missing_400(app_client):
    client, app_module, db = app_client
    payload = {
        "client_name": "Jane",
        "client_email": "jane@example.com",
        # main_challenge missing
        "main_goal": "feel better",
        "practitioner_ids": ["p1"],
    }
    r = client.post(
        "/api/practitioner-finder/inquiry",
        json=payload,
        content_type="application/json",
    )
    assert r.status_code == 400
    body = r.get_json()
    assert "error" in body


def test_inquiry_unknown_practitioner_id_skipped_not_found(app_client):
    client, app_module, db = app_client
    r = client.post(
        "/api/practitioner-finder/inquiry",
        json=_inquiry_body(practitioner_ids=["pZ"]),
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["sent_count"] == 0
    skipped = body["skipped"]
    assert any(
        s["practitioner_id"] == "pZ" and s["reason"] == "not_found"
        for s in skipped
    )
    # inquiries row still inserted for analytics
    cx = sqlite3.connect(db)
    count = cx.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0]
    cx.close()
    assert count == 1
