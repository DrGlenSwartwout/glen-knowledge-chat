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
    # FakeSMTP: 3 practitioner sends + 1 client receipt (Phase 2b) = 4
    all_sends = [s for inst in FakeSMTP.instances for s in inst.sent]
    assert len(all_sends) == 4
    # Reply-To = jane@example.com on the 3 practitioner sends (the receipt
    # has Reply-To = RM_INBOUND_INQUIRY_EMAIL and is sent TO jane).
    practitioner_sends = [s for s in all_sends if "jane@example.com" not in s[1]]
    assert len(practitioner_sends) == 3
    for _, _, raw in practitioner_sends:
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
    # Count sends after first POST: 2 practitioners + 1 client receipt (Phase 2b)
    sends_after_first = sum(len(inst.sent) for inst in FakeSMTP.instances)
    assert sends_after_first == 3   # p1 + p2 + client receipt
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


# --- Slice 3: token-gated routes ---

@pytest.fixture
def supabase_writes(monkeypatch):
    """Capture _set_practitioner_accepts_inquiries calls."""
    calls = []
    def fake(pid, value, verified=False):
        calls.append({"pid": pid, "value": value, "verified": verified})
        return True
    # set after _load_app via the existing fixture; the test re-patches per call
    return calls, fake


def _mint_auth_token(app_module, db, purpose, email, extra, ttl_seconds=86400):
    """Insert an auth_tokens row and return the PLAINTEXT token."""
    import secrets, json, sqlite3, time
    plain = secrets.token_urlsafe(32)
    th = app_module._hash_token(plain)
    now_iso = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    exp_iso = (__import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(seconds=ttl_seconds)).isoformat() + "Z"
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, purpose, email, extra, created_at, expires_at, consumed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL)",
            (th, purpose, email, json.dumps(extra), now_iso, exp_iso))
        cx.commit()
    return plain


def test_practitioner_claim_get_renders_form(app_client, monkeypatch):
    client, app_module, db = app_client
    plain = _mint_auth_token(app_module, db, "practitioner_claim", "a@e.com",
                             {"practitioner_id": "p1"}, ttl_seconds=7*86400)
    r = client.get(f"/practitioner-claim/{plain}")
    assert r.status_code == 200
    assert b"<form" in r.data
    # Practitioner name is rendered safely from extra/lookup
    assert b"Confirm" in r.data or b"accept client inquiries" in r.data


def test_practitioner_claim_post_flips_supabase(app_client, monkeypatch, supabase_writes):
    client, app_module, db = app_client
    calls, fake = supabase_writes
    monkeypatch.setattr(app_module, "_set_practitioner_accepts_inquiries", fake)
    plain = _mint_auth_token(app_module, db, "practitioner_claim", "a@e.com",
                             {"practitioner_id": "p1"}, ttl_seconds=7*86400)
    r = client.post(f"/practitioner-claim/{plain}")
    assert r.status_code == 200
    # Helper called with verified=True, value=True
    assert calls == [{"pid": "p1", "value": True, "verified": True}]
    # Token consumed (can't be reused)
    import sqlite3
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT consumed_at FROM auth_tokens WHERE purpose='practitioner_claim'").fetchone()
    assert row[0] is not None


def test_practitioner_claim_expired_token_returns_4xx(app_client, monkeypatch, supabase_writes):
    client, app_module, db = app_client
    calls, fake = supabase_writes
    monkeypatch.setattr(app_module, "_set_practitioner_accepts_inquiries", fake)
    plain = _mint_auth_token(app_module, db, "practitioner_claim", "a@e.com",
                             {"practitioner_id": "p1"}, ttl_seconds=-1)
    r = client.get(f"/practitioner-claim/{plain}")
    assert r.status_code in (400, 410)
    assert calls == []


def test_practitioner_claim_already_consumed_returns_4xx(app_client, monkeypatch, supabase_writes):
    client, app_module, db = app_client
    calls, fake = supabase_writes
    monkeypatch.setattr(app_module, "_set_practitioner_accepts_inquiries", fake)
    plain = _mint_auth_token(app_module, db, "practitioner_claim", "a@e.com",
                             {"practitioner_id": "p1"}, ttl_seconds=7*86400)
    r1 = client.post(f"/practitioner-claim/{plain}")
    assert r1.status_code == 200
    r2 = client.post(f"/practitioner-claim/{plain}")
    assert r2.status_code in (400, 410)
    assert len(calls) == 1  # only the first call hit Supabase


def test_practitioner_optout_records_and_flips(app_client, monkeypatch, supabase_writes):
    client, app_module, db = app_client
    calls, fake = supabase_writes
    monkeypatch.setattr(app_module, "_set_practitioner_accepts_inquiries", fake)
    plain = _mint_auth_token(app_module, db, "practitioner_optout", "a@e.com",
                             {"practitioner_id": "p1"}, ttl_seconds=365*86400)
    r = client.get(f"/practitioner-optout/{plain}")
    assert r.status_code == 200
    import sqlite3
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT email, practitioner_id FROM practitioner_inquiry_opt_outs").fetchone()
    assert row == ("a@e.com", "p1")
    assert calls == [{"pid": "p1", "value": False, "verified": False}]


def _seed_inquiry_with_reply_token(app_module, db, practitioner_id="p1"):
    """Insert a complete inquiry + a reply token. Returns (inquiry_id, plain_token)."""
    import secrets, uuid, sqlite3
    iid = str(uuid.uuid4())
    plain = secrets.token_urlsafe(32)
    th = app_module._hash_token(plain)
    now_iso = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    exp_iso = (__import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(days=30)).isoformat() + "Z"
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO inquiries (id, created_at, session_id, client_email, client_name, "
            "client_phone, ref_slug, main_challenge, main_goal, practitioner_count, ip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (iid, now_iso, "sess1", "client@e.com", "Client", "", "",
             "challenge text", "goal text", 1, "1.2.3.4"))
        cx.execute(
            "INSERT INTO inquiry_practitioners (id, inquiry_id, practitioner_id, "
            "practitioner_email, status, email_sent_at) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), iid, practitioner_id, "a@e.com", "sent", now_iso))
        cx.execute(
            "INSERT INTO inquiry_reply_tokens (token_hash, inquiry_id, practitioner_id, "
            "created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (th, iid, practitioner_id, now_iso, exp_iso))
        cx.commit()
    return iid, plain


def test_inquiry_reply_get_renders_context(app_client):
    client, app_module, db = app_client
    iid, plain = _seed_inquiry_with_reply_token(app_module, db)
    r = client.get(f"/inquiries/{iid}/p1/reply?token={plain}")
    assert r.status_code == 200
    assert b"challenge text" in r.data
    assert b"goal text" in r.data
    assert b"<form" in r.data
    # impression row written
    import sqlite3
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM inquiry_reply_impressions").fetchone()[0]
    assert n == 1


def test_inquiry_reply_xss_safe(app_client):
    """main_challenge with HTML must be escaped in the rendered page."""
    client, app_module, db = app_client
    import sqlite3, secrets, uuid
    iid = str(uuid.uuid4())
    plain = secrets.token_urlsafe(32)
    th = app_module._hash_token(plain)
    now_iso = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    exp_iso = (__import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(days=30)).isoformat() + "Z"
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO inquiries (id, created_at, session_id, client_email, client_name, "
            "client_phone, ref_slug, main_challenge, main_goal, practitioner_count, ip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (iid, now_iso, "sess1", "client@e.com", "Client", "", "",
             "<script>alert(1)</script>", "<img onerror=x>", 1, "1.2.3.4"))
        cx.execute(
            "INSERT INTO inquiry_practitioners (id, inquiry_id, practitioner_id, "
            "practitioner_email, status, email_sent_at) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), iid, "p1", "a@e.com", "sent", now_iso))
        cx.execute(
            "INSERT INTO inquiry_reply_tokens (token_hash, inquiry_id, practitioner_id, "
            "created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (th, iid, "p1", now_iso, exp_iso))
        cx.commit()
    r = client.get(f"/inquiries/{iid}/p1/reply?token={plain}")
    assert r.status_code == 200
    assert b"<script>alert(1)</script>" not in r.data
    assert b"&lt;script&gt;" in r.data or b"&#x3C;script&#x3E;" in r.data.lower() or b"&lt;script" in r.data


def test_inquiry_reply_post_forwards_and_marks_replied(app_client):
    client, app_module, db = app_client
    iid, plain = _seed_inquiry_with_reply_token(app_module, db)
    r = client.post(f"/inquiries/{iid}/p1/reply",
                    data={"token": plain, "body": "Sounds like a great fit. Available Tuesday."})
    assert r.status_code == 200
    # FakeSMTP captured a forward to the client
    assert any(b"client@e.com" in i.sent[0][2] or "client@e.com" in i.sent[0][1]
               for i in FakeSMTP.instances if i.sent)
    import sqlite3
    with sqlite3.connect(db) as cx:
        reply = cx.execute("SELECT body, reply_method FROM inquiry_replies WHERE inquiry_id=?",
                           (iid,)).fetchone()
        status = cx.execute("SELECT status FROM inquiry_practitioners WHERE inquiry_id=?",
                            (iid,)).fetchone()[0]
    assert reply == ("Sounds like a great fit. Available Tuesday.", "form")
    assert status == "replied"


def test_inquiry_reply_two_views_one_reply(app_client):
    client, app_module, db = app_client
    iid, plain = _seed_inquiry_with_reply_token(app_module, db)
    client.get(f"/inquiries/{iid}/p1/reply?token={plain}")
    client.get(f"/inquiries/{iid}/p1/reply?token={plain}")
    client.post(f"/inquiries/{iid}/p1/reply",
                data={"token": plain, "body": "Yes"})
    import sqlite3
    with sqlite3.connect(db) as cx:
        impressions = cx.execute("SELECT COUNT(*) FROM inquiry_reply_impressions").fetchone()[0]
        replies = cx.execute("SELECT COUNT(*) FROM inquiry_replies").fetchone()[0]
    assert impressions == 2
    assert replies == 1


def test_inquiry_reply_bad_token_4xx(app_client):
    client, app_module, db = app_client
    iid, _ = _seed_inquiry_with_reply_token(app_module, db)
    r = client.get(f"/inquiries/{iid}/p1/reply?token=not-a-real-token")
    assert r.status_code in (400, 403, 404)


def test_inquiry_dedupe_when_all_skipped(app_client):
    """Regression: a replay with the same set should dedupe even when all
    recipients are skipped (e.g. a bogus practitioner_id), not fall through
    to the 1-per-session rate limit."""
    client, app_module, db = app_client
    body = {
        "client_name": "Q",
        "client_email": "q@example.com",
        "main_challenge": "x",
        "main_goal": "y",
        "practitioner_ids": ["zzz-not-real"],
    }
    r1 = client.post("/api/practitioner-finder/inquiry", json=body)
    assert r1.status_code == 200
    j1 = r1.get_json()
    assert j1["sent_count"] == 0
    assert any(s["reason"] == "not_found" for s in j1["skipped"])
    r2 = client.post("/api/practitioner-finder/inquiry", json=body)
    assert r2.status_code == 200, r2.get_json()
    j2 = r2.get_json()
    assert j2.get("deduped") is True
    assert j2["inquiry_id"] == j1["inquiry_id"]


# ── Phase 2b: client receipt + share-with-practitioner ────────────────────────

def test_inquiry_route_sends_client_receipt(app_client):
    """Happy-path inquiry POST also sends a receipt to the client."""
    client, app_module, db = app_client
    body = {
        "client_name": "Jane Doe",
        "client_email": "jane@example.com",
        "main_challenge": "c",
        "main_goal": "g",
        "practitioner_ids": ["p1","p2"],  # both succeed in the fixture
    }
    r = client.post("/api/practitioner-finder/inquiry", json=body)
    assert r.status_code == 200
    # Three sends total: two to practitioners, one receipt to the client
    to_addrs = []
    for inst in FakeSMTP.instances:
        for frm, to, raw in inst.sent:
            to_addrs.extend(to)
    assert "jane@example.com" in to_addrs
    # Receipt has Reply-To = RM_INBOUND_INQUIRY_EMAIL
    receipt_raw = None
    for inst in FakeSMTP.instances:
        for frm, to, raw in inst.sent:
            if "jane@example.com" in to:
                receipt_raw = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                break
    assert receipt_raw is not None
    assert app_module.RM_INBOUND_INQUIRY_EMAIL in receipt_raw
    assert "Your inquiry was sent to 2 practitioners" in receipt_raw
    # Practitioner emails must NOT appear in the receipt body
    assert "a@e.com" not in receipt_raw  # p1's email
    assert "e@e.com" not in receipt_raw  # p5's email


def test_inquiry_route_skips_receipt_when_all_skipped(app_client):
    """No receipt if no practitioners were actually contacted."""
    client, app_module, db = app_client
    body = {
        "client_name": "X",
        "client_email": "x@example.com",
        "main_challenge": "c",
        "main_goal": "g",
        "practitioner_ids": ["p3"],  # opted_out_at_listing in the fixture
    }
    r = client.post("/api/practitioner-finder/inquiry", json=body)
    assert r.status_code == 200
    to_addrs = []
    for inst in FakeSMTP.instances:
        for frm, to, raw in inst.sent:
            to_addrs.extend(to)
    assert "x@example.com" not in to_addrs  # no receipt


def test_recent_inquiry_lookup(app_client):
    """_recent_inquiry_practitioner_ids returns only sent rows for this email
    inside the window."""
    client, app_module, db = app_client
    # Seed an inquiry directly
    import sqlite3, uuid
    iid = str(uuid.uuid4())
    now_iso = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO inquiries (id,created_at,session_id,client_email,client_name,"
            "client_phone,ref_slug,main_challenge,main_goal,practitioner_count,ip) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (iid, now_iso, "s","z@example.com","Z","","","ch","go",2,"1.2.3.4")
        )
        cx.execute(
            "INSERT INTO inquiry_practitioners "
            "(id,inquiry_id,practitioner_id,practitioner_email,status,email_sent_at) "
            "VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), iid, "pa", "pa@e.com", "sent", now_iso))
        cx.execute(
            "INSERT INTO inquiry_practitioners "
            "(id,inquiry_id,practitioner_id,practitioner_email,status,email_sent_at) "
            "VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), iid, "pb", "", "skipped_no_email", None))
        cx.commit()
    rows = app_module._recent_inquiry_practitioner_ids("z@example.com")
    assert len(rows) == 1
    assert rows[0][1] == "pa"


def test_share_page_get_renders(app_client, monkeypatch):
    """GET /share-with-practitioner/<token> renders the preview when token+payload exist."""
    client, app_module, db = app_client
    # Seed: an inbound_leads scoreapp row + an inquiry + a token
    import sqlite3, uuid, json as _json, secrets, hashlib
    from datetime import datetime, timedelta
    with sqlite3.connect(db) as cx:
        # inbound_leads table — must exist; call the init that creates it
        try:
            cx.execute("CREATE TABLE IF NOT EXISTS inbound_leads (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "received_at TEXT, source TEXT, email TEXT, first_name TEXT, last_name TEXT, "
                       "phone TEXT, raw_json TEXT, ghl_contact_id TEXT, ghl_error TEXT, "
                       "last_outbound_at TEXT, tags TEXT)")
        except Exception: pass
        payload = {"data": {
            "email": "shareclient@example.com",
            "total_score": {"percent": 67},
            "quiz_questions": [
                {"question": "Which system is most in need?", "answers": [{"answer": "Immune"}]},
                {"question": "What's the main challenge phase?", "answers": [{"answer": "Inflammation"}]},
            ],
        }}
        cx.execute(
            "INSERT INTO inbound_leads (received_at, source, email, raw_json) VALUES (?,?,?,?)",
            (datetime.utcnow().isoformat()+"Z", "scoreapp", "shareclient@example.com",
             _json.dumps(payload)))
        # Seed inquiry + sent row
        iid = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat()+"Z"
        cx.execute(
            "INSERT INTO inquiries (id,created_at,session_id,client_email,client_name,"
            "client_phone,ref_slug,main_challenge,main_goal,practitioner_count,ip) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (iid, now_iso,"s","shareclient@example.com","Sharer","","",
             "the challenge","the goal",1,"1.2.3.4"))
        cx.execute(
            "INSERT INTO inquiry_practitioners "
            "(id,inquiry_id,practitioner_id,practitioner_email,status,email_sent_at) "
            "VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), iid, "p1", "a@e.com", "sent", now_iso))
        # share token
        plain = secrets.token_urlsafe(32)
        th = hashlib.sha256(plain.encode()).hexdigest()
        exp = (datetime.utcnow()+timedelta(days=30)).isoformat()+"Z"
        cx.execute(
            "INSERT INTO auth_tokens (token_hash,email,purpose,extra,created_at,expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (th, "shareclient@example.com", "practitioner_share",
             _json.dumps({"days":30}), now_iso, exp))
        cx.commit()
    r = client.get(f"/share-with-practitioner/{plain}")
    assert r.status_code == 200
    assert b"the challenge" in r.data or b"Immune" in r.data
    assert b"<form" in r.data


def test_share_post_fans_out_marks_shared_at(app_client, monkeypatch):
    """POST /share-with-practitioner/<token> sends an email per recent practitioner,
    marks shared_at, and re-clicking does NOT re-send."""
    client, app_module, db = app_client
    import sqlite3, uuid, json as _json, secrets, hashlib
    from datetime import datetime, timedelta
    # Same seed pattern as the GET test
    with sqlite3.connect(db) as cx:
        try:
            cx.execute("CREATE TABLE IF NOT EXISTS inbound_leads (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "received_at TEXT, source TEXT, email TEXT, first_name TEXT, last_name TEXT, "
                       "phone TEXT, raw_json TEXT, ghl_contact_id TEXT, ghl_error TEXT, "
                       "last_outbound_at TEXT, tags TEXT)")
        except Exception: pass
        payload = {"data":{"email":"y@example.com","total_score":{"percent":50},
                  "quiz_questions":[{"question":"Q","answers":[{"answer":"A"}]}]}}
        cx.execute("INSERT INTO inbound_leads (received_at,source,email,raw_json) VALUES (?,?,?,?)",
                   (datetime.utcnow().isoformat()+"Z","scoreapp","y@example.com",_json.dumps(payload)))
        iid = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat()+"Z"
        cx.execute(
            "INSERT INTO inquiries (id,created_at,session_id,client_email,client_name,"
            "client_phone,ref_slug,main_challenge,main_goal,practitioner_count,ip) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (iid,now_iso,"s","y@example.com","Y","","","c","g",2,"1.2.3.4"))
        for pid, pemail in [("p1","a@e.com"),("p2","b@e.com")]:
            cx.execute(
                "INSERT INTO inquiry_practitioners "
                "(id,inquiry_id,practitioner_id,practitioner_email,status,email_sent_at) "
                "VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), iid, pid, pemail, "sent", now_iso))
        plain = secrets.token_urlsafe(32)
        th = hashlib.sha256(plain.encode()).hexdigest()
        exp = (datetime.utcnow()+timedelta(days=30)).isoformat()+"Z"
        cx.execute(
            "INSERT INTO auth_tokens (token_hash,email,purpose,extra,created_at,expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (th,"y@example.com","practitioner_share",_json.dumps({"days":30}),now_iso,exp))
        cx.commit()
    # Reset SMTP fake instances
    FakeSMTP.instances = []
    r1 = client.post(f"/share-with-practitioner/{plain}",
                     data={"token": plain})
    assert r1.status_code == 200
    sends = sum(len(i.sent) for i in FakeSMTP.instances)
    assert sends == 2  # one per practitioner
    # Bodies include the full Q&A
    raw_all = b"".join(
        (raw if isinstance(raw,bytes) else raw.encode("utf-8"))
        for i in FakeSMTP.instances for (frm,to,raw) in i.sent)
    assert b"Q" in raw_all and b"A" in raw_all
    # shared_at flipped
    with sqlite3.connect(db) as cx:
        n_shared = cx.execute(
            "SELECT COUNT(*) FROM inquiry_practitioners WHERE shared_at IS NOT NULL"
        ).fetchone()[0]
    assert n_shared == 2
    # Re-click: should NOT send again (token consumed; idempotency via shared_at)
    FakeSMTP.instances = []
    r2 = client.post(f"/share-with-practitioner/{plain}",
                     data={"token": plain})
    # Either 4xx (token consumed) OR 200 with 0 new sends
    new_sends = sum(len(i.sent) for i in FakeSMTP.instances)
    assert new_sends == 0
