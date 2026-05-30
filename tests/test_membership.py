import pytest
import sqlite3
from datetime import datetime, timedelta

from tests.test_begin_routes import _load_app


def test_init_membership_tables_idempotent_creates_three_tables(tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    # call twice; must not raise; must create all three tables
    app_module.init_membership_tables(cx)
    app_module.init_membership_tables(cx)
    names = {r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    expected = {"memberships", "escalation_queue", "studio_credit_intents"}
    assert expected.issubset(names)


def test_init_membership_tables_creates_expected_columns(tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    app_module.init_membership_tables(cx)
    mem_cols = {r[1] for r in cx.execute("PRAGMA table_info(memberships)").fetchall()}
    assert {"id","email","granted_at","expires_at","granted_by","source",
            "truly_vip_ref","notes","last_reminder_at"}.issubset(mem_cols)
    esc_cols = {r[1] for r in cx.execute("PRAGMA table_info(escalation_queue)").fetchall()}
    assert {"id","created_at","email","query_text","ai_response","ai_confidence",
            "flag_reason","status","glen_reply_url","glen_reply_text","replied_at"}.issubset(esc_cols)
    sci_cols = {r[1] for r in cx.execute("PRAGMA table_info(studio_credit_intents)").fetchall()}
    assert {"id","created_at","email","studio_ref","notes"}.issubset(sci_cols)


@pytest.fixture
def app_module_with_db(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    cx = sqlite3.connect(db)
    # auth_tokens exists from existing init; ensure it does
    try:
        import begin_funnel
        begin_funnel.init_journey_tables(cx)
    except Exception: pass
    # Ensure auth_tokens table exists in the test DB (mirror app.py:145 schema)
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
          token_hash TEXT PRIMARY KEY,
          email      TEXT,
          purpose    TEXT NOT NULL,
          extra      TEXT,
          created_at TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          consumed_at TEXT
        );
    """)
    app_module.init_membership_tables(cx)
    cx.close()
    return app_module, db


def test_mint_returns_plaintext_token_and_stores_hash(app_module_with_db):
    app_module, db = app_module_with_db
    plain = app_module._mint_membership_magic_link("user@example.com")
    assert isinstance(plain, str) and len(plain) > 20
    cx = sqlite3.connect(db)
    th = app_module._hash_token(plain)
    row = cx.execute(
        "SELECT email, purpose, consumed_at FROM auth_tokens WHERE token_hash=?",
        (th,)
    ).fetchone()
    assert row is not None
    assert row[0] == "user@example.com"
    assert row[1] == "membership_magic_link"
    assert row[2] is None


def test_validate_happy_path_returns_email(app_module_with_db):
    app_module, db = app_module_with_db
    plain = app_module._mint_membership_magic_link("happy@example.com")
    email = app_module._validate_membership_magic_link(plain)
    assert email == "happy@example.com"


def test_validate_returns_none_when_expired(app_module_with_db):
    app_module, db = app_module_with_db
    plain = app_module._mint_membership_magic_link("expired@example.com", ttl_min=-1)
    email = app_module._validate_membership_magic_link(plain)
    assert email is None


def test_validate_returns_none_when_consumed(app_module_with_db):
    app_module, db = app_module_with_db
    plain = app_module._mint_membership_magic_link("once@example.com")
    th = app_module._hash_token(plain)
    cx = sqlite3.connect(db)
    cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
               (datetime.utcnow().isoformat()+"Z", th))
    cx.commit(); cx.close()
    email = app_module._validate_membership_magic_link(plain)
    assert email is None


def test_validate_returns_none_when_purpose_mismatch(app_module_with_db):
    app_module, db = app_module_with_db
    # Manually insert a token with a different purpose; the validator should reject.
    import secrets
    plain = secrets.token_urlsafe(32)
    th = app_module._hash_token(plain)
    now_iso = datetime.utcnow().isoformat()+"Z"
    exp_iso = (datetime.utcnow()+timedelta(minutes=15)).isoformat()+"Z"
    cx = sqlite3.connect(db)
    cx.execute(
        "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
        "VALUES (?,?,?,?,?,?)",
        (th, "wrong@example.com", "other_purpose", "{}", now_iso, exp_iso))
    cx.commit(); cx.close()
    assert app_module._validate_membership_magic_link(plain) is None


# ── Slice 2: admin grant + admin escalations listing ─────────────────────────

import smtplib
import json as _json
from tests.test_inquiry import FakeSMTP


@pytest.fixture
def fake_smtp(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "u@e.com")
    monkeypatch.setenv("SMTP_PASS", "p")
    monkeypatch.setenv("SMTP_FROM", "hello@remedymatch.com")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    FakeSMTP.instances = []
    yield FakeSMTP


@pytest.fixture
def app_client_mem(monkeypatch, tmp_path, fake_smtp):
    """Flask test client with mem-init done and CONSOLE_SECRET set."""
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    # Patch both app.CONSOLE_SECRET and dashboard.CONSOLE_SECRET (the decorator
    # reads the dashboard module's copy, not app.py's).
    import dashboard as _dashboard
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "test-console-secret-xyz")
    monkeypatch.setattr(_dashboard, "CONSOLE_SECRET", "test-console-secret-xyz")
    cx = sqlite3.connect(db)
    # auth_tokens (mirror app.py:145 schema)
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
          token_hash TEXT PRIMARY KEY,
          email      TEXT,
          purpose    TEXT NOT NULL,
          extra      TEXT,
          created_at TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          consumed_at TEXT
        );
    """)
    # journey_events
    try:
        import begin_funnel
        begin_funnel.init_journey_tables(cx)
    except Exception: pass
    app_module.init_membership_tables(cx)
    cx.close()
    return app_module.app.test_client(), app_module, db


def _ckey():
    return {"X-Console-Key": "test-console-secret-xyz"}


def test_grant_requires_console_key(app_client_mem):
    client, _, _ = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"x@e.com","source":"video"})
    assert r.status_code in (401, 403)


def test_grant_happy_path_inserts_row_sends_email_logs_event(app_client_mem, fake_smtp):
    client, app_module, db = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"jane@example.com","source":"video",
                          "truly_vip_ref":"https://truly.vip/Results/abc"},
                    headers=_ckey())
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert "membership_id" in body
    assert "magic_link_url" in body
    assert body["magic_link_url"].endswith(  # ends with the /coaching/auth/<token>
        body["magic_link_url"].split("/coaching/auth/")[-1])
    assert "/coaching/auth/" in body["magic_link_url"]
    assert "expires_at" in body
    # memberships row
    cx = sqlite3.connect(db)
    mrow = cx.execute(
        "SELECT email, source, truly_vip_ref, granted_by FROM memberships WHERE id=?",
        (body["membership_id"],)
    ).fetchone()
    assert mrow == ("jane@example.com", "video", "https://truly.vip/Results/abc", "glen") or \
           (mrow[0] == "jane@example.com" and mrow[1] == "video" and mrow[2] == "https://truly.vip/Results/abc")
    # FakeSMTP captured the magic-link email TO the member
    sends = [s for inst in fake_smtp.instances for s in inst.sent]
    assert any("jane@example.com" in to for (_, to, _) in sends)
    # journey_events row
    je = cx.execute(
        "SELECT detail FROM journey_events WHERE trigger='membership_granted'"
    ).fetchall()
    assert len(je) == 1
    d = _json.loads(je[0][0])
    assert d.get("source") == "video"
    assert d.get("days") == 30
    cx.close()


def test_grant_rejects_unknown_source(app_client_mem):
    client, _, _ = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"x@e.com","source":"bogus_source"},
                    headers=_ckey())
    assert r.status_code == 400


def test_grant_rejects_missing_email(app_client_mem):
    client, _, _ = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"source":"video"},
                    headers=_ckey())
    assert r.status_code == 400


def test_grant_default_30_days_when_days_unset(app_client_mem):
    client, app_module, db = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"d30@example.com","source":"video"},
                    headers=_ckey())
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    granted_at, expires_at = cx.execute(
        "SELECT granted_at, expires_at FROM memberships WHERE email=?",
        ("d30@example.com",)
    ).fetchone()
    g = datetime.fromisoformat(granted_at.rstrip("Z"))
    e = datetime.fromisoformat(expires_at.rstrip("Z"))
    # ~30 days within a few seconds
    delta = (e - g).total_seconds()
    assert 30*86400 - 60 <= delta <= 30*86400 + 60
    cx.close()


def test_grant_custom_days_when_specified(app_client_mem):
    client, app_module, db = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"d90@example.com","source":"video","days":90},
                    headers=_ckey())
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    granted_at, expires_at = cx.execute(
        "SELECT granted_at, expires_at FROM memberships WHERE email=?",
        ("d90@example.com",)
    ).fetchone()
    g = datetime.fromisoformat(granted_at.rstrip("Z"))
    e = datetime.fromisoformat(expires_at.rstrip("Z"))
    delta = (e - g).total_seconds()
    assert 90*86400 - 60 <= delta <= 90*86400 + 60
    cx.close()


def test_admin_escalations_requires_console_key(app_client_mem):
    client, _, _ = app_client_mem
    r = client.get("/admin/escalations")
    assert r.status_code in (401, 403)


def test_admin_escalations_returns_only_pending(app_client_mem):
    client, app_module, db = app_client_mem
    # Seed two rows: one pending, one replied
    import uuid
    now_iso = datetime.utcnow().isoformat()+"Z"
    cx = sqlite3.connect(db)
    cx.execute(
        "INSERT INTO escalation_queue "
        "(id, created_at, email, query_text, ai_response, ai_confidence, flag_reason, status) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), now_iso, "a@e.com", "pending q", "ai resp", 0.5, "member_request", "pending"))
    cx.execute(
        "INSERT INTO escalation_queue "
        "(id, created_at, email, query_text, ai_response, ai_confidence, flag_reason, status) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), now_iso, "b@e.com", "replied q", "ai resp", 0.5, "member_request", "replied"))
    cx.commit(); cx.close()
    r = client.get("/admin/escalations", headers=_ckey())
    assert r.status_code == 200
    arr = r.get_json()
    assert isinstance(arr, list)
    emails = [item["email"] for item in arr]
    assert "a@e.com" in emails
    assert "b@e.com" not in emails


# ── Slice 3: member auth + dashboard shell ───────────────────────────────────

import re


@pytest.fixture
def app_client_member(monkeypatch, tmp_path, fake_smtp):
    """Test client with a seeded active membership for jane@example.com."""
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    cx = sqlite3.connect(db)
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
          token_hash TEXT PRIMARY KEY, email TEXT, purpose TEXT NOT NULL,
          extra TEXT, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
          consumed_at TEXT
        );
    """)
    try:
        import begin_funnel
        begin_funnel.init_journey_tables(cx)
    except Exception: pass
    app_module.init_membership_tables(cx)
    # seed an inbound_leads row for client_first lookup (used by /coaching)
    try:
        cx.execute("CREATE TABLE IF NOT EXISTS inbound_leads (id INTEGER PRIMARY KEY AUTOINCREMENT, received_at TEXT, source TEXT, email TEXT, first_name TEXT, last_name TEXT, phone TEXT, raw_json TEXT, ghl_contact_id TEXT, ghl_error TEXT, last_outbound_at TEXT, tags TEXT)")
        cx.execute("INSERT INTO inbound_leads (received_at, source, email, first_name) VALUES (?,?,?,?)",
                   (datetime.utcnow().isoformat()+"Z", "scoreapp", "jane@example.com", "Jane"))
    except Exception: pass
    # seed an active membership
    import uuid as _uuid
    cx.execute(
        "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source) "
        "VALUES (?,?,?,?,?,?)",
        (str(_uuid.uuid4()), "jane@example.com",
         datetime.utcnow().isoformat()+"Z",
         (datetime.utcnow()+timedelta(days=27)).isoformat()+"Z",
         "glen", "video")
    )
    cx.commit(); cx.close()
    return app_module.app.test_client(), app_module, db


def test_active_membership_for_email_returns_row(app_client_member):
    _, app_module, _ = app_client_member
    row = app_module._active_membership_for_email("jane@example.com")
    assert row is not None
    assert row["email"] == "jane@example.com"
    assert "days_remaining" in row
    assert 26 <= row["days_remaining"] <= 28


def test_active_membership_for_email_returns_none_when_expired(app_client_member):
    _, app_module, db = app_client_member
    cx = sqlite3.connect(db)
    cx.execute("UPDATE memberships SET expires_at=? WHERE email='jane@example.com'",
               ((datetime.utcnow() - timedelta(days=1)).isoformat()+"Z",))
    cx.commit(); cx.close()
    assert app_module._active_membership_for_email("jane@example.com") is None


def test_auth_token_consumes_and_sets_cookie_redirects_to_coaching(app_client_member):
    client, app_module, db = app_client_member
    plain = app_module._mint_membership_magic_link("jane@example.com")
    r = client.get(f"/coaching/auth/{plain}", follow_redirects=False)
    assert r.status_code in (302, 303)
    assert r.headers.get("Location", "").endswith("/coaching")
    set_cookie = r.headers.get("Set-Cookie", "")
    # multiple Set-Cookie headers can appear; use headers.getlist if available
    cookies = r.headers.getlist("Set-Cookie")
    blob = "\n".join(cookies)
    assert "rm_member_email" in blob
    assert "jane%40example.com" in blob or "jane@example.com" in blob
    # token consumed_at flipped
    cx = sqlite3.connect(db)
    th = app_module._hash_token(plain)
    consumed_at = cx.execute("SELECT consumed_at FROM auth_tokens WHERE token_hash=?", (th,)).fetchone()[0]
    cx.close()
    assert consumed_at is not None


def test_auth_token_bad_token_renders_error_410(app_client_member):
    client, _, _ = app_client_member
    r = client.get("/coaching/auth/not-a-real-token")
    assert r.status_code == 410
    assert b"<html" in r.data.lower() or b"<form" in r.data.lower() or b"invalid" in r.data.lower() or b"valid" in r.data.lower()


def test_login_request_for_unknown_email_returns_200_no_leak(app_client_member, fake_smtp):
    client, _, _ = app_client_member
    fake_smtp.instances = []
    r = client.post("/coaching/login-request", json={"email":"unknown@example.com"})
    assert r.status_code == 200
    j = r.get_json()
    assert "If an active membership" in j["message"]
    # We MUST still send the email so the response timing/content doesn't leak status
    sends = [s for inst in fake_smtp.instances for s in inst.sent]
    assert any("unknown@example.com" in to for (_, to, _) in sends)


def test_login_request_for_active_member_sends_magic_link(app_client_member, fake_smtp):
    client, _, _ = app_client_member
    fake_smtp.instances = []
    r = client.post("/coaching/login-request", json={"email":"jane@example.com"})
    assert r.status_code == 200
    sends = [s for inst in fake_smtp.instances for s in inst.sent]
    raw_all = b"".join(
        raw if isinstance(raw, bytes) else raw.encode("utf-8")
        for (_, _, raw) in sends)
    assert b"/coaching/auth/" in raw_all


def test_coaching_active_renders_iframe_and_days_remaining(app_client_member):
    client, app_module, db = app_client_member
    # Set the rm_member_email cookie
    client.set_cookie("rm_member_email", "jane@example.com")
    r = client.get("/coaching")
    assert r.status_code == 200
    # Iframe to /embed exists
    assert b"/embed" in r.data
    # Days remaining rendered
    assert re.search(rb"2[6-8]", r.data) is not None  # 26/27/28 days
    # Brand bar
    assert b"Remedy" in r.data
    # First name from inbound_leads or email local-part
    assert b"Jane" in r.data or b"jane" in r.data


def test_coaching_lapsed_renders_truly_vip_cta(app_client_member):
    client, app_module, db = app_client_member
    # No cookie, no active membership for the implicit visitor
    r = client.get("/coaching")
    assert r.status_code == 200
    # Lapsed state: CTA to Truly.VIP/Results
    body_lower = r.data.lower()
    assert b"truly.vip/results" in body_lower


# ── Slice 4: member-mode /chat overlay ───────────────────────────────────────

import json as _json4


def test_active_membership_for_email_returns_none_for_unknown(app_client_member):
    _, app_module, _ = app_client_member
    assert app_module._active_membership_for_email("nobody@example.com") is None


def test_member_context_aggregates_intake_inquiries_queries(app_client_member):
    _, app_module, db = app_client_member
    # Seed:
    #   - inbound_leads scoreapp row for jane (already partially seeded with first_name; add raw_json)
    #   - an inquiries row for jane
    #   - query_log rows for jane
    import uuid as _uuid
    now_iso = datetime.utcnow().isoformat()+"Z"
    cx = sqlite3.connect(db)
    cx.execute("UPDATE inbound_leads SET raw_json=? WHERE email=?",
               (_json4.dumps({"data":{"total_score":{"percent":67},
                              "quiz_questions":[{"question":"Which system?","answers":[{"answer":"Immune"}]}]}}),
                "jane@example.com"))
    # Ensure the inquiries table (and ip column) exist in the test DB.
    cx.execute("""
        CREATE TABLE IF NOT EXISTS inquiries (
          id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
          session_id TEXT NOT NULL, client_email TEXT NOT NULL,
          client_name TEXT, client_phone TEXT, ref_slug TEXT,
          main_challenge TEXT NOT NULL, main_goal TEXT NOT NULL,
          practitioner_count INTEGER NOT NULL
        )
    """)
    try:
        cx.execute("ALTER TABLE inquiries ADD COLUMN ip TEXT")
    except Exception:
        pass  # column already exists
    cx.execute(
        "INSERT INTO inquiries (id, created_at, session_id, client_email, client_name, "
        "client_phone, ref_slug, main_challenge, main_goal, practitioner_count, ip) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (str(_uuid.uuid4()), now_iso, "s1", "jane@example.com", "Jane", "", "",
         "fatigue and brain fog", "stable energy", 1, "1.2.3.4"))
    try:
        cx.execute("CREATE TABLE IF NOT EXISTS query_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, email TEXT, question TEXT, answer TEXT, sources TEXT, session_id TEXT)")
    except Exception: pass
    cx.execute("INSERT INTO query_log (ts, email, question, answer) VALUES (?,?,?,?)",
               (now_iso, "jane@example.com", "How do I rebuild my microbiome?", "..."))
    cx.execute("INSERT INTO query_log (ts, email, question, answer) VALUES (?,?,?,?)",
               (now_iso, "jane@example.com", "Should I do a heavy metal detox?", "..."))
    cx.commit(); cx.close()
    ctx = app_module._member_context_for_email("jane@example.com")
    assert isinstance(ctx, dict)
    # Includes the intake fragments
    blob = _json4.dumps(ctx)
    assert "Immune" in blob or "67" in blob
    # Includes recent inquiry
    assert "fatigue" in blob.lower() or "stable" in blob.lower()
    # Includes recent queries
    assert "microbiome" in blob.lower() or "metal" in blob.lower()


def test_member_context_returns_empty_dict_for_unknown_email(app_client_member):
    _, app_module, _ = app_client_member
    ctx = app_module._member_context_for_email("nobody-here@example.com")
    assert isinstance(ctx, dict)
    # Empty/falsy expected fields
    assert not ctx.get("recent_inquiries")
    assert not ctx.get("recent_queries")


def _make_fake_match(text="Test snippet"):
    """Minimal Pinecone-style match object for /chat handler stubs."""
    class _M:
        def __init__(self, t):
            self.metadata = {"text": t, "source": "test", "url": ""}
            self.score = 0.9
            self.id = "t1"
    return _M(text)


def test_chat_member_mode_prepends_member_context_to_rag(app_client_member, monkeypatch):
    """The /chat SSE handler MUST inject a 'MEMBER CONTEXT' block when an active
    member cookie is present.  We stub embed + query_all_namespaces so the handler
    reaches the context-build + test-seam code without external API calls."""
    client, app_module, db = app_client_member
    monkeypatch.setattr(app_module, "_LAST_CONTEXT_STR_FOR_TEST", None, raising=False)
    # Stub RAG pipeline so handler reaches build_context and our seam.
    monkeypatch.setattr(app_module, "embed", lambda text: [0.0] * 1536)
    monkeypatch.setattr(app_module, "query_all_namespaces",
                        lambda vec: [_make_fake_match()])
    # Set the member cookie
    client.set_cookie("rm_member_email", "jane@example.com")
    try:
        r = client.post("/chat",
                        json={"query": "What should I do about my fatigue?",
                              "level": "executive"},
                        buffered=True)
    except Exception:
        pass
    captured_ctx = getattr(app_module, "_LAST_CONTEXT_STR_FOR_TEST", None) or ""
    assert "MEMBER CONTEXT" in captured_ctx or "member" in captured_ctx.lower()


def test_chat_free_tier_unchanged_when_no_member_cookie(app_client_member, monkeypatch):
    """No cookie = no member context block in the system prompt."""
    client, app_module, db = app_client_member
    monkeypatch.setattr(app_module, "_LAST_CONTEXT_STR_FOR_TEST", None, raising=False)
    monkeypatch.setattr(app_module, "embed", lambda text: [0.0] * 1536)
    monkeypatch.setattr(app_module, "query_all_namespaces",
                        lambda vec: [_make_fake_match()])
    # NO set_cookie call — visitor is anonymous
    try:
        r = client.post("/chat",
                        json={"query": "Hi", "level": "executive"},
                        buffered=True)
    except Exception:
        pass
    captured_ctx = getattr(app_module, "_LAST_CONTEXT_STR_FOR_TEST", None) or ""
    assert "MEMBER CONTEXT" not in captured_ctx
