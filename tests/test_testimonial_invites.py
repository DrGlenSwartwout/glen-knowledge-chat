"""Phase 4: positive-sentiment detector + invite-candidate store."""
import sqlite3
from dashboard import testimonial_signals as ts
from dashboard import testimonial_invites as ti
from dashboard import product_reviews as pr


def _complete(payload):
    return lambda system, user: payload


# ── classify_positive_result ──────────────────────────────────────────────────

def test_classify_positive():
    out = ts.classify_positive_result("my energy is so much better since starting",
        _complete('{"positive": true, "confidence": 0.85, "quote": "my energy is so much better", "kind": "remedy"}'))
    assert out["positive"] is True and out["confidence"] == 0.85
    assert out["quote"] == "my energy is so much better" and out["kind"] == "remedy"


def test_classify_negative_or_neutral():
    out = ts.classify_positive_result("when will my order ship?",
        _complete('{"positive": false, "confidence": 0.1, "quote": "", "kind": "general"}'))
    assert out["positive"] is False


def test_classify_empty_and_bad_json_safe_default():
    assert ts.classify_positive_result("", _complete("anything"))["positive"] is False
    out = ts.classify_positive_result("text", _complete("not json"))
    assert out == {"positive": False, "confidence": 0.0, "quote": "", "kind": "general"}


def test_classify_clamps_confidence_and_kind():
    out = ts.classify_positive_result("x",
        _complete('{"positive": true, "confidence": 5, "quote": "q", "kind": "bogus"}'))
    assert out["confidence"] == 1.0 and out["kind"] == "general"


# ── candidate store ───────────────────────────────────────────────────────────

def _cx():
    return sqlite3.connect(":memory:")


def test_upsert_and_pending_queue():
    cx = _cx()
    cid = ti.upsert_candidate(cx, "A@x.com", "Ann", "feeling great", "chat", "remedy", 0.8)
    assert isinstance(cid, int)
    q = ti.pending_queue(cx)
    assert len(q) == 1 and q[0]["email"] == "a@x.com" and q[0]["quote"] == "feeling great"
    assert ti.pending_count(cx) == 1
    # re-detect updates the same row (UNIQUE email)
    cid2 = ti.upsert_candidate(cx, "a@x.com", "Ann", "even better now", "journal", "general", 0.9)
    assert cid2 == cid and ti.pending_count(cx) == 1
    assert ti.get(cx, cid)["quote"] == "even better now"


def test_set_status_dismiss_and_send():
    cx = _cx()
    c1 = ti.upsert_candidate(cx, "a@x.com", "A", "q", "chat", "general", 0.7)
    ti.set_status(cx, c1, "dismissed", by="Glen")
    assert ti.get(cx, c1)["status"] == "dismissed" and ti.pending_count(cx) == 0
    c2 = ti.upsert_candidate(cx, "b@x.com", "B", "q", "chat", "general", 0.7)
    ti.set_status(cx, c2, "sent", by="Glen", sent=True)
    row = ti.get(cx, c2)
    assert row["status"] == "sent" and row["sent_at"]


def test_should_skip_already_submitted():
    cx = _cx()
    pr.upsert_review(cx, "_results", "sub@x.com", "S", 5, body="b", kind="testimonial")
    assert ti.should_skip(cx, "sub@x.com") is True
    assert ti.should_skip(cx, "fresh@x.com") is False


def test_should_skip_pending_and_cooldown():
    cx = _cx()
    ti.upsert_candidate(cx, "pend@x.com", "P", "q", "chat", "general", 0.7)
    assert ti.should_skip(cx, "pend@x.com") is True            # pending exists
    c = ti.upsert_candidate(cx, "sent@x.com", "S", "q", "chat", "general", 0.7)
    ti.set_status(cx, c, "sent", by="Glen", sent=True)
    assert ti.should_skip(cx, "sent@x.com") is True            # within cooldown
    assert ti.should_skip(cx, "sent@x.com", cooldown_days=0) is False  # cooldown elapsed
    assert ti.should_skip(cx, "") is True                      # blank email


# ── send + console actions ────────────────────────────────────────────────────

def test_send_invite_email_link_and_quote():
    calls = []
    ok = ti.send_invite_email("a@x.com", "Ann", quote="my energy is back",
                              send=lambda to, subj, body, **k: calls.append((to, subj, body)))
    assert ok and "/results" in calls[0][2] and "my energy is back" in calls[0][2]
    assert "—" not in calls[0][2]  # voice: no em dash
    assert ti.send_invite_email("", "X", send=lambda *a, **k: None) is False


def test_approve_action_sends_and_marks_sent(monkeypatch):
    from dashboard import testimonial_invite_actions as tia
    from dashboard.rbac import Actor, OWNER
    cx = _cx()
    cid = ti.upsert_candidate(cx, "a@x.com", "A", "q", "chat", "general", 0.8)
    sent = []
    monkeypatch.setattr(ti, "send_invite_email", lambda email, name, **k: sent.append(email) or True)
    res = tia._exec_approve({"id": cid}, {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    assert res["status"] == "sent" and sent == ["a@x.com"]
    assert ti.get(cx, cid)["status"] == "sent"


def test_approve_send_failure_keeps_approved(monkeypatch):
    from dashboard import testimonial_invite_actions as tia
    from dashboard.rbac import Actor, OWNER
    cx = _cx()
    cid = ti.upsert_candidate(cx, "a@x.com", "A", "q", "chat", "general", 0.8)
    monkeypatch.setattr(ti, "send_invite_email", lambda *a, **k: False)
    res = tia._exec_approve({"id": cid}, {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    assert res["status"] == "approved" and res["sent"] is False


def test_dismiss_action():
    from dashboard import testimonial_invite_actions as tia
    from dashboard.rbac import Actor, OWNER
    cx = _cx()
    cid = ti.upsert_candidate(cx, "a@x.com", "A", "q", "chat", "general", 0.8)
    tia._exec_dismiss({"id": cid}, {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    assert ti.get(cx, cid)["status"] == "dismissed"


# ── scan endpoint glue (needs app import; run under doppler) ───────────────────

import importlib


def _reload_app(monkeypatch, tmp_path, invites="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TESTIMONIALS_ENABLED", "true")
    monkeypatch.setenv("TESTIMONIAL_INVITES_ENABLED", invites)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_recent_active_emails_window(monkeypatch, tmp_path):
    """Recent (within window) emails are found; old ones excluded. Guards the ts-format
    comparison (stored 'YYYY-MM-DD HH:MM:SS' must compare correctly)."""
    appmod = _reload_app(monkeypatch, tmp_path)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS query_log (id INTEGER PRIMARY KEY, email TEXT, query TEXT, ts TEXT)")
        cx.execute("INSERT INTO query_log (email, query, ts) VALUES (?,?,datetime('now'))", ("active@x.com", "hi"))
        cx.execute("INSERT INTO query_log (email, query, ts) VALUES (?,?,datetime('now','-30 days'))", ("old@x.com", "hi"))
        cx.commit()
        emails = appmod._recent_active_emails(cx, days=7)
    assert "active@x.com" in emails and "old@x.com" not in emails


def test_recent_active_emails_includes_email_feedback(monkeypatch, tmp_path):
    """personal_email_feedback (joined to users) is part of the scan population."""
    appmod = _reload_app(monkeypatch, tmp_path)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT)")
        cx.execute("CREATE TABLE IF NOT EXISTS personal_email_feedback "
                   "(id INTEGER PRIMARY KEY, user_id INTEGER, received_at TEXT, ai_summary TEXT)")
        cx.execute("INSERT INTO users (id,email,created_at) "
                   "VALUES (1,'fb@x.com',datetime('now')),(2,'oldfb@x.com',datetime('now'))")
        cx.execute("INSERT INTO personal_email_feedback (user_id,received_at,ai_summary) "
                   "VALUES (1, datetime('now'), 'feeling great'), (2, datetime('now','-200 days'), 'old')")
        cx.commit()
        emails = appmod._recent_active_emails(cx, days=90)
    assert "fb@x.com" in emails and "oldfb@x.com" not in emails


def test_scan_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, invites="false")
    assert appmod.app.test_client().post("/api/console/testimonial-invites/scan").status_code == 404
    assert appmod.app.test_client().get("/console/testimonial-invites").status_code == 404


def test_scan_dry_run_then_create_and_list(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "")  # pass-through
    monkeypatch.setattr(appmod, "_recent_active_emails", lambda cx, **k: ["happy@x.com", "calm@x.com"])
    monkeypatch.setattr(appmod, "_gather_comms_text", lambda cx, e, **k: "comms for " + e)
    monkeypatch.setattr(appmod, "_ts_complete",
                        lambda s, u: '{"positive": true, "confidence": 0.9, "quote": "so much better", "kind": "remedy"}')
    c = appmod.app.test_client()
    dry = c.post("/api/console/testimonial-invites/scan?dry_run=1").get_json()
    assert dry["dry_run"] is True and len(dry["candidates"]) == 2
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert ti.pending_count(cx) == 0  # dry run persists nothing
    res = c.post("/api/console/testimonial-invites/scan").get_json()
    assert len(res["candidates"]) == 2
    listed = c.get("/api/console/testimonial-invites").get_json()["pending"]
    assert sorted(r["email"] for r in listed) == ["calm@x.com", "happy@x.com"]


def test_scan_includes_gmail_source(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "")
    monkeypatch.setattr(appmod, "_recent_active_emails", lambda cx, **k: [])   # no DB comms
    monkeypatch.setattr(appmod, "_gather_comms_text", lambda cx, e, **k: "")
    monkeypatch.setattr(appmod, "_gmail_client_texts",
                        lambda cx, days: {"gm@x.com": "thank you, I feel amazing"})
    monkeypatch.setattr(appmod, "_ts_complete",
                        lambda s, u: '{"positive": true, "confidence": 0.9, "quote": "I feel amazing", "kind": "remedy"}')
    c = appmod.app.test_client()
    res = c.post("/api/console/testimonial-invites/scan").get_json()
    assert res["gmail_clients"] == 1
    assert len(res["candidates"]) == 1 and res["candidates"][0]["source"] == "gmail"
    listed = c.get("/api/console/testimonial-invites").get_json()["pending"]
    assert any(r["email"] == "gm@x.com" and r["source"] == "gmail" for r in listed)
