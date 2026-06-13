"""Phase 3 increment 1 — graduate personal-email engine to opted-in clients.

Segment cohort selection, ramp cap, dry-run, and the unsubscribe/revoke flow.
Network (Pinecone + LLM + SMTP) is stubbed.
"""
import importlib
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest


def _mods():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app"), importlib.import_module("incentive_engine")
    except Exception as e:
        pytest.skip(f"not importable in this env: {e}")


def _seed_tables(db):
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "email TEXT UNIQUE, name TEXT DEFAULT '', auth_method TEXT, created_at TEXT)")
        cx.execute("CREATE TABLE IF NOT EXISTS personal_email_state (user_id INTEGER PRIMARY KEY, "
                   "last_send_at TEXT, last_open_at TEXT, last_click_at TEXT, "
                   "consecutive_no_engagement_days INTEGER DEFAULT 0, topic_engagement_history TEXT, "
                   "topic_send_history TEXT, product_affinity TEXT, paused_until TEXT)")
        cx.execute("CREATE TABLE IF NOT EXISTS personal_email_sends (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "user_id INTEGER, sent_at TEXT, channel TEXT, topic TEXT, product_name TEXT, "
                   "coupon_code TEXT, subject TEXT, body_snippet TEXT, opened_at TEXT, clicked_at TEXT)")
        cx.commit()


def _add_person(db, email, tags):
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO people (email, tags, created_at, updated_at) VALUES (?,?,?,?)",
                   (email, json.dumps(tags), "", ""))
        cx.commit()


@pytest.fixture
def env(monkeypatch, tmp_path):
    app, ie = _mods()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(ie, "LOG_DB", db)
    app._init_people_table()
    _seed_tables(db)
    # stub network: fake Pinecone pool (empty -> fallback topic), LLM, SMTP
    fake = types.ModuleType("pinecone_content_pool")
    fake.candidate_topics_for_audience = lambda audience: []
    fake.fetch_source_text_for_topic = lambda *a, **k: "src"
    monkeypatch.setitem(sys.modules, "pinecone_content_pool", fake)
    sent = []
    monkeypatch.setattr(ie, "generate_personal_email", lambda **k: {"subject": "s", "body": "b"})
    monkeypatch.setattr(ie, "_send_email", lambda user, subj, body: sent.append(user["email"]))
    return app, ie, db, sent


# ── segment cohort ────────────────────────────────────────────────────────────

def test_cohort_only_opted_in_clients(env):
    app, ie, db, sent = env
    _add_person(db, "client@x.com", ["type:client", "consent:opted-in"])
    _add_person(db, "coldclient@x.com", ["type:client", "consent:cold-no-consent"])
    _add_person(db, "prospect@x.com", ["type:prospect", "consent:opted-in"])
    _add_person(db, "unsub@x.com", ["type:client", "consent:opted-in", "consent:unsubscribed"])
    _add_person(db, "bounced@x.com", ["type:client", "consent:opted-in", "email bounced"])
    cohort = ie._list_segment_cohort(("type:client", "consent:opted-in"))
    assert sorted(c["email"] for c in cohort) == ["client@x.com"]


def test_cohort_bootstraps_users(env):
    app, ie, db, sent = env
    _add_person(db, "new@x.com", ["type:client", "consent:opted-in"])
    ie._list_segment_cohort()
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT auth_method FROM users WHERE email=?", ("new@x.com",)).fetchone()
    assert row and row[0] == "segment-bootstrap"


# ── ramp cap + dry-run ────────────────────────────────────────────────────────

def test_cap_limits_sends(env):
    app, ie, db, sent = env
    for i in range(5):
        _add_person(db, f"c{i}@x.com", ["type:client", "consent:opted-in"])
    s = ie.run_daily_send_for_segment(cap=2)
    assert s["sent"] == 2 and s["capped"] is True
    assert len(sent) == 2


def test_dry_run_sends_nothing(env):
    app, ie, db, sent = env
    _add_person(db, "c@x.com", ["type:client", "consent:opted-in"])
    s = ie.run_daily_send_for_segment(cap=10, dry_run=True)
    assert s["sent"] == 1 and sent == []


# ── unsubscribe / revoke ──────────────────────────────────────────────────────

def test_unsub_token_roundtrip(env):
    app, ie, db, sent = env
    t = ie.unsubscribe_token("A@X.com")
    assert ie.verify_unsub_token("a@x.com", t)
    assert not ie.verify_unsub_token("a@x.com", "bad")


def test_revoke_consent_removes_optin_and_excludes(env):
    app, ie, db, sent = env
    _add_person(db, "u@x.com", ["type:client", "consent:opted-in"])
    assert ie.revoke_consent("u@x.com") is True
    with sqlite3.connect(db) as cx:
        tags = set(json.loads(cx.execute("SELECT tags FROM people WHERE email=?", ("u@x.com",)).fetchone()[0]))
    assert "consent:opted-in" not in tags and "consent:unsubscribed" in tags
    assert ie._list_segment_cohort() == []  # now excluded


def test_unsubscribe_endpoint(env):
    app, ie, db, sent = env
    _add_person(db, "e@x.com", ["type:client", "consent:opted-in"])
    c = app.app.test_client()
    tok = ie.unsubscribe_token("e@x.com")
    r = c.get(f"/unsubscribe?email=e@x.com&t={tok}")
    assert r.status_code == 200 and b"unsubscribed" in r.data.lower()
    with sqlite3.connect(db) as cx:
        tags = set(json.loads(cx.execute("SELECT tags FROM people WHERE email=?", ("e@x.com",)).fetchone()[0]))
    assert "consent:unsubscribed" in tags
    assert c.get("/unsubscribe?email=e@x.com&t=bad").status_code == 400
