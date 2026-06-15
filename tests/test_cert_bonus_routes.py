"""Tests for the cert-bonus Flask endpoints (ships dark behind CERT_BONUS_ENABLED).

Isolation strategy:
  - We monkeypatch appmod.LOG_DB to a per-test tmp sqlite file so the cron sweep
    and commitment store never touch the real shared chat_log.db.
  - We still filter / delete by the test email for robustness.
  - practitioner_portal.modules_completed_for_email is monkeypatched (no Supabase).
"""
import datetime
import os
import sqlite3

import pytest

import app as appmod
from dashboard import cert_bonus
from dashboard import subscriptions as subs

TEST_EMAIL = "doc@x.com"


def _console_key():
    return appmod.CONSOLE_SECRET or os.environ.get("CONSOLE_SECRET", "")


def _cron_secret():
    return os.environ.get("CRON_SECRET") or appmod.CONSOLE_SECRET or os.environ.get("CONSOLE_SECRET", "")


def _three_months_ago():
    """A started_at exactly 3 full calendar months before today → months 1..3 due."""
    today = datetime.date.today().isoformat()
    return subs.add_months(today, -3)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Point the app at a fresh tmp LOG_DB and ensure a console secret exists."""
    p = tmp_path / "chat_log.db"
    monkeypatch.setattr(appmod, "LOG_DB", str(p))
    if not _console_key():
        monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
        monkeypatch.setenv("CONSOLE_SECRET", "test-secret")
    # modules_completed_for_email → 2 (level 1 + 2 due)
    monkeypatch.setattr(appmod._pp, "modules_completed_for_email", lambda email: 2)
    return str(p)


def _cx(db):
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    return cx


# ── /api/cert/commitment ──────────────────────────────────────────────────────

def test_commitment_requires_console_key(db):
    c = appmod.app.test_client()
    r = c.post("/api/cert/commitment", json={"email": TEST_EMAIL, "kind": "pif",
                                             "started_at": "2026-01-15"})
    assert r.status_code == 401, r.data


def test_commitment_set_and_clear(db):
    c = appmod.app.test_client()
    started = _three_months_ago()
    r = c.post("/api/cert/commitment",
               headers={"X-Console-Key": _console_key()},
               json={"email": TEST_EMAIL, "kind": "pif", "started_at": started})
    assert r.status_code == 200, r.data

    cx = _cx(db)
    cb = cert_bonus.get_commitment(cx, TEST_EMAIL)
    assert cb and cb["kind"] == "pif" and cb["active"]
    cx.close()

    r2 = c.post("/api/cert/commitment",
                headers={"X-Console-Key": _console_key()},
                json={"email": TEST_EMAIL, "clear": True})
    assert r2.status_code == 200, r2.data
    cx = _cx(db)
    assert cert_bonus.get_commitment(cx, TEST_EMAIL)["active"] == 0
    cx.close()


# ── /api/cron/biofield-bonuses ──────────────────────────────────────────────────

def _seed_commitment(db):
    cx = _cx(db)
    cert_bonus.init_tables(cx)
    cx.execute("DELETE FROM cert_commitments WHERE email=?", (TEST_EMAIL,))
    cx.execute("DELETE FROM cert_bonus_grants WHERE email=?", (TEST_EMAIL,))
    cx.commit()
    cert_bonus.set_commitment(cx, TEST_EMAIL, kind="pif", started_at=_three_months_ago())
    cx.commit()
    cx.close()


def test_cron_requires_secret(db, monkeypatch):
    monkeypatch.setenv("CERT_BONUS_ENABLED", "1")
    c = appmod.app.test_client()
    assert c.post("/api/cron/biofield-bonuses").status_code == 401
    assert c.post("/api/cron/biofield-bonuses",
                  headers={"X-Cron-Secret": "wrong"}).status_code == 401


def test_cron_grants_and_is_idempotent(db, monkeypatch):
    monkeypatch.setenv("CERT_BONUS_ENABLED", "1")
    _seed_commitment(db)
    c = appmod.app.test_client()

    r = c.post("/api/cron/biofield-bonuses", headers={"X-Cron-Secret": _cron_secret()})
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["granted"] == 5, body  # monthly 1..3 + level 1..2

    cx = _cx(db)
    todos = cx.execute(
        "SELECT * FROM todos WHERE category='biofield-bonus' AND title LIKE ?",
        (f"%{TEST_EMAIL}%",)).fetchall()
    assert len(todos) == 5, [dict(t) for t in todos]
    grants = cert_bonus.granted_pairs(cx, TEST_EMAIL)
    assert len(grants) == 5
    cx.close()

    # Second identical run → no new grants, still 5 todos.
    r2 = c.post("/api/cron/biofield-bonuses", headers={"X-Cron-Secret": _cron_secret()})
    assert r2.status_code == 200, r2.data
    assert r2.get_json()["granted"] == 0
    cx = _cx(db)
    todos2 = cx.execute(
        "SELECT * FROM todos WHERE category='biofield-bonus' AND title LIKE ?",
        (f"%{TEST_EMAIL}%",)).fetchall()
    assert len(todos2) == 5
    cx.close()


def test_cron_dry_run_writes_nothing(db, monkeypatch):
    monkeypatch.setenv("CERT_BONUS_ENABLED", "1")
    _seed_commitment(db)
    c = appmod.app.test_client()

    r = c.post("/api/cron/biofield-bonuses?dry_run=1",
               headers={"X-Cron-Secret": _cron_secret()})
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["granted"] == 5
    assert body["dry_run"] is True

    cx = _cx(db)
    todos = cx.execute(
        "SELECT COUNT(*) AS n FROM todos WHERE category='biofield-bonus' AND title LIKE ?",
        (f"%{TEST_EMAIL}%",)).fetchone()
    assert todos["n"] == 0
    assert cert_bonus.granted_pairs(cx, TEST_EMAIL) == set()
    cx.close()


def test_cron_disabled_grants_nothing(db, monkeypatch):
    monkeypatch.delenv("CERT_BONUS_ENABLED", raising=False)
    _seed_commitment(db)
    c = appmod.app.test_client()
    r = c.post("/api/cron/biofield-bonuses", headers={"X-Cron-Secret": _cron_secret()})
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["granted"] == 0
    cx = _cx(db)
    has_todos = cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='todos'").fetchone()
    if has_todos:
        n = cx.execute(
            "SELECT COUNT(*) AS n FROM todos WHERE category='biofield-bonus' AND title LIKE ?",
            (f"%{TEST_EMAIL}%",)).fetchone()["n"]
        assert n == 0
    assert cert_bonus.granted_pairs(cx, TEST_EMAIL) == set()
    cx.close()
