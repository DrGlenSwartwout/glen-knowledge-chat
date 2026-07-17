"""Tests for detect_household_candidates() — the cluster-finding pass."""

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


def _seed(db_path):
    """Create the people + household_candidates tables."""
    with sqlite3.connect(db_path) as cx:
        cx.execute("""
            CREATE TABLE people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                first_name TEXT DEFAULT '',
                last_name TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                city TEXT DEFAULT '',
                state TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                address1 TEXT DEFAULT '',
                zip TEXT DEFAULT ''
            )
        """)
        cx.execute("""
            CREATE TABLE households (
                id INTEGER PRIMARY KEY, slug TEXT UNIQUE, name TEXT,
                head_person_id INTEGER, address TEXT, notes TEXT,
                created_at TEXT, updated_at TEXT, created_by TEXT
            )
        """)
        cx.execute("""
            CREATE TABLE household_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TEXT NOT NULL, signal TEXT NOT NULL,
                person_ids TEXT NOT NULL, status TEXT DEFAULT 'pending',
                resolved_at TEXT, resolved_by TEXT, household_id INTEGER
            )
        """)
        cx.commit()


def _insert_person(db, email, first="", last="", phone="", city="", state="", tags=None):
    with sqlite3.connect(db) as cx:
        cur = cx.execute(
            "INSERT INTO people (email, first_name, last_name, phone, city, state, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (email, first, last, phone, city, state, json.dumps(tags or []))
        )
        cx.commit()
        return cur.lastrowid


def test_detect_shared_email_signal(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed(tmp_db)
    _insert_person(tmp_db, "share@x.com", first="A", last="One")
    # SQLite TEXT UNIQUE is case-sensitive — different casing allowed
    _insert_person(tmp_db, "Share@x.com", first="B", last="Two")

    summary = app.detect_household_candidates()
    assert summary["new_pending"] >= 1
    with sqlite3.connect(tmp_db) as cx:
        rows = cx.execute("SELECT signal, person_ids FROM household_candidates WHERE status='pending'").fetchall()
    signals = {r[0] for r in rows}
    assert "shared-email" in signals


def test_detect_shared_phone_lastname(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed(tmp_db)
    _insert_person(tmp_db, "a@x.com", first="A", last="Perdomo", phone="+18087562539")
    _insert_person(tmp_db, "b@x.com", first="B", last="Perdomo", phone="+18087562539")
    _insert_person(tmp_db, "c@x.com", first="C", last="Other",   phone="+18087562539")  # different lastname — NOT a cluster

    summary = app.detect_household_candidates()
    with sqlite3.connect(tmp_db) as cx:
        rows = cx.execute("SELECT signal, person_ids FROM household_candidates WHERE status='pending'").fetchall()
    perdomo_clusters = [r for r in rows if r[0] == "shared-phone-lastname"]
    assert len(perdomo_clusters) == 1
    ids = sorted(json.loads(perdomo_clusters[0][1]))
    assert len(ids) == 2   # only the two Perdomos


def test_detect_skips_already_in_household(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed(tmp_db)
    _insert_person(tmp_db, "a@x.com", first="A", last="Smith", phone="+1555", tags=["household:smith"])
    _insert_person(tmp_db, "b@x.com", first="B", last="Smith", phone="+1555")

    summary = app.detect_household_candidates()
    assert summary["new_pending"] == 0
    assert summary["skipped_already_household"] >= 1


def test_detect_dedup_against_dismissed(monkeypatch, tmp_db):
    """A cluster previously dismissed should not re-surface."""
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed(tmp_db)
    p1 = _insert_person(tmp_db, "a@x.com", first="A", last="Jones", phone="+1999")
    p2 = _insert_person(tmp_db, "b@x.com", first="B", last="Jones", phone="+1999")
    with sqlite3.connect(tmp_db) as cx:
        cx.execute(
            "INSERT INTO household_candidates (detected_at, signal, person_ids, status) VALUES (?, ?, ?, ?)",
            ("2026-05-25T00:00:00", "shared-phone-lastname", json.dumps(sorted([p1, p2])), "dismissed")
        )
        cx.commit()

    summary = app.detect_household_candidates()
    assert summary["new_pending"] == 0
    assert summary["skipped_dedup"] >= 1
