"""Unified People hub — Phase 1 feeders.

Covers the pure practitioner mapping, the additive upsert / merge_tags endpoint
behavior, and the contact-type tag filter. Follows the LOG_DB/CONSOLE_SECRET
monkeypatch pattern used across the suite (e.g. test_household_api.py).
"""
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
        pytest.skip(f"app module not importable in this env: {e}")


@pytest.fixture
def app_db(monkeypatch, tmp_path):
    app = _app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    app._init_people_table()  # full schema in the tmp db
    return app, db


def _tags(db, email):
    with sqlite3.connect(db) as cx:
        row = cx.execute("SELECT tags FROM people WHERE email=?", (email,)).fetchone()
    return set(json.loads(row[0])) if row else None


# ── _practitioner_to_person (pure) ────────────────────────────────────────────

def test_engaged_practitioner_tagged_opted_in():
    app = _app()
    p = app._practitioner_to_person({
        "id": "abc", "name": "Jane Doe", "email": "Jane@Clinic.com",
        "portal_role": "licensed", "tier": "panel_in_cert",
    })
    assert p["email"] == "jane@clinic.com"
    assert "type:practitioner" in p["tags"]
    assert "consent:opted-in" in p["tags"]
    assert "type:practitioner-cold" not in p["tags"]
    assert "pract:abc" in p["tags"]


def test_scraped_practitioner_tagged_cold():
    app = _app()
    p = app._practitioner_to_person({
        "id": "xyz", "name": "Cold Lead", "email": "cold@dir.com",
        "portal_role": None, "wholesale_unlocked_at": None, "tier": "org_member",
    })
    assert "type:practitioner-cold" in p["tags"]
    assert "consent:cold-no-consent" in p["tags"]
    assert "type:practitioner" not in p["tags"]


def test_wholesale_unlock_counts_as_engaged():
    app = _app()
    p = app._practitioner_to_person({
        "id": "1", "name": "W", "email": "w@x.com",
        "portal_role": None, "wholesale_unlocked_at": "2026-06-01T00:00:00Z",
    })
    assert "type:practitioner" in p["tags"]
    assert "consent:opted-in" in p["tags"]


def test_practitioner_without_email_is_skipped():
    app = _app()
    assert app._practitioner_to_person({"id": "1", "name": "No Email", "email": ""}) is None
    assert app._practitioner_to_person({"id": "2", "name": "Null"}) is None


# ── merge_tags additive upsert ────────────────────────────────────────────────

def test_merge_tags_unions_without_clobbering(app_db):
    app, db = app_db
    client = app.app.test_client()
    hdr = {"X-Console-Key": "testkey"}
    # seed an existing person (a PB client) with a tag + a city
    client.post("/api/people", json=[{
        "email": "dup@x.com", "city": "Hilo", "tags": ["pb:active"],
    }], headers=hdr)
    # feeder upserts the same email as pr-media, no city provided
    r = client.post("/api/people?merge_tags=1", json=[{
        "email": "dup@x.com",
        "tags": ["type:pr-media", "consent:cold-no-consent"],
    }], headers=hdr)
    assert r.status_code == 200 and r.get_json()["updated"] == 1
    assert _tags(db, "dup@x.com") == {
        "pb:active", "type:pr-media", "consent:cold-no-consent"}
    # scalar not clobbered by the blank
    with sqlite3.connect(db) as cx:
        city = cx.execute("SELECT city FROM people WHERE email=?", ("dup@x.com",)).fetchone()[0]
    assert city == "Hilo"


def test_default_post_overwrites_tags(app_db):
    app, db = app_db
    client = app.app.test_client()
    hdr = {"X-Console-Key": "testkey"}
    client.post("/api/people", json=[{"email": "o@x.com", "tags": ["pb:active"]}], headers=hdr)
    # no merge flag → last write wins (existing behavior preserved)
    client.post("/api/people", json=[{"email": "o@x.com", "tags": ["type:client"]}], headers=hdr)
    assert _tags(db, "o@x.com") == {"type:client"}


def test_merge_is_idempotent(app_db):
    app, db = app_db
    client = app.app.test_client()
    hdr = {"X-Console-Key": "testkey"}
    payload = [{"email": "i@x.com", "tags": ["type:pr-media"]}]
    client.post("/api/people?merge_tags=1", json=payload, headers=hdr)
    client.post("/api/people?merge_tags=1", json=payload, headers=hdr)
    assert _tags(db, "i@x.com") == {"type:pr-media"}  # no dupes


def test_merge_skips_blank_email(app_db):
    app, db = app_db
    client = app.app.test_client()
    r = client.post("/api/people?merge_tags=1", json=[{"email": "", "tags": ["x"]}],
                    headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 0


# ── contact-type filter on GET /api/people ────────────────────────────────────

def test_type_tag_filter(app_db):
    app, db = app_db
    client = app.app.test_client()
    hdr = {"X-Console-Key": "testkey"}
    client.post("/api/people?merge_tags=1", json=[
        {"email": "prac@x.com", "tags": ["type:practitioner-cold"]},
        {"email": "client@x.com", "tags": ["type:client"]},
    ], headers=hdr)
    r = client.get("/api/people?tags=type:practitioner-cold", headers=hdr)
    emails = [p["email"] for p in r.get_json()["people"]]
    assert emails == ["prac@x.com"]


def test_type_param_is_exact_not_substring(app_db):
    """?type=type:practitioner must NOT match type:practitioner-cold."""
    app, db = app_db
    client = app.app.test_client()
    hdr = {"X-Console-Key": "testkey"}
    client.post("/api/people?merge_tags=1", json=[
        {"email": "engaged@x.com", "tags": ["type:practitioner", "consent:opted-in"]},
        {"email": "cold@x.com", "tags": ["type:practitioner-cold", "consent:cold-no-consent"]},
    ], headers=hdr)
    r = client.get("/api/people?type=type:practitioner", headers=hdr)
    assert [p["email"] for p in r.get_json()["people"]] == ["engaged@x.com"]
    r = client.get("/api/people?type=type:practitioner-cold", headers=hdr)
    assert [p["email"] for p in r.get_json()["people"]] == ["cold@x.com"]


# ── cross-source dedup keeps every tag ────────────────────────────────────────

def test_cross_source_tags_union_on_same_email(app_db):
    app, db = app_db
    client = app.app.test_client()
    hdr = {"X-Console-Key": "testkey"}
    # same person arrives from practitioner feeder then media feeder
    client.post("/api/people?merge_tags=1", json=[{
        "email": "both@x.com", "tags": ["type:practitioner", "consent:opted-in"]}],
        headers=hdr)
    client.post("/api/people?merge_tags=1", json=[{
        "email": "both@x.com", "tags": ["type:pr-media"]}], headers=hdr)
    assert _tags(db, "both@x.com") == {
        "type:practitioner", "consent:opted-in", "type:pr-media"}
