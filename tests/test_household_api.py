"""Integration tests for household API endpoints + ghl_update_tags helper.

GHL calls are mocked so tests don't hit the live API. Uses the existing
`tmp_db` fixture from conftest.py + monkeypatching LOG_DB on the app
module, matching the pattern in test_full_report.py.
"""

import importlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
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


def _seed_people_schema(db_path):
    """Create the people table in the test DB with just the columns we use."""
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
                tags TEXT DEFAULT '[]'
            )
        """)
        cx.commit()


def _seed_household_tables(db_path):
    """Create the household tables in the test DB."""
    with sqlite3.connect(db_path) as cx:
        cx.execute("""
            CREATE TABLE households (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                head_person_id INTEGER,
                address TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by TEXT NOT NULL
            )
        """)
        cx.execute("""
            CREATE TABLE household_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TEXT NOT NULL,
                signal TEXT NOT NULL,
                person_ids TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_at TEXT DEFAULT '',
                resolved_by TEXT DEFAULT '',
                household_id INTEGER
            )
        """)
        cx.commit()


def _seed_person(db_path, email, first="", last="", phone="", city="", state="", tags=None):
    """Insert a person row and return its id."""
    with sqlite3.connect(db_path) as cx:
        cur = cx.execute(
            "INSERT INTO people (email, first_name, last_name, phone, city, state, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (email, first, last, phone, city, state, json.dumps(tags or []))
        )
        cx.commit()
        return cur.lastrowid


def test_ghl_update_tags_add_calls_lookup_and_put(monkeypatch, tmp_db):
    """ghl_update_tags(email, add={...}) looks up contact, merges tag set, PUTs."""
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)

    captured = {}

    def fake_ghl_get(path, params=None):
        captured["get_path"] = path
        captured["get_params"] = params
        return {"contacts": [{"id": "C123", "email": "test@x.com", "tags": ["existing"]}]}, None

    def fake_ghl_put(path, payload):
        captured["put_path"] = path
        captured["put_payload"] = payload
        return {}, None

    monkeypatch.setattr(app, "_ghl_get", fake_ghl_get)
    monkeypatch.setattr(app, "_ghl_put", fake_ghl_put)
    monkeypatch.setattr(app, "GHL_API_KEY", "fake-key")

    contact_id, err = app.ghl_update_tags("test@x.com", add={"household:smith"})
    assert err is None
    assert contact_id == "C123"
    assert captured["get_path"] == "/contacts/lookup"
    assert captured["put_path"] == "/contacts/C123"
    assert set(captured["put_payload"]["tags"]) == {"existing", "household:smith"}


def test_ghl_update_tags_remove_subtracts_from_existing(monkeypatch, tmp_db):
    """ghl_update_tags(email, remove={...}) subtracts tags before PUT."""
    app = _app()

    def fake_ghl_get(path, params=None):
        return {"contacts": [{"id": "C456", "email": "test@x.com", "tags": ["keep", "household:old"]}]}, None

    captured = {}
    def fake_ghl_put(path, payload):
        captured["payload"] = payload
        return {}, None

    monkeypatch.setattr(app, "_ghl_get", fake_ghl_get)
    monkeypatch.setattr(app, "_ghl_put", fake_ghl_put)
    monkeypatch.setattr(app, "GHL_API_KEY", "fake-key")

    contact_id, err = app.ghl_update_tags("test@x.com", remove={"household:old"})
    assert err is None
    assert contact_id == "C456"
    assert set(captured["payload"]["tags"]) == {"keep"}


def test_ghl_update_tags_falls_through_to_upsert_when_no_contact(monkeypatch, tmp_db):
    """If lookup returns empty, fall through to ghl_upsert_contact so the
    contact gets created with the add tags."""
    app = _app()

    def fake_ghl_get(path, params=None):
        return {"contacts": []}, None

    captured = {}
    def fake_upsert(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None):
        captured["upsert_call"] = {"email": email, "extra_tags": list(extra_tags or [])}
        return "C789", True, None

    monkeypatch.setattr(app, "_ghl_get", fake_ghl_get)
    monkeypatch.setattr(app, "ghl_upsert_contact", fake_upsert)
    monkeypatch.setattr(app, "GHL_API_KEY", "fake-key")

    contact_id, err = app.ghl_update_tags("new@x.com", add={"household:smith"})
    assert err is None
    assert contact_id == "C789"
    assert captured["upsert_call"]["email"] == "new@x.com"
    assert "household:smith" in captured["upsert_call"]["extra_tags"]


def test_create_household_writes_db_and_tags(monkeypatch, tmp_db):
    """POST /api/households creates the household row, tags every member's
    people.tags JSON, and tags the head with household-head: too."""
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db)
    _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")   # disables GHL calls

    pid_lotika = _seed_person(tmp_db, "lotika@x.com", first="Lotika", last="Savant")
    pid_omika  = _seed_person(tmp_db, "omika@x.com",  first="Omika",  last="Savant")

    client = app.app.test_client()
    r = client.post("/api/households",
                    headers={"X-Console-Key": "testkey"},
                    json={"name": "Savant",
                          "head_person_id": pid_lotika,
                          "member_person_ids": [pid_lotika, pid_omika]})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    slug = body["household"]["slug"]
    assert slug == "savant"

    # DB row exists
    with sqlite3.connect(tmp_db) as cx:
        row = cx.execute("SELECT name, head_person_id FROM households WHERE slug=?", (slug,)).fetchone()
        assert row == ("Savant", pid_lotika)
        # Both members carry household:savant; head also has household-head:savant
        lotika_tags = json.loads(cx.execute("SELECT tags FROM people WHERE id=?", (pid_lotika,)).fetchone()[0])
        omika_tags  = json.loads(cx.execute("SELECT tags FROM people WHERE id=?", (pid_omika,)).fetchone()[0])
        assert "household:savant" in lotika_tags
        assert "household-head:savant" in lotika_tags
        assert "household:savant" in omika_tags
        assert "household-head:savant" not in omika_tags


def test_create_household_rejects_member_already_in_household(monkeypatch, tmp_db):
    """409 when a member is already in another household."""
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db)
    _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")

    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith", tags=["household:smith-old"])
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")

    client = app.app.test_client()
    r = client.post("/api/households",
                    headers={"X-Console-Key": "testkey"},
                    json={"name": "Smith", "head_person_id": p1, "member_person_ids": [p1, p2]})
    assert r.status_code == 409
    body = r.get_json()
    assert "current_household" in body
    assert body["current_household"]["slug"] == "smith-old"


def test_create_household_strips_relationship_family_shared_email_tag(monkeypatch, tmp_db):
    """Members carrying the legacy relationship:family-shared-email tag have
    it stripped on household creation (the new household: tags supersede it)."""
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db)
    _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")

    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Jones", tags=["relationship:family-shared-email", "client"])
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Jones", tags=["relationship:family-shared-email"])

    client = app.app.test_client()
    r = client.post("/api/households",
                    headers={"X-Console-Key": "testkey"},
                    json={"name": "Jones", "head_person_id": p1, "member_person_ids": [p1, p2]})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        for pid in (p1, p2):
            tags = json.loads(cx.execute("SELECT tags FROM people WHERE id=?", (pid,)).fetchone()[0])
            assert "relationship:family-shared-email" not in tags
        # Non-household tags preserved
        p1_tags = json.loads(cx.execute("SELECT tags FROM people WHERE id=?", (p1,)).fetchone()[0])
        assert "client" in p1_tags


def _create_test_household(client, name, head_id, member_ids):
    """Helper that POSTs and returns the parsed body."""
    r = client.post("/api/households", headers={"X-Console-Key": "testkey"},
                    json={"name": name, "head_person_id": head_id, "member_person_ids": member_ids})
    assert r.status_code == 200
    return r.get_json()


def test_get_households_lists_all(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")

    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    p3 = _seed_person(tmp_db, "c@x.com", first="C", last="Jones")
    p4 = _seed_person(tmp_db, "d@x.com", first="D", last="Jones")

    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1, p2])
    _create_test_household(client, "Jones", p3, [p3, p4])

    r = client.get("/api/households", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    body = r.get_json()
    slugs = {h["slug"] for h in body["households"]}
    assert slugs == {"smith", "jones"}
    smith = next(h for h in body["households"] if h["slug"] == "smith")
    assert smith["member_count"] == 2
    assert smith["head"]["id"] == p1


def test_get_household_detail_returns_members(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="Lotika", last="Savant")
    p2 = _seed_person(tmp_db, "b@x.com", first="Omika",  last="Savant")
    client = app.app.test_client()
    _create_test_household(client, "Savant", p1, [p1, p2])

    r = client.get("/api/households/savant", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["slug"] == "savant"
    assert body["name"] == "Savant"
    assert len(body["members"]) == 2
    head = next(m for m in body["members"] if m["is_head"])
    assert head["id"] == p1


def test_get_person_household_returns_household_when_member(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    p3 = _seed_person(tmp_db, "c@x.com", first="C", last="Other")
    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1, p2])

    r = client.get(f"/api/people/{p1}/household", headers={"X-Console-Key": "testkey"})
    body = r.get_json()
    assert body["household"]["slug"] == "smith"
    r = client.get(f"/api/people/{p3}/household", headers={"X-Console-Key": "testkey"})
    body = r.get_json()
    assert body["household"] is None


def test_get_household_candidates_returns_pending_only(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")

    with sqlite3.connect(tmp_db) as cx:
        cx.execute("INSERT INTO household_candidates (detected_at, signal, person_ids, status) VALUES (?, ?, ?, ?)",
                   ("2026-05-26T00:00:00", "shared-email", json.dumps([1, 2]), "pending"))
        cx.execute("INSERT INTO household_candidates (detected_at, signal, person_ids, status) VALUES (?, ?, ?, ?)",
                   ("2026-05-26T00:00:00", "shared-email", json.dumps([3, 4]), "dismissed"))
        cx.commit()

    client = app.app.test_client()
    r = client.get("/api/household-candidates?status=pending", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["signal"] == "shared-email"


def test_patch_household_renames_keeping_slug(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1, p2])

    r = client.patch("/api/households/smith", headers={"X-Console-Key": "testkey"},
                     json={"name": "Smith Family"})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        row = cx.execute("SELECT slug, name FROM households WHERE id=1").fetchone()
        assert row == ("smith", "Smith Family")   # slug never changes


def test_patch_household_changes_head_moves_tag(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1, p2])

    r = client.patch("/api/households/smith", headers={"X-Console-Key": "testkey"},
                     json={"head_person_id": p2})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        t1 = set(json.loads(cx.execute("SELECT tags FROM people WHERE id=?", (p1,)).fetchone()[0]))
        t2 = set(json.loads(cx.execute("SELECT tags FROM people WHERE id=?", (p2,)).fetchone()[0]))
    assert "household-head:smith" not in t1
    assert "household-head:smith" in t2


def test_add_member_with_conflict_returns_409(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    p3 = _seed_person(tmp_db, "c@x.com", first="C", last="Jones")
    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1])
    _create_test_household(client, "Jones", p3, [p3])

    # p3 already in jones — adding to smith must 409
    r = client.post("/api/households/smith/members", headers={"X-Console-Key": "testkey"},
                    json={"person_id": p3})
    assert r.status_code == 409
    assert r.get_json()["current_household"]["slug"] == "jones"


def test_remove_member_strips_tag(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1, p2])

    r = client.delete(f"/api/households/smith/members/{p2}", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        t2 = set(json.loads(cx.execute("SELECT tags FROM people WHERE id=?", (p2,)).fetchone()[0]))
    assert "household:smith" not in t2


def test_remove_head_blocked_with_409(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1, p2])

    r = client.delete(f"/api/households/smith/members/{p1}", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 409
    assert "change head first" in r.get_json()["error"].lower()


def test_disband_removes_tags_and_household_row(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1, p2])

    r = client.delete("/api/households/smith", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM households WHERE slug=?", ("smith",)).fetchone()[0] == 0
        for pid in (p1, p2):
            tags = set(json.loads(cx.execute("SELECT tags FROM people WHERE id=?", (pid,)).fetchone()[0]))
            assert "household:smith" not in tags
            assert "household-head:smith" not in tags


def test_resync_ghl_pushes_household_tags_to_all_members(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Smith")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Smith")
    client = app.app.test_client()
    _create_test_household(client, "Smith", p1, [p1, p2])

    calls = []
    def fake_update(email, add=None, remove=None):
        calls.append({"email": email, "add": sorted(add or []), "remove": sorted(remove or [])})
        return "C", None
    monkeypatch.setattr(app, "ghl_update_tags", fake_update)

    r = client.post("/api/households/smith/resync-ghl", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    emails = {c["email"] for c in calls}
    assert emails == {"a@x.com", "b@x.com"}
    head_call = next(c for c in calls if c["email"] == "a@x.com")
    assert "household-head:smith" in head_call["add"]
    assert "household:smith" in head_call["add"]
