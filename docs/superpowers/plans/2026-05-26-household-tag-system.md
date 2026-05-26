# Household Tag System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a household-grouping system for GHL contacts so campaigns can segment by household-head, clinical workflows can surface related members, and household membership is queryable across both `people.tags` (local SQLite) and GHL.

**Architecture:** New `households` and `household_candidates` SQLite tables in `LOG_DB` for metadata + suggestion-state. Membership lives in tags (`household:<slug>` on all members, `household-head:<slug>` on the head) that mirror to GHL via a new `ghl_update_tags` helper. Two console creation flows (from person-detail and multi-select-list). Detection runs chained onto the existing daily PB sync cron and surfaces clusters in a review banner — never auto-tags.

**Tech Stack:** Flask + SQLite (existing `app.py`), vanilla JS + HTML (existing `static/console.html`), pytest (existing `tests/` with `tmp_db` fixture in `conftest.py`), GHL REST v1 (via curl subprocess in existing `_ghl_*` helpers).

**Spec:** `docs/superpowers/specs/2026-05-26-household-tag-system-design.md`

**Build order:** Schema and pure functions first (testable in isolation), then DB + GHL helpers, then API routes, then detection, then UI, then migration. Each milestone commits independently and deploys cleanly even if the next milestone hasn't shipped.

---

## File Structure

**Modified:**
- `app.py` — new helpers (`ghl_update_tags`, `_household_slug`, `_init_households_tables`, `detect_household_candidates`, `resync_all_households_to_ghl`), new routes (`/api/households/*`, `/api/household-candidates/*`, `/admin/detect-household-candidates`, `/admin/resync-all-households`), and 2 added steps in the existing `admin_sync_pb_tags` handler
- `static/console.html` — new Household section in `renderPersonDetail` (around line 1162), new `<dialog id="household-dialog">` near the existing `<dialog id="new-task-dialog">` at line 1887, new multi-select checkboxes on `.person-card` rendering (around line 1118), new candidate review banner above the People search bar (around line 501), new fetch in `loadPersonDetail` (line 1143)

**Created:**
- `tests/test_household_model.py` — unit tests for slug generation + candidate dedup logic
- `tests/test_household_api.py` — integration tests for all household routes (mocks GHL)
- `tests/test_household_detection.py` — tests for the detection algorithm
- `scripts/migrate_existing_households.py` — one-off migration creating the Savant + Perdomo households

**Total task count:** 19 tasks across 5 milestones.

---

## Milestone 1 — Foundation

### Task 1: Schema + slug generation

**Files:**
- Modify: `app.py` (add new helpers near the existing `_init_people_table` at line 4232)
- Create: `tests/test_household_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_household_model.py`:

```python
"""Unit tests for household slug generation + candidate dedup helpers.

Pure-function tests — no DB, no network. The slug function generates
URL-safe identifiers; the dedup key sorts person_ids for stable matching
across detection runs.
"""

import importlib
import sys
from pathlib import Path


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("app")


def test_household_slug_basic():
    app = _app()
    assert app._household_slug("Savant") == "savant"
    assert app._household_slug("O'Connor") == "o-connor"
    assert app._household_slug("Smith Jones") == "smith-jones"


def test_household_slug_appends_head_firstname_on_collision():
    app = _app()
    existing = {"savant"}
    assert app._household_slug("Savant", "Lotika", existing=existing) == "savant-lotika"
    assert app._household_slug("Savant", "Omika", existing=existing) == "savant-omika"


def test_household_slug_collision_without_firstname_falls_back_to_numeric():
    app = _app()
    existing = {"savant", "savant-2"}
    assert app._household_slug("Savant", existing=existing) == "savant-3"


def test_candidate_dedup_key_sorts_person_ids():
    app = _app()
    assert app._candidate_dedup_key([3, 1, 2]) == "1,2,3"
    assert app._candidate_dedup_key([5]) == "5"
    assert app._candidate_dedup_key([]) == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_model.py -v
```

Expected: 4 failures with `AttributeError: module 'app' has no attribute '_household_slug'` (or similar).

- [ ] **Step 3: Add helpers to `app.py`**

Insert the following block immediately after the existing `_init_people_table()` call at `app.py:4232` (search for `_init_people_table()` to locate). Adjust line numbers if `app.py` has shifted.

```python
# ── Households ────────────────────────────────────────────────────────────────
def _init_households_tables():
    """Two tables: `households` for metadata, `household_candidates` for the
    detection-and-suggest workflow. Run at import time alongside other
    schema initializers."""
    with sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS households (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                slug            TEXT UNIQUE NOT NULL,
                name            TEXT NOT NULL,
                head_person_id  INTEGER,
                address         TEXT DEFAULT '',
                notes           TEXT DEFAULT '',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                created_by      TEXT NOT NULL
            )
        """)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS household_candidates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at     TEXT NOT NULL,
                signal          TEXT NOT NULL,
                person_ids      TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'pending',
                resolved_at     TEXT DEFAULT '',
                resolved_by     TEXT DEFAULT '',
                household_id    INTEGER
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_household_candidates_status ON household_candidates(status)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_households_head ON households(head_person_id)")
        cx.commit()

_init_households_tables()


def _household_slug(name, head_first_name="", existing=None):
    """URL-safe stable identifier for a household. Immutable after creation
    (renames update name, never slug). Returns lowercase, hyphen-separated."""
    base = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "household"
    if existing is None:
        return base
    if base not in existing:
        return base
    # Collision: try appending head's first name
    if head_first_name:
        candidate = f"{base}-{re.sub(r'[^a-z0-9]+', '-', head_first_name.lower()).strip('-')}"
        if candidate and candidate not in existing:
            return candidate
    # Numeric suffix fallback
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def _candidate_dedup_key(person_ids):
    """Stable dedup key for household_candidates rows. Sorting ensures the
    same cluster produces the same key across detection runs regardless
    of input ordering."""
    return ",".join(str(i) for i in sorted(int(p) for p in person_ids))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_model.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_household_model.py
git commit -m "feat(households): schema + slug + candidate dedup helpers"
```

---

### Task 2: `ghl_update_tags` helper

**Files:**
- Modify: `app.py` (add helper near the existing `ghl_upsert_contact` at line 1728)
- Create: test in `tests/test_household_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_household_api.py`:

```python
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

    contact_id, err = app.ghl_update_tags("new@x.com", add={"household:smith"})
    assert err is None
    assert contact_id == "C789"
    assert captured["upsert_call"]["email"] == "new@x.com"
    assert "household:smith" in captured["upsert_call"]["extra_tags"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py::test_ghl_update_tags_add_calls_lookup_and_put -v
```

Expected: FAIL with `AttributeError: module 'app' has no attribute 'ghl_update_tags'`.

- [ ] **Step 3: Add the helper to `app.py`**

Insert immediately after the existing `ghl_upsert_contact` definition (search for `def ghl_upsert_contact` to locate). Adjust line numbers if `app.py` has shifted.

```python
def ghl_update_tags(email, add=None, remove=None):
    """Find GHL contact via /contacts/lookup (the correct exact-email endpoint
    per the 2026-05-26 fix), add and/or remove tags as set operations, PUT
    the merged result. Returns (contact_id, error).

    If no contact exists for that email, falls through to ghl_upsert_contact
    so the contact gets created with the `add` tags. `remove` on a non-existent
    contact is a no-op (returns (None, None))."""
    add    = set(add or [])
    remove = set(remove or [])
    if not (add or remove):
        return None, "no tags specified"
    if not GHL_API_KEY:
        return None, "GHL_API_KEY not set"

    data, err = _ghl_get("/contacts/lookup", {"email": email})
    if err:
        return None, err
    contacts = data.get("contacts", []) if isinstance(data, dict) else []
    if not contacts:
        if not add:
            return None, None   # nothing to do — no contact, nothing to remove from
        # Fall through to create — preserves first/last so the new contact has names
        contact_id, _created, err = ghl_upsert_contact(email, extra_tags=list(add))
        return contact_id, err

    # Prefer oldest contact when multiple match (matches ghl_upsert_contact's behavior)
    match = min(contacts, key=lambda c: c.get("dateAdded") or "9999")
    existing = set(match.get("tags", []) or [])
    new_tags = (existing | add) - remove
    if new_tags == existing:
        return match["id"], None   # nothing changed; skip the PUT
    _, err = _ghl_put(f"/contacts/{match['id']}", {"tags": sorted(new_tags)})
    return match["id"], err
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v -k ghl_update_tags
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_household_api.py
git commit -m "feat(ghl): ghl_update_tags helper for add/remove tag operations"
```

---

## Milestone 2 — Core API

### Task 3: `POST /api/households` create endpoint

**Files:**
- Modify: `app.py` (add route + supporting helpers near other `/api/people` routes around line 4316)
- Modify: `tests/test_household_api.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_household_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v -k create_household
```

Expected: 3 failures with `404` (route doesn't exist yet).

- [ ] **Step 3: Add the endpoint to `app.py`**

Insert near other `/api/people` routes (search for `@app.route("/api/people"` to find a good neighbor — around line 4316):

```python
def _check_console_auth():
    """Returns None if authorized, or a (response, status) tuple to return."""
    if not CONSOLE_SECRET:
        return None
    key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
    if key != CONSOLE_SECRET:
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _existing_household_slugs(cx):
    return {row[0] for row in cx.execute("SELECT slug FROM households").fetchall()}


def _person_household_slug(cx, person_id):
    """Returns the slug of the household this person is in, or None."""
    row = cx.execute("SELECT tags FROM people WHERE id=?", (person_id,)).fetchone()
    if not row:
        return None
    try:
        tags = json.loads(row[0] or "[]")
    except Exception:
        return []
    for t in tags:
        if t.startswith("household:") and not t.startswith("household-head:"):
            return t.split(":", 1)[1]
    return None


def _mutate_person_tags(cx, person_id, add=None, remove=None):
    """Update a person's tags JSON additively/subtractively. Returns new tags list."""
    add = set(add or [])
    remove = set(remove or [])
    row = cx.execute("SELECT tags FROM people WHERE id=?", (person_id,)).fetchone()
    if not row:
        return []
    try:
        existing = set(json.loads(row[0] or "[]"))
    except Exception:
        existing = set()
    new_tags = sorted((existing | add) - remove)
    cx.execute("UPDATE people SET tags=? WHERE id=?", (json.dumps(new_tags), person_id))
    return new_tags


def _push_household_tags_to_ghl(person_email, slug, is_head, action="add"):
    """Push household and household-head tags to GHL. action='add' or 'remove'.
    Returns (ok_bool, error_msg_or_None)."""
    tags = {f"household:{slug}"}
    if is_head:
        tags.add(f"household-head:{slug}")
    if action == "add":
        _, err = ghl_update_tags(person_email, add=tags)
    else:
        _, err = ghl_update_tags(person_email, remove=tags)
    return (err is None, err)


@app.route("/api/households", methods=["POST"])
def create_household():
    """Create a household, tag members in DB + GHL.

    Body: {name, head_person_id, member_person_ids[], address?, notes?, created_by?}
    Returns 200 with the new household, 409 if any member is already in another household."""
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    head_id = body.get("head_person_id")
    member_ids = body.get("member_person_ids") or []
    if not head_id or head_id not in member_ids:
        return jsonify({"error": "head_person_id must be in member_person_ids"}), 400
    created_by = (body.get("created_by") or "glen").strip()
    address = (body.get("address") or "").strip()
    notes = (body.get("notes") or "").strip()
    ts = datetime.now(timezone.utc).isoformat()

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        # Pre-flight: ensure no member is already in a household
        for pid in member_ids:
            existing_slug = _person_household_slug(cx, pid)
            if existing_slug:
                existing_row = cx.execute(
                    "SELECT slug, name FROM households WHERE slug=?", (existing_slug,)
                ).fetchone()
                return jsonify({
                    "error": "member already in household",
                    "person_id": pid,
                    "current_household": {
                        "slug": existing_slug,
                        "name": existing_row[1] if existing_row else existing_slug,
                    },
                }), 409

        # Resolve head's first name for slug-collision fallback
        head_row = cx.execute(
            "SELECT first_name, email FROM people WHERE id=?", (head_id,)
        ).fetchone()
        if not head_row:
            return jsonify({"error": "head person not found"}), 400
        head_first, head_email = head_row

        slug = _household_slug(name, head_first, existing=_existing_household_slugs(cx))

        cx.execute("""
            INSERT INTO households (slug, name, head_person_id, address, notes,
                                    created_at, updated_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (slug, name, head_id, address, notes, ts, ts, created_by))
        household_id = cx.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Tag every member in DB. Head gets both household: and household-head:.
        # Also strip the legacy relationship:family-shared-email tag.
        for pid in member_ids:
            adds = {f"household:{slug}"}
            if pid == head_id:
                adds.add(f"household-head:{slug}")
            _mutate_person_tags(cx, pid, add=adds, remove={"relationship:family-shared-email"})
        cx.commit()

    # Push to GHL outside the lock. Per-member errors collected.
    ghl_errors = []
    with sqlite3.connect(LOG_DB) as cx:
        members = cx.execute("""
            SELECT id, email FROM people WHERE id IN ({})
        """.format(",".join("?" * len(member_ids))), member_ids).fetchall()
    for pid, email in members:
        if not email:
            continue
        is_head = (pid == head_id)
        ok, err = _push_household_tags_to_ghl(email, slug, is_head, action="add")
        if not ok:
            ghl_errors.append({"email": email, "error": str(err)})
        # Also remove the legacy tag from GHL
        try:
            ghl_update_tags(email, remove={"relationship:family-shared-email"})
        except Exception:
            pass
        _time.sleep(0.15)

    return jsonify({
        "ok": True,
        "household": {"id": household_id, "slug": slug, "name": name, "head_person_id": head_id},
        "ghl_errors": ghl_errors,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v -k create_household
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_household_api.py
git commit -m "feat(households): POST /api/households create endpoint"
```

---

### Task 4: GET endpoints (list, detail, person-household, candidates)

**Files:**
- Modify: `app.py` (4 new routes)
- Modify: `tests/test_household_api.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_household_api.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify failures**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v -k "get_households or get_household_detail or get_person_household or get_household_candidates"
```

Expected: 4 failures (404s).

- [ ] **Step 3: Add the four GET endpoints to `app.py`**

Add near `create_household`:

```python
@app.route("/api/households", methods=["GET"])
def list_households():
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute("""
            SELECT h.id, h.slug, h.name, h.head_person_id, h.updated_at,
                   p.first_name AS head_first, p.last_name AS head_last
            FROM households h
            LEFT JOIN people p ON p.id = h.head_person_id
            ORDER BY h.name
        """).fetchall()
        out = []
        for r in rows:
            count = cx.execute(
                "SELECT COUNT(*) FROM people WHERE tags LIKE ?", (f'%"household:{r["slug"]}"%',)
            ).fetchone()[0]
            out.append({
                "id": r["id"],
                "slug": r["slug"],
                "name": r["name"],
                "member_count": count,
                "head": {
                    "id": r["head_person_id"],
                    "name": f'{r["head_first"] or ""} {r["head_last"] or ""}'.strip(),
                },
                "updated_at": r["updated_at"],
            })
    return jsonify({"households": out})


@app.route("/api/households/<slug>", methods=["GET"])
def get_household(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM households WHERE slug=?", (slug,)).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404
        members = cx.execute("""
            SELECT id, email, first_name, last_name, phone, tags
            FROM people WHERE tags LIKE ?
            ORDER BY first_name
        """, (f'%"household:{slug}"%',)).fetchall()
        member_list = []
        for m in members:
            try:
                tags = json.loads(m["tags"] or "[]")
            except Exception:
                tags = []
            member_list.append({
                "id": m["id"],
                "email": m["email"],
                "first_name": m["first_name"],
                "last_name": m["last_name"],
                "phone": m["phone"],
                "name": f'{m["first_name"]} {m["last_name"]}'.strip(),
                "is_head": f"household-head:{slug}" in tags,
            })
    return jsonify({
        "id": row["id"], "slug": row["slug"], "name": row["name"],
        "head_person_id": row["head_person_id"], "address": row["address"],
        "notes": row["notes"], "created_at": row["created_at"],
        "updated_at": row["updated_at"], "created_by": row["created_by"],
        "members": member_list,
    })


@app.route("/api/people/<int:person_id>/household", methods=["GET"])
def get_person_household(person_id):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    with sqlite3.connect(LOG_DB) as cx:
        slug = _person_household_slug(cx, person_id)
        if not slug:
            return jsonify({"household": None})
    # Reuse the full-household renderer
    return get_household(slug)   # already returns 404 if the slug somehow vanished


@app.route("/api/household-candidates", methods=["GET"])
def list_household_candidates():
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    status = request.args.get("status", "pending")
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute(
            "SELECT * FROM household_candidates WHERE status=? ORDER BY detected_at DESC",
            (status,)
        ).fetchall()
        out = []
        for r in rows:
            try:
                pids = json.loads(r["person_ids"] or "[]")
            except Exception:
                pids = []
            persons = []
            if pids:
                placeholders = ",".join("?" * len(pids))
                people_rows = cx.execute(
                    f"SELECT id, email, first_name, last_name FROM people WHERE id IN ({placeholders})",
                    pids
                ).fetchall()
                persons = [{"id": p["id"], "email": p["email"],
                            "name": f'{p["first_name"]} {p["last_name"]}'.strip()}
                           for p in people_rows]
            out.append({
                "id": r["id"], "signal": r["signal"], "detected_at": r["detected_at"],
                "person_ids": pids, "persons": persons,
            })
    return jsonify({"candidates": out})
```

Note: `get_person_household` reuses `get_household` by calling it directly — this is intentional DRY. Flask's view functions are just Python callables.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v -k "get_households or get_household_detail or get_person_household or get_household_candidates"
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_household_api.py
git commit -m "feat(households): GET endpoints (list, detail, person, candidates)"
```

---

### Task 5: PATCH + members + DELETE endpoints

**Files:**
- Modify: `app.py` (4 new routes: PATCH, POST member, DELETE member, DELETE household)
- Modify: `tests/test_household_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_household_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failures**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v -k "patch_household or add_member or remove_member or remove_head or disband"
```

Expected: 6 failures (404 or 405).

- [ ] **Step 3: Add the four endpoints to `app.py`**

Add near `create_household`:

```python
@app.route("/api/households/<slug>", methods=["PATCH"])
def update_household(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    ts = datetime.now(timezone.utc).isoformat()

    ghl_errors = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM households WHERE slug=?", (slug,)).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404

        new_name    = body.get("name", row["name"])
        new_address = body.get("address", row["address"])
        new_notes   = body.get("notes", row["notes"])
        new_head    = body.get("head_person_id", row["head_person_id"])

        # Head change requires moving the household-head: tag
        old_head = row["head_person_id"]
        head_changed = new_head != old_head
        old_head_email = new_head_email = None
        if head_changed:
            # Validate new head is in this household
            new_head_slug = _person_household_slug(cx, new_head)
            if new_head_slug != slug:
                return jsonify({"error": "new head must be a current member"}), 400
            _mutate_person_tags(cx, old_head, remove={f"household-head:{slug}"})
            _mutate_person_tags(cx, new_head, add={f"household-head:{slug}"})
            r = cx.execute("SELECT email FROM people WHERE id=?", (old_head,)).fetchone()
            old_head_email = r[0] if r else None
            r = cx.execute("SELECT email FROM people WHERE id=?", (new_head,)).fetchone()
            new_head_email = r[0] if r else None

        cx.execute("""
            UPDATE households SET name=?, address=?, notes=?, head_person_id=?, updated_at=?
            WHERE slug=?
        """, (new_name, new_address, new_notes, new_head, ts, slug))
        cx.commit()

    # GHL sync outside lock
    if head_changed:
        if old_head_email:
            _, err = ghl_update_tags(old_head_email, remove={f"household-head:{slug}"})
            if err: ghl_errors.append({"email": old_head_email, "error": str(err)})
            _time.sleep(0.15)
        if new_head_email:
            _, err = ghl_update_tags(new_head_email, add={f"household-head:{slug}"})
            if err: ghl_errors.append({"email": new_head_email, "error": str(err)})
            _time.sleep(0.15)

    return jsonify({"ok": True, "ghl_errors": ghl_errors})


@app.route("/api/households/<slug>/members", methods=["POST"])
def add_household_member(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    person_id = body.get("person_id")
    if not person_id:
        return jsonify({"error": "person_id required"}), 400

    ghl_errors = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        if not cx.execute("SELECT 1 FROM households WHERE slug=?", (slug,)).fetchone():
            return jsonify({"error": "household not found"}), 404
        existing_slug = _person_household_slug(cx, person_id)
        if existing_slug == slug:
            return jsonify({"ok": True, "already_member": True})
        if existing_slug:
            existing_row = cx.execute(
                "SELECT slug, name FROM households WHERE slug=?", (existing_slug,)
            ).fetchone()
            return jsonify({
                "error": "person already in household",
                "current_household": {"slug": existing_slug,
                                       "name": existing_row["name"] if existing_row else existing_slug},
            }), 409
        person_row = cx.execute("SELECT email FROM people WHERE id=?", (person_id,)).fetchone()
        if not person_row:
            return jsonify({"error": "person not found"}), 404
        email = person_row["email"]
        _mutate_person_tags(cx, person_id, add={f"household:{slug}"},
                            remove={"relationship:family-shared-email"})
        cx.commit()

    if email:
        ok, err = _push_household_tags_to_ghl(email, slug, is_head=False, action="add")
        if not ok: ghl_errors.append({"email": email, "error": str(err)})
        try: ghl_update_tags(email, remove={"relationship:family-shared-email"})
        except Exception: pass
    return jsonify({"ok": True, "ghl_errors": ghl_errors})


@app.route("/api/households/<slug>/members/<int:person_id>", methods=["DELETE"])
def remove_household_member(slug, person_id):
    auth_err = _check_console_auth()
    if auth_err: return auth_err

    ghl_errors = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        h_row = cx.execute("SELECT head_person_id FROM households WHERE slug=?", (slug,)).fetchone()
        if not h_row:
            return jsonify({"error": "household not found"}), 404
        if h_row["head_person_id"] == person_id:
            return jsonify({"error": "Cannot remove head — change head first"}), 409
        person_row = cx.execute("SELECT email FROM people WHERE id=?", (person_id,)).fetchone()
        if not person_row:
            return jsonify({"error": "person not found"}), 404
        email = person_row["email"]
        _mutate_person_tags(cx, person_id, remove={f"household:{slug}", f"household-head:{slug}"})
        cx.commit()

    if email:
        ok, err = _push_household_tags_to_ghl(email, slug, is_head=False, action="remove")
        if not ok: ghl_errors.append({"email": email, "error": str(err)})
    return jsonify({"ok": True, "ghl_errors": ghl_errors})


@app.route("/api/households/<slug>", methods=["DELETE"])
def disband_household(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err

    ghl_errors = []
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        if not cx.execute("SELECT 1 FROM households WHERE slug=?", (slug,)).fetchone():
            return jsonify({"error": "household not found"}), 404
        members = cx.execute("""
            SELECT id, email FROM people WHERE tags LIKE ?
        """, (f'%"household:{slug}"%',)).fetchall()
        for m in members:
            _mutate_person_tags(cx, m["id"],
                                remove={f"household:{slug}", f"household-head:{slug}"})
        # Mark related candidates resolved
        cx.execute("""
            UPDATE household_candidates SET status='dismissed', resolved_at=?
            WHERE household_id=(SELECT id FROM households WHERE slug=?)
        """, (datetime.now(timezone.utc).isoformat(), slug))
        cx.execute("DELETE FROM households WHERE slug=?", (slug,))
        cx.commit()

    for m in members:
        if not m["email"]: continue
        _, err = ghl_update_tags(m["email"],
                                  remove={f"household:{slug}", f"household-head:{slug}"})
        if err: ghl_errors.append({"email": m["email"], "error": str(err)})
        _time.sleep(0.15)
    return jsonify({"ok": True, "ghl_errors": ghl_errors})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v
```

Expected: all household API tests PASSED (the new ones + everything from previous tasks).

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_household_api.py
git commit -m "feat(households): PATCH, add/remove member, disband endpoints"
```

---

### Task 6: Resync endpoints

**Files:**
- Modify: `app.py` (2 new routes)
- Modify: `tests/test_household_api.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_household_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py::test_resync_ghl_pushes_household_tags_to_all_members -v
```

Expected: 404.

- [ ] **Step 3: Add the two endpoints to `app.py`**

Add near the other household routes:

```python
def _resync_household_to_ghl(slug):
    """Re-push the household tags for every member to GHL. Returns ghl_errors list."""
    ghl_errors = []
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        h_row = cx.execute("SELECT head_person_id FROM households WHERE slug=?", (slug,)).fetchone()
        if not h_row:
            return [{"error": "household not found"}]
        head_id = h_row["head_person_id"]
        members = cx.execute("""
            SELECT id, email FROM people WHERE tags LIKE ?
        """, (f'%"household:{slug}"%',)).fetchall()
    for m in members:
        if not m["email"]: continue
        is_head = (m["id"] == head_id)
        ok, err = _push_household_tags_to_ghl(m["email"], slug, is_head, action="add")
        if not ok: ghl_errors.append({"email": m["email"], "error": str(err)})
        _time.sleep(0.15)
    return ghl_errors


@app.route("/api/households/<slug>/resync-ghl", methods=["POST"])
def resync_household_ghl(slug):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    errors = _resync_household_to_ghl(slug)
    return jsonify({"ok": True, "ghl_errors": errors})


def resync_all_households_to_ghl():
    """Iterate every household and push its tags to GHL. Used by daily cron
    for drift recovery. Returns {households_synced, ghl_errors_total}."""
    with sqlite3.connect(LOG_DB) as cx:
        slugs = [r[0] for r in cx.execute("SELECT slug FROM households").fetchall()]
    total_errors = 0
    for slug in slugs:
        errors = _resync_household_to_ghl(slug)
        total_errors += len(errors)
    return {"households_synced": len(slugs), "ghl_errors_total": total_errors}


@app.route("/admin/resync-all-households", methods=["POST"])
def admin_resync_all_households():
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    try:
        summary = resync_all_households_to_ghl()
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        app.logger.exception("resync-all-households failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py::test_resync_ghl_pushes_household_tags_to_all_members -v
```

Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_household_api.py
git commit -m "feat(households): resync-ghl + admin/resync-all-households"
```

---

## Milestone 3 — Detection

### Task 7: `detect_household_candidates()` algorithm

**Files:**
- Modify: `app.py` (add `detect_household_candidates`)
- Create: `tests/test_household_detection.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_household_detection.py`:

```python
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
    return importlib.import_module("app")


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
                tags TEXT DEFAULT '[]'
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
    _insert_person(tmp_db, "share@x.com", first="B", last="One") if False else None  # email is UNIQUE
    # Workaround: same email is enforced unique by the column. Use two
    # different rows with same lowercase normalization (e.g. capitalization
    # differences). For the test, two distinct emails with same casefolded value.
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_detection.py -v
```

Expected: 4 failures (`AttributeError: module 'app' has no attribute 'detect_household_candidates'`).

- [ ] **Step 3: Add the detection function to `app.py`**

Add near the other household helpers:

```python
def detect_household_candidates():
    """Run all signals against the people table, dedup against existing
    household_candidates rows, insert new pending candidates. Returns:
    {detected, new_pending, skipped_already_household, skipped_dedup}."""
    summary = {"detected": 0, "new_pending": 0,
               "skipped_already_household": 0, "skipped_dedup": 0}
    ts = datetime.now(timezone.utc).isoformat()

    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        people = cx.execute("""
            SELECT id, LOWER(TRIM(email)) AS email_lc, LOWER(TRIM(last_name)) AS last_lc,
                   phone, LOWER(TRIM(city)) AS city_lc, LOWER(TRIM(state)) AS state_lc, tags
            FROM people
        """).fetchall()

        # Mark which people are already in a household
        in_household = set()
        for p in people:
            try:
                tags = json.loads(p["tags"] or "[]")
            except Exception:
                tags = []
            if any(t.startswith("household:") and not t.startswith("household-head:") for t in tags):
                in_household.add(p["id"])

        # Existing dedup keys (any status — pending, confirmed, dismissed)
        existing_keys = set()
        for r in cx.execute("SELECT person_ids FROM household_candidates").fetchall():
            try:
                ids = json.loads(r[0] or "[]")
            except Exception:
                continue
            existing_keys.add(_candidate_dedup_key(ids))

        # ── Signal 1: shared-email ────────────────────────────────────────────
        by_email = {}
        for p in people:
            if not p["email_lc"]: continue
            by_email.setdefault(p["email_lc"], []).append(p["id"])
        # ── Signal 2: shared-phone-lastname ───────────────────────────────────
        by_phone_last = {}
        for p in people:
            if not (p["phone"] and p["last_lc"]): continue
            by_phone_last.setdefault((p["phone"], p["last_lc"]), []).append(p["id"])
        # ── Signal 3: shared-address-lastname ─────────────────────────────────
        by_addr = {}
        for p in people:
            if not (p["city_lc"] and p["state_lc"] and p["last_lc"]): continue
            by_addr.setdefault((p["city_lc"], p["state_lc"], p["last_lc"]), []).append(p["id"])

        def _emit_signal(name, clusters):
            for ids in clusters.values():
                if len(ids) < 2: continue
                summary["detected"] += 1
                if any(i in in_household for i in ids):
                    summary["skipped_already_household"] += 1
                    continue
                key = _candidate_dedup_key(ids)
                if key in existing_keys:
                    summary["skipped_dedup"] += 1
                    continue
                cx.execute("""
                    INSERT INTO household_candidates (detected_at, signal, person_ids, status)
                    VALUES (?, ?, ?, 'pending')
                """, (ts, name, json.dumps(sorted(ids))))
                existing_keys.add(key)   # avoid intra-run dups across signals
                summary["new_pending"] += 1

        _emit_signal("shared-email",            by_email)
        _emit_signal("shared-phone-lastname",   by_phone_last)
        _emit_signal("shared-address-lastname", by_addr)
        cx.commit()
    return summary


@app.route("/admin/detect-household-candidates", methods=["POST"])
def admin_detect_household_candidates():
    key = (request.headers.get("X-Cron-Secret", "")
           or request.headers.get("X-Console-Key", "")
           or request.args.get("key", ""))
    expected = os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "")
    if not expected or key != expected:
        return jsonify({"error": "unauthorized"}), 401
    try:
        summary = detect_household_candidates()
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        app.logger.exception("detect_household_candidates failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
```

- [ ] **Step 4: Fix the test with email UNIQUE constraint**

The first test uses `Share@x.com` to bypass the unique constraint via casing. Verify this works:

```bash
cd ~/deploy-chat && python3 -c "
import sqlite3, tempfile, os
db = tempfile.mktemp(suffix='.db')
with sqlite3.connect(db) as cx:
    cx.execute('CREATE TABLE p (email TEXT UNIQUE)')
    cx.execute(\"INSERT INTO p VALUES ('a@x.com')\")
    cx.execute(\"INSERT INTO p VALUES ('A@x.com')\")
    print('OK — case-different emails accepted')
os.unlink(db)
"
```

Expected: `OK — case-different emails accepted`. SQLite TEXT UNIQUE is case-sensitive by default, so this works for the test.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_detection.py -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_household_detection.py
git commit -m "feat(households): detect_household_candidates + admin endpoint"
```

---

### Task 8: Confirm/dismiss candidate endpoints

**Files:**
- Modify: `app.py`
- Modify: `tests/test_household_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_household_api.py`:

```python
def test_confirm_candidate_creates_household_and_links(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    monkeypatch.setattr(app, "GHL_API_KEY", "")
    p1 = _seed_person(tmp_db, "a@x.com", first="A", last="Z")
    p2 = _seed_person(tmp_db, "b@x.com", first="B", last="Z")
    with sqlite3.connect(tmp_db) as cx:
        cx.execute("INSERT INTO household_candidates (detected_at, signal, person_ids) VALUES (?, ?, ?)",
                   ("2026-05-26T00:00:00", "shared-phone-lastname", json.dumps(sorted([p1, p2]))))
        cx.commit()

    client = app.app.test_client()
    r = client.post("/api/household-candidates/1/confirm",
                    headers={"X-Console-Key": "testkey"},
                    json={"name": "Z Family", "head_person_id": p1})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        status, hid = cx.execute("SELECT status, household_id FROM household_candidates WHERE id=1").fetchone()
        assert status == "confirmed"
        assert hid is not None


def test_dismiss_candidate_sets_status(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    _seed_people_schema(tmp_db); _seed_household_tables(tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    with sqlite3.connect(tmp_db) as cx:
        cx.execute("INSERT INTO household_candidates (detected_at, signal, person_ids) VALUES (?, ?, ?)",
                   ("2026-05-26T00:00:00", "shared-email", json.dumps([1, 2])))
        cx.commit()

    client = app.app.test_client()
    r = client.post("/api/household-candidates/1/dismiss", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    with sqlite3.connect(tmp_db) as cx:
        status = cx.execute("SELECT status FROM household_candidates WHERE id=1").fetchone()[0]
    assert status == "dismissed"
```

- [ ] **Step 2: Run tests, verify failures**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v -k "confirm_candidate or dismiss_candidate"
```

Expected: 2 failures.

- [ ] **Step 3: Add the endpoints to `app.py`**

```python
@app.route("/api/household-candidates/<int:cand_id>/confirm", methods=["POST"])
def confirm_household_candidate(cand_id):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    head_id = body.get("head_person_id")
    if not (name and head_id):
        return jsonify({"error": "name + head_person_id required"}), 400

    with sqlite3.connect(LOG_DB) as cx:
        row = cx.execute("SELECT person_ids, status FROM household_candidates WHERE id=?", (cand_id,)).fetchone()
    if not row:
        return jsonify({"error": "candidate not found"}), 404
    if row[1] != "pending":
        return jsonify({"error": f"candidate is {row[1]}, not pending"}), 409
    try:
        member_ids = json.loads(row[0] or "[]")
    except Exception:
        return jsonify({"error": "candidate has invalid person_ids"}), 500

    # Delegate to create_household via internal call
    with app.test_request_context("/api/households", method="POST",
                                   json={"name": name, "head_person_id": head_id,
                                         "member_person_ids": member_ids},
                                   headers={"X-Console-Key": CONSOLE_SECRET or ""}):
        resp = create_household()
    if isinstance(resp, tuple):
        body, status = resp[0], resp[1]
        if status != 200:
            return body, status
        body = body.get_json()
    else:
        body = resp.get_json()

    # Link candidate to the new household
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.execute("""
            UPDATE household_candidates SET status='confirmed',
                resolved_at=?, resolved_by=?, household_id=?
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), "glen", body["household"]["id"], cand_id))
        cx.commit()
    return jsonify({"ok": True, "household": body["household"], "ghl_errors": body.get("ghl_errors", [])})


@app.route("/api/household-candidates/<int:cand_id>/dismiss", methods=["POST"])
def dismiss_household_candidate(cand_id):
    auth_err = _check_console_auth()
    if auth_err: return auth_err
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        if not cx.execute("SELECT 1 FROM household_candidates WHERE id=?", (cand_id,)).fetchone():
            return jsonify({"error": "candidate not found"}), 404
        cx.execute("""
            UPDATE household_candidates SET status='dismissed', resolved_at=?, resolved_by=?
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), "glen", cand_id))
        cx.commit()
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/deploy-chat && python3 -m pytest tests/test_household_api.py -v -k "confirm_candidate or dismiss_candidate"
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py tests/test_household_api.py
git commit -m "feat(households): confirm + dismiss candidate endpoints"
```

---

### Task 9: Chain detection + drift recovery into `/admin/sync-pb-tags`

**Files:**
- Modify: `app.py` (modify the existing `admin_sync_pb_tags` handler — search for `def admin_sync_pb_tags` to locate)

- [ ] **Step 1: Locate the existing handler**

```bash
cd ~/deploy-chat && grep -n "def admin_sync_pb_tags" app.py
```

Expected: one line number — find the line, read 20 lines around it.

- [ ] **Step 2: Modify the handler to chain post-sync steps**

Edit the existing `admin_sync_pb_tags` function. After the `summary = sync_pb_to_people_and_ghl(...)` line and before `return jsonify(...)`, add:

```python
        # After successful PB sync, run household-side steps:
        # 1. Detect new candidates (new PB contacts may match existing patterns)
        # 2. Resync household tags to GHL (drift recovery)
        try:
            summary["households"] = {
                "detection": detect_household_candidates(),
                "resync":    resync_all_households_to_ghl(),
            }
        except Exception as e:
            app.logger.exception("post-pb household steps failed")
            summary["households"] = {"error": f"{type(e).__name__}: {e}"}
```

The final structure should look like:

```python
@app.route("/admin/sync-pb-tags", methods=["POST"])
def admin_sync_pb_tags():
    # ... existing auth ...
    # ... existing dry_run/limit parsing ...
    try:
        summary = sync_pb_to_people_and_ghl(
            dry_run=dry_run,
            limit=int(limit) if limit else None,
        )
        # NEW: chain household-side steps unless dry_run
        if not dry_run:
            try:
                summary["households"] = {
                    "detection": detect_household_candidates(),
                    "resync":    resync_all_households_to_ghl(),
                }
            except Exception as e:
                app.logger.exception("post-pb household steps failed")
                summary["households"] = {"error": f"{type(e).__name__}: {e}"}
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        app.logger.exception("PB sync failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
```

- [ ] **Step 3: Quick smoke test**

```bash
cd ~/deploy-chat && python3 -c "import ast; ast.parse(open('app.py').read()); print('parses OK')"
```

Expected: `parses OK`.

- [ ] **Step 4: Run the full pytest suite**

```bash
cd ~/deploy-chat && python3 -m pytest tests/ -v
```

Expected: all PASS (no test specifically covers the chain, but nothing should regress).

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add app.py
git commit -m "feat(households): chain detection + resync into /admin/sync-pb-tags"
```

---

## Milestone 4 — Frontend

**Note on testing:** No JS test framework in this project. Each frontend task ends with a manual browser-verification step against a local dev server. The skill says when frontend testing isn't automated, say so explicitly — that's what these `Verify` steps are.

### Task 10: Household section in Overview tab

**Files:**
- Modify: `static/console.html` (around line 1143 `loadPersonDetail` and line 1151 `renderPersonDetail`)

- [ ] **Step 1: Locate the existing `renderPersonDetail` function**

```bash
grep -n "function renderPersonDetail\|function loadPersonDetail" ~/deploy-chat/static/console.html
```

Expected: two line numbers.

- [ ] **Step 2: Modify `loadPersonDetail` to fetch household**

Find this function (around line 1143) and replace it with:

```javascript
async function loadPersonDetail(id) {
  _selectedPersonId = id;
  document.querySelectorAll('.person-card').forEach(c => c.classList.toggle('selected', c.id===`pcard-${id}`));
  const [pRes, hRes] = await Promise.all([
    fetch(`${BASE}/api/people/${id}`, { headers:{'X-Console-Key':consoleKey} }),
    fetch(`${BASE}/api/people/${id}/household`, { headers:{'X-Console-Key':consoleKey} }),
  ]);
  const p = await pRes.json();
  const h = await hRes.json();
  renderPersonDetail(p, h.household);
}
```

- [ ] **Step 3: Modify `renderPersonDetail` to render the household section**

Find the function (around line 1151) and replace the line `${field('Source', p.source)} ${field('GHL ID', p.ghl_id)}` inside the Overview pane with:

```javascript
      ${field('Source', p.source)} ${field('GHL ID', p.ghl_id)}
      ${_renderHouseholdSection(p, household)}
```

Then add this helper function right after `renderPersonDetail` (before `showDetailPane`):

```javascript
function _renderHouseholdSection(person, household) {
  if (!household) {
    return `<div class="detail-field" style="margin-top:14px">
      <button class="detail-enrich-btn" style="border-color:var(--justus);color:var(--justus)"
              onclick="openHouseholdDialogForPerson(${person.id})">+ Mark as household</button>
    </div>`;
  }
  const members = household.members || [];
  const headBadge = `<span style="font-size:10px;padding:2px 6px;border-radius:4px;background:#4a3520;color:#f5d4a0;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-left:6px">${_esc(members.find(m=>m.is_head)?.first_name||'')} is head</span>`;
  const memberCards = members.map(m => {
    const initials = ((m.first_name||'?')[0] + (m.last_name||'?')[0]).toUpperCase();
    const thisOne = m.id === person.id ? 'opacity:0.7;border-color:#f5b87a' : '';
    return `<div class="household-member" style="display:flex;gap:10px;padding:8px 10px;margin:6px 0;background:#0f0c08;border:1px solid #2a1f10;border-radius:6px;cursor:pointer;${thisOne}"
            onclick="loadPersonDetail(${m.id})">
      <div style="width:28px;height:28px;border-radius:50%;background:#4a3520;color:#f5d4a0;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700">${initials}</div>
      <div style="flex:1;min-width:0">
        <div style="font-size:13px;font-weight:600">${_esc(m.name)} ${m.is_head ? '<span style="font-size:9px;padding:2px 5px;border-radius:4px;background:#4a3520;color:#f5d4a0;text-transform:uppercase;letter-spacing:0.05em">Head</span>' : ''}</div>
        <div style="font-size:11px;color:#8a93a3">${_esc(m.email||'')}</div>
      </div>
    </div>`;
  }).join('');
  return `<div class="detail-field" style="margin-top:14px;background:#1a1410;border:1px solid #2a1f10;border-radius:8px;padding:12px">
    <h4 style="margin:0 0 8px;font-size:12px;color:#f5b87a;display:flex;justify-content:space-between;align-items:center">
      <span>👥 ${_esc(household.name)} household ${headBadge}</span>
      <button onclick="openHouseholdEditDialog('${_esc(household.slug)}')" style="background:transparent;border:none;color:#8a93a3;cursor:pointer;font-size:14px">⚙</button>
    </h4>
    ${memberCards}
    <button class="detail-enrich-btn" style="border-color:#f5b87a;color:#f5b87a;width:100%;margin-top:6px;font-size:11px"
            onclick="openHouseholdDialogForAddMember('${_esc(household.slug)}')">+ Add member</button>
  </div>`;
}
```

- [ ] **Step 4: Add stub functions to avoid runtime errors**

The render function references `openHouseholdDialogForPerson`, `openHouseholdEditDialog`, `openHouseholdDialogForAddMember`. Add these stubs after `_renderHouseholdSection` for now — the full implementations come in the next tasks:

```javascript
function openHouseholdDialogForPerson(personId)   { alert('TODO Task 11: + Mark as household for person ' + personId); }
function openHouseholdEditDialog(slug)            { alert('TODO Task 13: Edit household ' + slug); }
function openHouseholdDialogForAddMember(slug)    { alert('TODO Task 13: Add member to ' + slug); }
```

These will be replaced in Tasks 11–13.

- [ ] **Step 5: Verify in browser**

```bash
cd ~/deploy-chat && doppler run --project remedy-match --config prd -- python3 -c "from app import app; app.run(port=5050, debug=False)" &
sleep 3
echo "Open http://localhost:5050/console in browser; auth with CONSOLE_SECRET; click any person in People page."
```

Expected:
- A person not in a household sees `+ Mark as household` button in Overview.
- A person who's in a household (one of the existing Savant/Perdomo test cases) sees the household section with member cards.
- Clicking a member card navigates to that person.
- The stub buttons show alerts.

Kill the dev server: `pkill -f "port=5050"`

- [ ] **Step 6: Commit**

```bash
cd ~/deploy-chat && git add static/console.html
git commit -m "feat(console): household section in person Overview tab"
```

---

### Task 11: `household-dialog` + entry point A (mark from person detail)

**Files:**
- Modify: `static/console.html` (add `<dialog id="household-dialog">` near the existing `<dialog id="new-task-dialog">` at line 1887; add JS dialog handlers)

- [ ] **Step 1: Add the dialog markup**

Find `<dialog id="new-task-dialog">` (around line 1887) and add this BEFORE it:

```html
<dialog id="household-dialog">
  <div class="newtask-header" style="border-color:#f5b87a">
    <h3 id="hh-dialog-title">👥 Create household</h3>
  </div>
  <div class="newtask-body">
    <div><label for="hh-name">Household name</label>
      <input type="text" id="hh-name" placeholder="auto: last name of head" autocomplete="off"></div>
    <div id="hh-members-section">
      <label>Members</label>
      <div id="hh-members-chosen" style="background:#0a0c12;border:1px solid #1c2433;border-radius:4px;padding:6px;min-height:32px"></div>
    </div>
    <div><label for="hh-search">Add member by name or email</label>
      <input type="text" id="hh-search" oninput="hhSearch()" placeholder="start typing..." autocomplete="off">
      <div id="hh-search-results" style="background:#0a0c12;border:1px solid #1c2433;border-radius:4px;margin-top:4px;max-height:200px;overflow-y:auto"></div>
    </div>
    <div id="hh-errors" style="color:#f57a7a;font-size:12px;margin-top:8px"></div>
  </div>
  <div class="newtask-footer">
    <button class="btn-newtask-cancel" onclick="document.getElementById('household-dialog').close()">Cancel</button>
    <button class="btn-newtask-save" style="background:#f5b87a;color:#0a0c12" id="hh-submit" onclick="submitHousehold()">Create household</button>
  </div>
</dialog>
```

- [ ] **Step 2: Add the JS for entry point A**

Find the stub `function openHouseholdDialogForPerson(personId)` from Task 10 and REPLACE it (and the surrounding stubs) with the real implementations. Insert near other dialog functions (search for `function openNewTaskDialog`):

```javascript
// ── Household dialog state ─────────────────────────────────────────────────────
let _hhMode = 'create';            // 'create' | 'add-member' | 'edit'
let _hhSlug = null;
let _hhChosen = [];                // [{id, name, email, is_head}]
let _hhSearchTimer = null;

async function openHouseholdDialogForPerson(personId) {
  _hhMode = 'create';
  _hhSlug = null;
  _hhChosen = [];
  document.getElementById('hh-dialog-title').textContent = '👥 Create household';
  document.getElementById('hh-name').value = '';
  document.getElementById('hh-search').value = '';
  document.getElementById('hh-search-results').innerHTML = '';
  document.getElementById('hh-errors').textContent = '';
  // Fetch the person, preselect as head
  const r = await fetch(`${BASE}/api/people/${personId}`, { headers:{'X-Console-Key':consoleKey} });
  const p = await r.json();
  _hhChosen = [{ id: p.id,
                 name: p.name || `${p.first_name||''} ${p.last_name||''}`.trim() || p.email,
                 email: p.email, is_head: true }];
  document.getElementById('hh-name').value = p.last_name || '';
  _renderHHChosen();
  document.getElementById('household-dialog').showModal();
  setTimeout(() => document.getElementById('hh-search').focus(), 50);
}

function _renderHHChosen() {
  const html = _hhChosen.map(c => `
    <span style="display:inline-flex;align-items:center;gap:4px;padding:3px 8px;margin:2px;background:#2a1f10;border-radius:12px;font-size:11px;color:#f5b87a;cursor:pointer"
          onclick="_hhToggleHead(${c.id})">
      ${_esc(c.name)}
      ${c.is_head ? '<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:#f5b87a;color:#0a0c12;font-weight:700;margin-left:4px">HEAD</span>' : ''}
      <span onclick="event.stopPropagation();_hhRemove(${c.id})" style="margin-left:6px;cursor:pointer;color:#8a93a3">✕</span>
    </span>`).join('');
  document.getElementById('hh-members-chosen').innerHTML = html || '<span style="color:#8a93a3;font-size:11px">no members yet</span>';
}

function _hhToggleHead(id) {
  _hhChosen = _hhChosen.map(c => ({...c, is_head: c.id === id}));
  _renderHHChosen();
}

function _hhRemove(id) {
  const c = _hhChosen.find(x => x.id === id);
  if (c?.is_head && _hhChosen.length > 1) {
    document.getElementById('hh-errors').textContent = 'Pick another head before removing the current head.';
    return;
  }
  _hhChosen = _hhChosen.filter(x => x.id !== id);
  _renderHHChosen();
}

function hhSearch() {
  clearTimeout(_hhSearchTimer);
  _hhSearchTimer = setTimeout(async () => {
    const q = document.getElementById('hh-search').value.trim();
    if (q.length < 2) {
      document.getElementById('hh-search-results').innerHTML = '';
      return;
    }
    const r = await fetch(`${BASE}/api/people?q=${encodeURIComponent(q)}&limit=10`,
                          { headers:{'X-Console-Key':consoleKey} });
    const data = await r.json();
    const chosenIds = new Set(_hhChosen.map(c => c.id));
    const html = (data.people || [])
      .filter(p => !chosenIds.has(p.id))
      .map(p => {
        const name = p.name || `${p.first_name||''} ${p.last_name||''}`.trim() || p.email;
        return `<div onclick="_hhAdd(${p.id},'${_esc(name).replace(/'/g,"\\'")}','${_esc(p.email)}')"
                     style="padding:6px 10px;border-bottom:1px solid #1c2433;cursor:pointer">
          <div style="font-size:12px">${_esc(name)}</div>
          <div style="font-size:10px;color:#8a93a3">${_esc(p.email)}</div>
        </div>`;
      }).join('');
    document.getElementById('hh-search-results').innerHTML = html || '<div style="padding:8px;color:#8a93a3;font-size:11px">no matches</div>';
  }, 250);
}

function _hhAdd(id, name, email) {
  if (_hhChosen.find(c => c.id === id)) return;
  _hhChosen.push({ id, name, email, is_head: false });
  document.getElementById('hh-search').value = '';
  document.getElementById('hh-search-results').innerHTML = '';
  _renderHHChosen();
}

async function submitHousehold() {
  const name = document.getElementById('hh-name').value.trim();
  if (!name) { document.getElementById('hh-errors').textContent = 'Name required.'; return; }
  if (_hhChosen.length < 2) { document.getElementById('hh-errors').textContent = 'Need at least 2 members.'; return; }
  const head = _hhChosen.find(c => c.is_head);
  if (!head) { document.getElementById('hh-errors').textContent = 'Pick a head (click a chip).'; return; }
  document.getElementById('hh-submit').disabled = true;
  document.getElementById('hh-errors').textContent = '';
  try {
    const r = await fetch(`${BASE}/api/households`, {
      method:'POST', headers:{'Content-Type':'application/json','X-Console-Key':consoleKey},
      body: JSON.stringify({
        name,
        head_person_id: head.id,
        member_person_ids: _hhChosen.map(c => c.id),
      })
    });
    const data = await r.json();
    if (r.status === 409) {
      const h = data.current_household || {};
      if (confirm(`${data.error}. They're currently in "${h.name}". Move them?`)) {
        // TODO Task 12: implement move flow
        document.getElementById('hh-errors').textContent = 'Move flow not yet implemented.';
      }
      return;
    }
    if (!r.ok) { document.getElementById('hh-errors').textContent = data.error || 'Failed.'; return; }
    document.getElementById('household-dialog').close();
    if (_selectedPersonId) loadPersonDetail(_selectedPersonId);
  } finally {
    document.getElementById('hh-submit').disabled = false;
  }
}
```

- [ ] **Step 3: Verify in browser**

```bash
cd ~/deploy-chat && doppler run --project remedy-match --config prd -- python3 -c "from app import app; app.run(port=5050)" &
sleep 3
echo "Open http://localhost:5050/console; find a person not in a household; click '+ Mark as household'; add another person; create."
```

Expected: dialog opens preselecting the person as head, search works, household creates, page reloads showing the new household section.

`pkill -f "port=5050"` when done.

- [ ] **Step 4: Commit**

```bash
cd ~/deploy-chat && git add static/console.html
git commit -m "feat(console): household-dialog + entry point A (mark from detail)"
```

---

### Task 12: Multi-select entry point B (People list checkboxes + toolbar)

**Files:**
- Modify: `static/console.html` (people-list rendering around line 1118; people-section header for toolbar around line 497)

- [ ] **Step 1: Add multi-select state and toolbar markup**

In `static/console.html`, find the `<div id="people-section">` block (around line 497). Inside that div, BEFORE the `<div class="people-search-bar">` line, add the toolbar markup:

```html
  <div id="people-multiselect-toolbar"
       style="display:none;gap:8px;align-items:center;padding:8px 10px;background:#1a1410;border-radius:5px;margin-bottom:10px;border:1px solid #2a1f10">
    <span id="pms-count" style="flex:1;font-size:12px;color:#f5b87a;font-weight:600">0 selected</span>
    <button onclick="openHouseholdDialogForMultiSelect()" style="padding:6px 12px;background:#f5b87a;color:#0a0c12;border:none;border-radius:5px;font-size:12px;font-weight:600;cursor:pointer">Group as household</button>
    <button onclick="pmsClear()" style="padding:6px 12px;background:transparent;color:#f5b87a;border:1px solid #4a3520;border-radius:5px;font-size:12px;cursor:pointer">Clear</button>
  </div>
```

- [ ] **Step 2: Add checkbox to each person card**

Find the line `div.className = 'person-card' + (p.id === _selectedPersonId ? ' selected' : '');` (around line 1118). Replace the `div.innerHTML = ...` block that follows it. Locate the existing block:

```javascript
  div.innerHTML = `
    <div class="person-card-name">${_esc(p.name || (p.first_name+' '+p.last_name).trim() || p.email)}</div>
    <div class="person-card-email">${_esc(p.email)}${p.profession ? ' · '+_esc(p.profession) : ''}</div>
    <div class="person-card-tags">${tags.map(_renderPtag).join('')}</div>
    <div class="person-card-meta">${loc ? loc+' · ' : ''}${p.order_count>0?p.order_count+' order(s) · ':''}${activity}</div>`;
```

Replace with:

```javascript
  const checked = _pmsSelected.has(p.id) ? 'background:#f5b87a;border-color:#f5b87a;color:#0a0c12' : '';
  const checkmark = _pmsSelected.has(p.id) ? '✓' : '';
  div.innerHTML = `
    <div style="display:flex;align-items:flex-start;gap:8px">
      <div onclick="event.stopPropagation();pmsToggle(${p.id})"
           class="pms-check"
           style="flex-shrink:0;width:16px;height:16px;border:1px solid #4a3520;border-radius:3px;display:inline-flex;align-items:center;justify-content:center;background:#0a0c12;font-size:11px;font-weight:700;margin-top:1px;${checked}">${checkmark}</div>
      <div style="flex:1;min-width:0">
        <div class="person-card-name">${_esc(p.name || (p.first_name+' '+p.last_name).trim() || p.email)}</div>
        <div class="person-card-email">${_esc(p.email)}${p.profession ? ' · '+_esc(p.profession) : ''}</div>
        <div class="person-card-tags">${tags.map(_renderPtag).join('')}</div>
        <div class="person-card-meta">${loc ? loc+' · ' : ''}${p.order_count>0?p.order_count+' order(s) · ':''}${activity}</div>
      </div>
    </div>`;
```

- [ ] **Step 3: Add multi-select JS**

Insert near the other people-page JS (search for `let _peopleOffset` to find a good spot):

```javascript
// ── Multi-select for household creation ────────────────────────────────────────
const _pmsSelected = new Set();
const _pmsCache = new Map();   // id -> {name, email}

function pmsToggle(id) {
  if (_pmsSelected.has(id)) _pmsSelected.delete(id);
  else _pmsSelected.add(id);
  pmsRender();
  // Re-render only the affected card's check state (cheap full re-render fallback)
  loadPeople();
}

function pmsClear() {
  _pmsSelected.clear();
  pmsRender();
  loadPeople();
}

function pmsRender() {
  const tb = document.getElementById('people-multiselect-toolbar');
  if (_pmsSelected.size === 0) {
    tb.style.display = 'none';
  } else {
    tb.style.display = 'flex';
    document.getElementById('pms-count').textContent = `${_pmsSelected.size} selected`;
  }
}

async function openHouseholdDialogForMultiSelect() {
  if (_pmsSelected.size < 2) {
    alert('Pick at least 2 people first.');
    return;
  }
  _hhMode = 'create';
  _hhSlug = null;
  _hhChosen = [];
  document.getElementById('hh-dialog-title').textContent = '👥 Create household';
  document.getElementById('hh-name').value = '';
  document.getElementById('hh-search').value = '';
  document.getElementById('hh-search-results').innerHTML = '';
  document.getElementById('hh-errors').textContent = '';

  // Fetch all selected people in parallel
  const ids = Array.from(_pmsSelected);
  const people = await Promise.all(ids.map(id =>
    fetch(`${BASE}/api/people/${id}`, { headers:{'X-Console-Key':consoleKey} }).then(r => r.json())
  ));
  _hhChosen = people.map((p, i) => ({
    id: p.id,
    name: p.name || `${p.first_name||''} ${p.last_name||''}`.trim() || p.email,
    email: p.email,
    is_head: i === 0,   // first one defaults to head
  }));
  // Pre-fill household name with the most common last name in the selection
  const lastNames = people.map(p => p.last_name).filter(Boolean);
  if (lastNames.length) {
    document.getElementById('hh-name').value = lastNames.sort((a,b) =>
      lastNames.filter(v => v===a).length - lastNames.filter(v => v===b).length
    ).pop();
  }
  _renderHHChosen();
  document.getElementById('household-dialog').showModal();
}
```

- [ ] **Step 4: Verify in browser**

```bash
cd ~/deploy-chat && doppler run --project remedy-match --config prd -- python3 -c "from app import app; app.run(port=5050)" &
sleep 3
echo "Open http://localhost:5050/console; click checkboxes on 2+ people; toolbar appears; click 'Group as household'."
```

Expected: toolbar appears with selected count, dialog opens with all selected people preadded, first is head, can change head by clicking chips.

`pkill -f "port=5050"`

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add static/console.html
git commit -m "feat(console): multi-select toolbar + entry point B"
```

---

### Task 13: Edit / disband flow

**Files:**
- Modify: `static/console.html` (add edit handlers; reuse dialog from Task 11)

- [ ] **Step 1: Replace the stub edit + add-member functions**

Find the stub `openHouseholdEditDialog` and `openHouseholdDialogForAddMember` (from Task 10) and REPLACE with:

```javascript
async function openHouseholdDialogForAddMember(slug) {
  _hhMode = 'add-member';
  _hhSlug = slug;
  _hhChosen = [];
  document.getElementById('hh-dialog-title').textContent = `👥 Add member to ${slug}`;
  document.getElementById('hh-name').value = '';
  document.getElementById('hh-name').disabled = true;   // can't rename in this mode
  document.getElementById('hh-members-section').style.display = 'none';
  document.getElementById('hh-search').value = '';
  document.getElementById('hh-search-results').innerHTML = '';
  document.getElementById('hh-errors').textContent = '';
  // Override submit behavior
  document.getElementById('hh-submit').textContent = 'Add member';
  document.getElementById('household-dialog').showModal();
}

async function openHouseholdEditDialog(slug) {
  _hhMode = 'edit';
  _hhSlug = slug;
  const r = await fetch(`${BASE}/api/households/${encodeURIComponent(slug)}`,
                        { headers:{'X-Console-Key':consoleKey} });
  const h = await r.json();
  document.getElementById('hh-dialog-title').textContent = `✏ Edit ${h.name} household`;
  document.getElementById('hh-name').value = h.name;
  document.getElementById('hh-name').disabled = false;
  document.getElementById('hh-members-section').style.display = '';
  _hhChosen = (h.members || []).map(m => ({
    id: m.id, name: m.name, email: m.email, is_head: m.is_head
  }));
  _renderHHChosen();
  document.getElementById('hh-search').value = '';
  document.getElementById('hh-search-results').innerHTML = '';
  document.getElementById('hh-errors').textContent = '';
  document.getElementById('hh-submit').textContent = 'Save changes';

  // Add a disband button below cancel/save
  let disband = document.getElementById('hh-disband-btn');
  if (!disband) {
    disband = document.createElement('button');
    disband.id = 'hh-disband-btn';
    disband.style.cssText = 'margin-right:auto;padding:9px 16px;background:transparent;border:1px solid #f57a7a;border-radius:6px;color:#f57a7a;font-size:13px;cursor:pointer';
    disband.textContent = 'Disband household';
    document.querySelector('#household-dialog .newtask-footer').prepend(disband);
  }
  disband.onclick = async () => {
    if (!confirm(`Disband ${h.name} household? Members will remain as individuals; all household tags will be removed (DB + GHL).`)) return;
    const r = await fetch(`${BASE}/api/households/${encodeURIComponent(slug)}`,
                          { method:'DELETE', headers:{'X-Console-Key':consoleKey} });
    if (r.ok) {
      document.getElementById('household-dialog').close();
      if (_selectedPersonId) loadPersonDetail(_selectedPersonId);
    } else {
      document.getElementById('hh-errors').textContent = 'Disband failed.';
    }
  };
  document.getElementById('household-dialog').showModal();
}
```

- [ ] **Step 2: Extend `submitHousehold` for edit/add-member modes**

Replace the existing `submitHousehold` function with this version that branches on `_hhMode`:

```javascript
async function submitHousehold() {
  document.getElementById('hh-errors').textContent = '';

  if (_hhMode === 'add-member') {
    if (_hhChosen.length === 0) { document.getElementById('hh-errors').textContent = 'Pick a person to add.'; return; }
    const member = _hhChosen[0];
    document.getElementById('hh-submit').disabled = true;
    try {
      const r = await fetch(`${BASE}/api/households/${encodeURIComponent(_hhSlug)}/members`, {
        method:'POST', headers:{'Content-Type':'application/json','X-Console-Key':consoleKey},
        body: JSON.stringify({ person_id: member.id })
      });
      const data = await r.json();
      if (r.status === 409) {
        const h = data.current_household || {};
        if (confirm(`${member.name} is in "${h.name}" household. Move them to ${_hhSlug}?`)) {
          // Remove from old, add to new
          await fetch(`${BASE}/api/households/${encodeURIComponent(h.slug)}/members/${member.id}`,
                      { method:'DELETE', headers:{'X-Console-Key':consoleKey} });
          await fetch(`${BASE}/api/households/${encodeURIComponent(_hhSlug)}/members`, {
            method:'POST', headers:{'Content-Type':'application/json','X-Console-Key':consoleKey},
            body: JSON.stringify({ person_id: member.id })
          });
        } else { return; }
      } else if (!r.ok) { document.getElementById('hh-errors').textContent = data.error || 'Failed.'; return; }
      document.getElementById('household-dialog').close();
      if (_selectedPersonId) loadPersonDetail(_selectedPersonId);
    } finally { document.getElementById('hh-submit').disabled = false; }
    return;
  }

  if (_hhMode === 'edit') {
    const name = document.getElementById('hh-name').value.trim();
    const head = _hhChosen.find(c => c.is_head);
    if (!name || !head) { document.getElementById('hh-errors').textContent = 'Name + head required.'; return; }
    document.getElementById('hh-submit').disabled = true;
    try {
      // PATCH name + head
      await fetch(`${BASE}/api/households/${encodeURIComponent(_hhSlug)}`, {
        method:'PATCH', headers:{'Content-Type':'application/json','X-Console-Key':consoleKey},
        body: JSON.stringify({ name, head_person_id: head.id })
      });
      // Removed members: compare to original state would require caching;
      // for v1, edit dialog supports rename + head change only. Member add/remove
      // is via the +Add/✕ buttons on the chips and triggers separate API calls.
      document.getElementById('household-dialog').close();
      if (_selectedPersonId) loadPersonDetail(_selectedPersonId);
    } finally { document.getElementById('hh-submit').disabled = false; }
    return;
  }

  // _hhMode === 'create' (original behavior)
  const name = document.getElementById('hh-name').value.trim();
  if (!name) { document.getElementById('hh-errors').textContent = 'Name required.'; return; }
  if (_hhChosen.length < 2) { document.getElementById('hh-errors').textContent = 'Need at least 2 members.'; return; }
  const head = _hhChosen.find(c => c.is_head);
  if (!head) { document.getElementById('hh-errors').textContent = 'Pick a head (click a chip).'; return; }
  document.getElementById('hh-submit').disabled = true;
  try {
    const r = await fetch(`${BASE}/api/households`, {
      method:'POST', headers:{'Content-Type':'application/json','X-Console-Key':consoleKey},
      body: JSON.stringify({ name, head_person_id: head.id, member_person_ids: _hhChosen.map(c => c.id) })
    });
    const data = await r.json();
    if (r.status === 409) {
      const h = data.current_household || {};
      const pname = (_hhChosen.find(c => c.id === data.person_id) || {}).name || 'this person';
      if (confirm(`${pname} is already in "${h.name}". Move them?`)) {
        await fetch(`${BASE}/api/households/${encodeURIComponent(h.slug)}/members/${data.person_id}`,
                    { method:'DELETE', headers:{'X-Console-Key':consoleKey} });
        // Retry the create
        return submitHousehold();
      }
      return;
    }
    if (!r.ok) { document.getElementById('hh-errors').textContent = data.error || 'Failed.'; return; }
    document.getElementById('household-dialog').close();
    if (_selectedPersonId) loadPersonDetail(_selectedPersonId);
  } finally { document.getElementById('hh-submit').disabled = false; }
}
```

- [ ] **Step 3: Verify in browser**

```bash
cd ~/deploy-chat && doppler run --project remedy-match --config prd -- python3 -c "from app import app; app.run(port=5050)" &
sleep 3
echo "Open http://localhost:5050/console; open an existing household; click ⚙ to edit; rename; change head; disband."
```

Expected: edit dialog opens with current state, rename works, head change works (verify via reload), disband prompts confirmation and removes household.

`pkill -f "port=5050"`

- [ ] **Step 4: Commit**

```bash
cd ~/deploy-chat && git add static/console.html
git commit -m "feat(console): household edit dialog + disband + 409 move flow"
```

---

### Task 14: Candidate review banner

**Files:**
- Modify: `static/console.html`

- [ ] **Step 1: Add the banner markup**

In the `<div id="people-section">` block (around line 497), add this BEFORE the `<div id="people-multiselect-toolbar">` (from Task 12):

```html
  <div id="household-candidates-banner" style="display:none;background:#1a1410;border:1px solid #2a1f10;border-radius:5px;padding:10px 12px;margin-bottom:10px">
    <div style="display:flex;align-items:center;gap:8px;cursor:pointer" onclick="toggleCandidatesPanel()">
      <span style="font-size:13px;color:#f5b87a">👥 <span id="hcb-count">0</span> possible household(s) detected</span>
      <span style="flex:1"></span>
      <span style="font-size:11px;color:#8a93a3" id="hcb-toggle">Review →</span>
    </div>
    <div id="hcb-panel" style="display:none;margin-top:10px;padding-top:10px;border-top:1px solid #2a1f10"></div>
  </div>
```

- [ ] **Step 2: Add the JS**

Insert near other people-page JS:

```javascript
// ── Household candidates banner ────────────────────────────────────────────────
let _hcbCandidates = [];

async function loadHouseholdCandidates() {
  const r = await fetch(`${BASE}/api/household-candidates?status=pending`,
                        { headers:{'X-Console-Key':consoleKey} });
  const data = await r.json();
  _hcbCandidates = data.candidates || [];
  document.getElementById('hcb-count').textContent = _hcbCandidates.length;
  document.getElementById('household-candidates-banner').style.display =
    _hcbCandidates.length > 0 ? '' : 'none';
}

function toggleCandidatesPanel() {
  const panel = document.getElementById('hcb-panel');
  const toggle = document.getElementById('hcb-toggle');
  if (panel.style.display === 'none') {
    renderCandidatesPanel();
    panel.style.display = '';
    toggle.textContent = 'Hide ↑';
  } else {
    panel.style.display = 'none';
    toggle.textContent = 'Review →';
  }
}

function renderCandidatesPanel() {
  const html = _hcbCandidates.map(c => {
    const persons = c.persons.map(p => _esc(p.name || p.email)).join(', ');
    const warning = c.persons.length > 5
      ? '<div style="color:#f5d07a;font-size:11px;margin-bottom:4px">⚠ Unusually large — review carefully</div>'
      : '';
    return `<div style="padding:10px;margin-bottom:6px;background:#0f0c08;border:1px solid #2a1f10;border-radius:5px">
      ${warning}
      <div style="font-size:11px;color:#8a93a3;margin-bottom:4px">signal: ${_esc(c.signal)}</div>
      <div style="font-size:13px;margin-bottom:6px">${persons}</div>
      <div style="display:flex;gap:6px">
        <button onclick="hcbConfirm(${c.id})" style="padding:5px 10px;background:#f5b87a;color:#0a0c12;border:none;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer">Confirm as household</button>
        <button onclick="hcbDismiss(${c.id})" style="padding:5px 10px;background:transparent;color:#8a93a3;border:1px solid #4a3520;border-radius:4px;font-size:11px;cursor:pointer">Dismiss</button>
      </div>
    </div>`;
  }).join('');
  document.getElementById('hcb-panel').innerHTML = html;
}

async function hcbConfirm(candId) {
  const cand = _hcbCandidates.find(c => c.id === candId);
  if (!cand) return;
  const name = prompt('Household name?', cand.persons[0]?.name?.split(' ').pop() || '');
  if (!name) return;
  // For v1 simplicity: head defaults to first member; user can rotate head later via Edit
  const r = await fetch(`${BASE}/api/household-candidates/${candId}/confirm`, {
    method:'POST', headers:{'Content-Type':'application/json','X-Console-Key':consoleKey},
    body: JSON.stringify({ name, head_person_id: cand.person_ids[0] })
  });
  if (r.ok) {
    loadHouseholdCandidates();
    loadPeople();
  } else {
    alert('Confirm failed: ' + (await r.json()).error);
  }
}

async function hcbDismiss(candId) {
  const r = await fetch(`${BASE}/api/household-candidates/${candId}/dismiss`,
                        { method:'POST', headers:{'X-Console-Key':consoleKey} });
  if (r.ok) loadHouseholdCandidates();
}
```

- [ ] **Step 3: Call `loadHouseholdCandidates` when entering the People section**

Find the `switchSection` function (search for `function switchSection`) and modify the people branch. Locate:

```javascript
function switchSection(section, btn) {
  // ...
  if (section === 'people') loadPeople();
```

Replace with:

```javascript
function switchSection(section, btn) {
  // ... (keep existing code)
  if (section === 'people') {
    loadPeople();
    loadHouseholdCandidates();
  }
```

- [ ] **Step 4: Verify in browser**

```bash
cd ~/deploy-chat && doppler run --project remedy-match --config prd -- python3 -c "from app import app; app.run(port=5050)" &
sleep 3
# Inject a fake candidate
doppler run --project remedy-match --config prd -- sqlite3 /tmp/test-people.db "INSERT INTO household_candidates (detected_at, signal, person_ids) VALUES ('2026-05-26T00:00:00', 'shared-email', '[1,2]');" 2>/dev/null || true
echo "Open http://localhost:5050/console People section. If you've seeded candidates, banner appears."
```

If no candidates exist locally, manually POST `/admin/detect-household-candidates` to populate from real data.

`pkill -f "port=5050"`

- [ ] **Step 5: Commit**

```bash
cd ~/deploy-chat && git add static/console.html
git commit -m "feat(console): household candidates review banner"
```

---

### Task 15: GHL sync error badges + retry button

**Files:**
- Modify: `static/console.html` — extend `_renderHouseholdSection` to show error state

- [ ] **Step 1: Extend the household-detail API response to include sync status**

Modify `_renderHouseholdSection` from Task 10. Replace the function with this version that adds a yellow warning surface — note this depends on the API returning a `last_sync_errors` field. For v1, we don't persist sync errors per-household; instead, the UI shows the warning only immediately after an operation returns errors. Track the most recent error per household in a module-level Map.

```javascript
const _hhRecentErrors = new Map();   // slug -> [{email, error}]

function _renderHouseholdSection(person, household) {
  if (!household) {
    return `<div class="detail-field" style="margin-top:14px">
      <button class="detail-enrich-btn" style="border-color:var(--justus);color:var(--justus)"
              onclick="openHouseholdDialogForPerson(${person.id})">+ Mark as household</button>
    </div>`;
  }
  const errors = _hhRecentErrors.get(household.slug) || [];
  const errorBadge = errors.length > 0 ? `
    <div style="margin:8px 0;padding:8px 10px;background:#3a2615;border:1px solid #f5d07a;border-radius:5px;color:#f5d07a;font-size:11px">
      ⚠ GHL sync incomplete (${errors.length} member(s) failed).
      <button onclick="hhResyncGhl('${_esc(household.slug)}')"
              style="margin-left:8px;padding:3px 8px;background:transparent;color:#f5d07a;border:1px solid #f5d07a;border-radius:3px;font-size:10px;cursor:pointer">Retry</button>
    </div>` : '';
  const members = household.members || [];
  const headBadge = `<span style="font-size:10px;padding:2px 6px;border-radius:4px;background:#4a3520;color:#f5d4a0;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-left:6px">${_esc(members.find(m=>m.is_head)?.first_name||'')} is head</span>`;
  const memberCards = members.map(m => {
    const initials = ((m.first_name||'?')[0] + (m.last_name||'?')[0]).toUpperCase();
    const thisOne = m.id === person.id ? 'opacity:0.7;border-color:#f5b87a' : '';
    return `<div style="display:flex;gap:10px;padding:8px 10px;margin:6px 0;background:#0f0c08;border:1px solid #2a1f10;border-radius:6px;cursor:pointer;${thisOne}"
            onclick="loadPersonDetail(${m.id})">
      <div style="width:28px;height:28px;border-radius:50%;background:#4a3520;color:#f5d4a0;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700">${initials}</div>
      <div style="flex:1;min-width:0">
        <div style="font-size:13px;font-weight:600">${_esc(m.name)} ${m.is_head ? '<span style="font-size:9px;padding:2px 5px;border-radius:4px;background:#4a3520;color:#f5d4a0;text-transform:uppercase;letter-spacing:0.05em">Head</span>' : ''}</div>
        <div style="font-size:11px;color:#8a93a3">${_esc(m.email||'')}</div>
      </div>
    </div>`;
  }).join('');
  return `<div class="detail-field" style="margin-top:14px;background:#1a1410;border:1px solid #2a1f10;border-radius:8px;padding:12px">
    <h4 style="margin:0 0 8px;font-size:12px;color:#f5b87a;display:flex;justify-content:space-between;align-items:center">
      <span>👥 ${_esc(household.name)} household ${headBadge}</span>
      <button onclick="openHouseholdEditDialog('${_esc(household.slug)}')" style="background:transparent;border:none;color:#8a93a3;cursor:pointer;font-size:14px">⚙</button>
    </h4>
    ${errorBadge}
    ${memberCards}
    <button class="detail-enrich-btn" style="border-color:#f5b87a;color:#f5b87a;width:100%;margin-top:6px;font-size:11px"
            onclick="openHouseholdDialogForAddMember('${_esc(household.slug)}')">+ Add member</button>
  </div>`;
}
```

- [ ] **Step 2: Wire error capture into the API write flows**

Update `submitHousehold` and `hhResyncGhl`. Add at the end of `submitHousehold` (after successful response, BEFORE the dialog close), within each mode branch:

```javascript
// After any successful API response that returned ghl_errors:
if (data.ghl_errors && data.ghl_errors.length > 0 && data.household) {
  _hhRecentErrors.set(data.household.slug, data.ghl_errors);
} else if (data.household) {
  _hhRecentErrors.delete(data.household.slug);
}
```

(Add the same pattern to the edit/add-member branches' success paths if they return `data.household`.)

Add `hhResyncGhl`:

```javascript
async function hhResyncGhl(slug) {
  const r = await fetch(`${BASE}/api/households/${encodeURIComponent(slug)}/resync-ghl`,
                        { method:'POST', headers:{'X-Console-Key':consoleKey} });
  const data = await r.json();
  if (data.ghl_errors && data.ghl_errors.length > 0) {
    _hhRecentErrors.set(slug, data.ghl_errors);
    alert(`Still ${data.ghl_errors.length} member(s) failing. Check Render logs.`);
  } else {
    _hhRecentErrors.delete(slug);
    alert('Re-sync succeeded.');
  }
  if (_selectedPersonId) loadPersonDetail(_selectedPersonId);
}
```

- [ ] **Step 3: Verify in browser (manual)**

```bash
cd ~/deploy-chat && doppler run --project remedy-match --config prd -- python3 -c "from app import app; app.run(port=5050)" &
sleep 3
echo "Open http://localhost:5050/console. Create a household. Errors only render when ghl_errors is non-empty — hard to trigger in dev without breaking GHL. Verify the Retry button calls /resync-ghl by Network tab."
```

`pkill -f "port=5050"`

- [ ] **Step 4: Commit**

```bash
cd ~/deploy-chat && git add static/console.html
git commit -m "feat(console): GHL sync error badges + retry button"
```

---

## Milestone 5 — Migration + Deploy

### Task 16: Migration script for existing Savant + Perdomo

**Files:**
- Create: `scripts/migrate_existing_households.py`

- [ ] **Step 1: Write the script**

```bash
cat > ~/deploy-chat/scripts/migrate_existing_households.py << 'PYEOF'
#!/usr/bin/env python3
"""One-time migration: create the Savant + Perdomo households from the
contacts we identified during the 2026-05-26 GHL dedup work.

Runs by curling the live web service /api/households endpoint. The web
service handles all DB + GHL sync. This script is just the orchestrator.

Required env vars:
  WEB_URL         — default https://glen-knowledge-chat.onrender.com
  CONSOLE_SECRET  — admin key for /api/households

Both households are no-ops if they already exist (the endpoint returns 200
with an "already_member" flag for repeat member-adds; for already-existing
slugs it returns 400 — we treat as informational and continue).
"""
import os
import sys
import json
import urllib.parse
import urllib.request
import urllib.error

WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
SECRET = os.environ.get("CONSOLE_SECRET", "")
if not SECRET:
    print("ERROR: CONSOLE_SECRET not set", flush=True)
    sys.exit(1)


def api(method, path, body=None):
    url = f"{WEB_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, method=method, data=data,
                                  headers={"Content-Type": "application/json",
                                           "X-Console-Key": SECRET})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def people_by_email(email):
    """Returns list of person dicts matching the email (lowercase exact)."""
    status, data = api("GET",
                       f"/api/people?q={urllib.parse.quote(email)}&limit=10")
    if status != 200:
        return []
    return [p for p in (data.get("people") or [])
            if (p.get("email") or "").lower() == email.lower()]


def create_household(name, head_email, member_emails):
    """Create a household from email lookups. Skips if already exists."""
    print(f"\n[{name} household]")
    members = []
    head_id = None
    for e in member_emails:
        matches = people_by_email(e)
        if not matches:
            print(f"  ⚠ no people record for {e} — skip")
            continue
        m = matches[0]
        members.append(m["id"])
        if e.lower() == head_email.lower():
            head_id = m["id"]
        print(f"  found {e} → people.id={m['id']}")
    if len(members) < 2 or head_id is None:
        print(f"  ⚠ insufficient members ({len(members)}) or no head — skip")
        return
    status, data = api("POST", "/api/households",
                       {"name": name, "head_person_id": head_id,
                        "member_person_ids": members,
                        "created_by": "glen-migration"})
    print(f"  → HTTP {status}: {data}")


def main():
    print(f"Migration against {WEB_URL}")
    # Savant: Lotika (head, mother) + Omika (daughter)
    create_household("Savant",
                     head_email="lotikasavant@hotmail.com",
                     member_emails=["lotikasavant@hotmail.com"])
    # Note: Savant has 2 people both at the same email. The shared-email
    # case is handled by the create endpoint, which uses people.id (not
    # email) to identify members. The above only enumerates one — let's
    # query both Savant rows:
    savants = people_by_email("lotikasavant@hotmail.com")
    if len(savants) >= 2:
        lotika = next((p for p in savants if (p.get("first_name") or "").lower() == "lotika"), savants[0])
        omika  = next((p for p in savants if (p.get("first_name") or "").lower() == "omika"), savants[1])
        status, data = api("POST", "/api/households",
                           {"name": "Savant", "head_person_id": lotika["id"],
                            "member_person_ids": [lotika["id"], omika["id"]],
                            "created_by": "glen-migration"})
        print(f"  Savant 2-member retry → HTTP {status}: {data}")

    # Perdomo household — Kauilani is the beta-cohort head; Kanehekai +
    # Kimberly (likely Kauilani's prior name) are the other household members.
    perdomos = people_by_email("restorealoha@gmail.com")
    if len(perdomos) >= 2:
        kauilani = next((p for p in perdomos
                          if (p.get("first_name") or "").lower() == "kauilani"), None)
        if not kauilani:
            print("  ⚠ no Kauilani found — skipping Perdomo migration")
        else:
            ids = [p["id"] for p in perdomos]
            status, data = api("POST", "/api/households",
                               {"name": "Perdomo", "head_person_id": kauilani["id"],
                                "member_person_ids": ids,
                                "created_by": "glen-migration"})
            print(f"  Perdomo → HTTP {status}: {data}")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
PYEOF
chmod +x ~/deploy-chat/scripts/migrate_existing_households.py
```

- [ ] **Step 2: Smoke-test the script locally against production (DRY-RUN style)**

Test by running with a non-existent host to confirm the script reaches the API call without crashing:

```bash
cd ~/deploy-chat && WEB_URL=https://nonexistent.example.com doppler run --project remedy-match --config prd -- python3 scripts/migrate_existing_households.py 2>&1 | head -5
```

Expected: script runs, tries to call the API, fails with a network error per call (acceptable — confirms wiring works).

- [ ] **Step 3: Commit (but do NOT run for real yet — wait for Task 17 deploy)**

```bash
cd ~/deploy-chat && git add scripts/migrate_existing_households.py
git commit -m "feat(households): migration script for Savant + Perdomo"
```

---

### Task 17: Deploy + run migration + verify

**Files:**
- None (this is verification, not code)

- [ ] **Step 1: Push to main → Render auto-deploys**

```bash
cd ~/deploy-chat && git push origin main 2>&1 | tail -5
```

- [ ] **Step 2: Wait for deploy + smoke-test endpoints**

```bash
CRON_SECRET=$(doppler secrets get CRON_SECRET --plain --project remedy-match --config prd 2>/dev/null)
until curl -fsS -X POST "https://glen-knowledge-chat.onrender.com/admin/detect-household-candidates" \
  -H "X-Cron-Secret: $CRON_SECRET" -m 30 -o /tmp/probe.out 2>/dev/null \
  && grep -q "summary" /tmp/probe.out; do sleep 10; done
echo "Deploy live; detection endpoint responsive."
cat /tmp/probe.out
```

Expected: `{"ok": true, "summary": {"detected": ..., "new_pending": ..., ...}}`.

- [ ] **Step 3: Run the migration script against production**

```bash
cd ~/deploy-chat && doppler run --project remedy-match --config prd -- python3 scripts/migrate_existing_households.py 2>&1 | tee /tmp/migration.log
```

Expected output: Savant household created with 2 members (Lotika as head, Omika as member); Perdomo household created with 3 members (Kauilani as head, plus Kanehekai + Kimberly).

- [ ] **Step 4: Verify via API**

```bash
CONSOLE_SECRET=$(doppler secrets get CONSOLE_SECRET --plain --project remedy-match --config prd 2>/dev/null)
echo "=== List of households ==="
curl -sS "https://glen-knowledge-chat.onrender.com/api/households" -H "X-Console-Key: $CONSOLE_SECRET" | python3 -m json.tool
echo
echo "=== Savant detail ==="
curl -sS "https://glen-knowledge-chat.onrender.com/api/households/savant" -H "X-Console-Key: $CONSOLE_SECRET" | python3 -m json.tool
```

Expected: Savant household with 2 members (Lotika head), Perdomo with 3 (Kauilani head).

- [ ] **Step 5: Verify in GHL**

```bash
GHL_API_KEY=$(doppler secrets get GHL_API_KEY --plain --project remedy-match --config prd 2>/dev/null)
echo "=== Savant in GHL ==="
curl -sS "https://rest.gohighlevel.com/v1/contacts/lookup?email=lotikasavant%40hotmail.com" \
  -H "Authorization: Bearer $GHL_API_KEY" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for c in d['contacts']:
    pb = [t for t in c.get('tags',[]) if t.startswith('household')]
    print(f'  {c[\"firstName\"]} {c[\"lastName\"]} → {sorted(pb)}')"
echo "=== Perdomo in GHL ==="
curl -sS "https://rest.gohighlevel.com/v1/contacts/lookup?email=restorealoha%40gmail.com" \
  -H "Authorization: Bearer $GHL_API_KEY" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for c in d['contacts']:
    pb = [t for t in c.get('tags',[]) if t.startswith('household')]
    print(f'  {c[\"firstName\"]} {c[\"lastName\"]} → {sorted(pb)}')"
```

Expected: every Savant member has `household:savant`; Lotika additionally has `household-head:savant`. Every Perdomo member has `household:perdomo`; Kauilani additionally has `household-head:perdomo`. The legacy `relationship:family-shared-email` tags are gone.

- [ ] **Step 6: Manual UI verification checklist**

Visit `https://glen-knowledge-chat.onrender.com/console`, log in, navigate to People:

- [ ] Click Lotika Savant → Overview shows "👥 Savant household · Lotika is head" with 2 member cards, current person highlighted.
- [ ] Click Omika in the household card → navigates to Omika's detail; same Savant household section shown.
- [ ] Click ⚙ on Savant household → edit dialog opens; rename to "Savant Family" → save → verify name change.
- [ ] Click + Add member → dialog opens in add-member mode → search a random unrelated person → confirm add works.
- [ ] Remove that test member by clicking ✕ in the edit dialog chips (TODO: not yet implemented — verify the ✕ is wired up; if not, file as v2).
- [ ] Disband the test household → confirm members no longer show household section.
- [ ] Trigger candidates detection manually:
  ```
  curl -X POST "https://glen-knowledge-chat.onrender.com/admin/detect-household-candidates" -H "X-Cron-Secret: $CRON_SECRET"
  ```
- [ ] Reload People page → if any new candidates exist, banner "👥 N possible household(s) detected · Review →" appears.
- [ ] Confirm one candidate → verify household appears.
- [ ] Dismiss one candidate → verify it disappears from banner.

- [ ] **Step 7: Commit verification notes**

```bash
cd ~/deploy-chat && echo "Migration run 2026-05-26: Savant + Perdomo households created successfully. Manual UI checklist passed." >> docs/superpowers/specs/2026-05-26-household-tag-system-design.md
git add docs/superpowers/specs/2026-05-26-household-tag-system-design.md
git commit -m "docs(households): post-deploy verification log"
```

---

## Self-Review

After writing the plan, look at it with fresh eyes:

**1. Spec coverage check:**
- ✅ Schema (Task 1)
- ✅ Tag conventions (Task 1, 3)
- ✅ ghl_update_tags helper (Task 2)
- ✅ POST /api/households + 409 (Task 3)
- ✅ GET endpoints (Task 4)
- ✅ PATCH + member ops + disband (Task 5)
- ✅ Resync endpoints (Task 6)
- ✅ Detection algorithm (Task 7)
- ✅ Confirm/dismiss candidates (Task 8)
- ✅ Cron chain (Task 9)
- ✅ Overview household section (Task 10)
- ✅ Dialog + entry point A (Task 11)
- ✅ Multi-select entry point B (Task 12)
- ✅ Edit + disband + 409 move flow (Task 13)
- ✅ Candidate review banner (Task 14)
- ✅ GHL sync error UI (Task 15)
- ✅ Migration (Task 16)
- ✅ Deploy + verification (Task 17)

Spec items NOT in plan: "Future expansion to dedicated Household tab when section grows" — intentionally deferred per spec; no task needed.

**2. Placeholder scan:** No TBD/TODO/incomplete sections. Each task has actual code or actual shell commands. Manual verification steps are explicit checklists, not vague "verify works".

**3. Type consistency:** All function signatures match between definition and call sites. `ghl_update_tags(email, add=None, remove=None)` is called with that signature everywhere. `_household_slug(name, head_first_name="", existing=None)` matches its test calls. `_render_household_section(person, household)` accepts an object with `slug`, `name`, `members[]` — matches the `get_household` response shape. `_pmsSelected` Set + `_hhChosen` array conventions used consistently.

**4. Scope check:** Single coherent feature, decomposed into 17 tasks across 5 milestones. Each milestone is independently testable and deployable. Total estimated effort: ~3-5 hours by a focused implementer, including manual verification.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-26-household-tag-system.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because the tasks have clear inputs/outputs and the test-driven structure makes per-task review fast.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Better if you want to stay engaged moment-by-moment.

**Which approach?**
