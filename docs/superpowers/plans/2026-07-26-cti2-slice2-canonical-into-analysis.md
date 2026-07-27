# CTI-2 Slice 2 — canonical attributes reach the biofield analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A canonical/doc-approved clinical attribute shows up in the biofield causal-chain narrative Glen reviews — by merging `canonical_tags.get_person` into what `/api/people` returns, with zero change to the narrative or the local app.

**Architecture:** One pure helper `_merge_canonical_into_person(cx, person)` in `app.py` that unions canonical clinical fields into a `/api/people` row, preserving each field's serialized shape. Wired into `get_people` (list) and `get_person` (by id). Read-through — no writes to `people`, `people.tags` never touched.

**Tech Stack:** Python 3, Flask (`app.py`), `dashboard/canonical_tags.py` (existing store), SQLite (tests) / Postgres (prod). Tests mirror `tests/test_people_tags_api.py`.

## Global Constraints

- **`people.tags` is never read-merged or written.** It is the CRM/GHL bucket; `/api/people` feeds the console people list. Merge only `conditions`, `terrain_concerns`, `body_systems`, `challenges`, `goals`.
- **Preserve each field's serialized shape.** Discrete fields are JSON-string lists (`json.dumps([...])`); scalar fields (`challenges`, `goals`) are plain strings. The merged output must keep that shape so the console people view and the narrative's `_profile_block` are unaffected.
- **Read-through, not projection.** Do NOT call `rebuild_people_columns`/`import_from_people`; do NOT write any `people` column. Canonical is merged into the *response* only.
- **Best-effort, never breaks the endpoint.** Any failure resolving/reading canonical for a person returns that person unchanged; `get_people` returning 200 must not depend on the merge succeeding.
- **Discrete merge = union + case-insensitive dedup, first-seen casing kept** (people items first, then canonical).
- Console-gate on both routes is unchanged. NEVER `cur.lastrowid` in new code.
- **Running tests:** run the targeted file; do NOT run a bare full suite from a shell with real creds (sends real email) — use `bash ci/run-tests.sh` for the gate.

---

## File Structure

**Modify:**
- `app.py` — add `_merge_canonical_into_person` (near `get_people`, ~line 34032); call it in `get_people` (~34063) and `get_person` (~34100).

**Create (test):**
- `tests/test_people_canonical_merge.py`

---

### Task 1: The `_merge_canonical_into_person` helper

**Files:**
- Modify: `app.py` (insert just before `def get_people`, ~line 34063; module constants near the other `_people_*` helpers)
- Test: `tests/test_people_canonical_merge.py`

**Interfaces:**
- Consumes: `canonical_tags.get_person(cx, email) -> {conditions:[...], terrain_concerns:[...], body_systems:[...], challenges:str, goals:str, tags:[...]}`.
- Produces: `_merge_canonical_into_person(cx, person: dict) -> dict` (mutates + returns `person`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_people_canonical_merge.py
"""Unit tests for _merge_canonical_into_person (app.py): read-through merge of
canonical clinical attributes into a /api/people row. Env-gated like the other
api tests (importing app builds OpenAI/Pinecone clients)."""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("requires app env (use doppler run / CI)", allow_module_level=True)

import app  # noqa: E402
from dashboard import canonical_tags as ct  # noqa: E402


def _cx():
    cx = sqlite3.connect(":memory:")
    ct.init_tables(cx)
    return cx


def _seed_canon(cx, email, **fields):
    for f, vals in fields.items():
        vals = vals if isinstance(vals, (list, tuple)) else [vals]
        for v in vals:
            ct.set_attr(cx, email, f, v, source="test")
    cx.commit()


def test_discrete_union_preserves_json_string_shape():
    cx = _cx()
    _seed_canon(cx, "c@x.com", conditions="ocular hypertension")
    person = {"email": "c@x.com", "conditions": json.dumps(["glaucoma"]),
              "tags": json.dumps(["vip"])}
    out = app._merge_canonical_into_person(cx, person)
    assert json.loads(out["conditions"]) == ["glaucoma", "ocular hypertension"]
    assert isinstance(out["conditions"], str)          # still a JSON string


def test_discrete_dedup_is_case_insensitive_first_seen_wins():
    cx = _cx()
    _seed_canon(cx, "c@x.com", conditions="glaucoma")
    person = {"email": "c@x.com", "conditions": json.dumps(["Glaucoma"])}
    out = app._merge_canonical_into_person(cx, person)
    assert json.loads(out["conditions"]) == ["Glaucoma"]   # people casing kept, one item


def test_tags_is_never_merged():
    cx = _cx()
    _seed_canon(cx, "c@x.com", tags="from-canonical")
    person = {"email": "c@x.com", "tags": json.dumps(["crm-tag"])}
    out = app._merge_canonical_into_person(cx, person)
    assert json.loads(out["tags"]) == ["crm-tag"]          # canonical tag NOT added


def test_scalar_canonical_wins_when_present_else_people():
    cx = _cx()
    _seed_canon(cx, "a@x.com", challenges="fatigue")
    a = app._merge_canonical_into_person(cx, {"email": "a@x.com", "challenges": ""})
    assert a["challenges"] == "fatigue"
    # canonical empty -> people value kept
    b = app._merge_canonical_into_person(cx, {"email": "b@x.com", "goals": "sleep"})
    assert b["goals"] == "sleep"


def test_no_canonical_row_returns_person_unchanged():
    cx = _cx()
    person = {"email": "nobody@x.com", "conditions": json.dumps(["x"]),
              "challenges": "y"}
    out = app._merge_canonical_into_person(cx, dict(person))
    assert json.loads(out["conditions"]) == ["x"] and out["challenges"] == "y"


def test_empty_everything_leaves_field_untouched():
    cx = _cx()
    person = {"email": "c@x.com", "conditions": None}
    out = app._merge_canonical_into_person(cx, person)
    assert out["conditions"] is None                        # not forced to "[]"


def test_malformed_people_json_degrades_to_canonical():
    cx = _cx()
    _seed_canon(cx, "c@x.com", conditions="glaucoma")
    person = {"email": "c@x.com", "conditions": "not json"}
    out = app._merge_canonical_into_person(cx, person)
    assert json.loads(out["conditions"]) == ["glaucoma"]


def test_best_effort_get_person_raises_returns_unchanged(monkeypatch):
    cx = _cx()
    monkeypatch.setattr(ct, "get_person",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    person = {"email": "c@x.com", "conditions": json.dumps(["x"])}
    out = app._merge_canonical_into_person(cx, person)      # must not raise
    assert json.loads(out["conditions"]) == ["x"]


def test_blank_email_returns_unchanged():
    cx = _cx()
    person = {"email": "", "conditions": json.dumps(["x"])}
    assert app._merge_canonical_into_person(cx, person)["conditions"] == json.dumps(["x"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_people_canonical_merge.py -v` (with dummy env keys: `PINECONE_API_KEY=pc-x OPENAI_API_KEY=sk-x ANTHROPIC_API_KEY=sk-x CONSOLE_SECRET=x` so the module doesn't skip)
Expected: FAIL — `AttributeError: module 'app' has no attribute '_merge_canonical_into_person'`

- [ ] **Step 3: Write the implementation**

Insert into `app.py` just before `def get_people` (near the `_people_search_query` helper, ~line 34032):

```python
_CANON_MERGE_DISCRETE = ("conditions", "terrain_concerns", "body_systems")
_CANON_MERGE_SCALAR = ("challenges", "goals")


def _merge_canonical_into_person(cx, person):
    """Read-through: union the client's canonical clinical attributes
    (canonical_tags.person_attributes) into a /api/people row, so doc-approved
    conditions/terrain/systems reach the biofield narrative's profile block.

    Preserves each field's serialized shape (discrete = JSON-string list, scalar
    = plain string) so the console people view and the narrative are unaffected.
    NEVER touches `tags` (the CRM/GHL bucket). Best-effort: any failure returns
    the person unchanged. Mutates and returns `person`.
    """
    email = (person.get("email") or "").strip()
    if not email:
        return person
    try:
        from dashboard import canonical_tags as _ct
        canon = _ct.get_person(cx, email)
    except Exception:
        return person
    for f in _CANON_MERGE_DISCRETE:
        try:
            existing = json.loads(person.get(f) or "[]")
            if not isinstance(existing, list):
                existing = []
        except (TypeError, ValueError):
            existing = []
        merged, seen = [], set()
        for v in list(existing) + list(canon.get(f) or []):
            s = str(v).strip()
            k = s.lower()
            if s and k not in seen:
                seen.add(k)
                merged.append(s)
        if merged:                       # leave the field untouched when empty
            person[f] = json.dumps(merged)
    for f in _CANON_MERGE_SCALAR:
        cv = str(canon.get(f) or "").strip()
        if cv:
            person[f] = cv
    return person
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_people_canonical_merge.py -v` (with the dummy env keys)
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_people_canonical_merge.py
git commit -m "feat(cti2): _merge_canonical_into_person — read-through canonical merge for /api/people"
```

---

### Task 2: Wire the helper into `get_people` and `get_person`

**Files:**
- Modify: `app.py` — `get_people` (~34063), `get_person` (~34100)
- Test: `tests/test_people_canonical_merge.py` (extend with endpoint tests)

**Interfaces:**
- Consumes: `_merge_canonical_into_person` (Task 1).
- Produces: no signature change — `GET /api/people` and `GET /api/people/<id>` now return canonical-merged clinical fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_people_canonical_merge.py`:

```python
# --- endpoint integration -------------------------------------------------

@pytest.fixture
def app_env(tmp_path, monkeypatch):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        cx.execute(
            "CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "email TEXT UNIQUE, name TEXT, first_name TEXT, last_name TEXT, "
            "phone TEXT, city TEXT, state TEXT, country TEXT, island TEXT, "
            "profession TEXT, title TEXT, organizations TEXT, ghl_id TEXT, "
            "source TEXT, tags TEXT DEFAULT '[]', roles TEXT, challenges TEXT, "
            "goals TEXT, terrain_concerns TEXT DEFAULT '[]', "
            "body_systems TEXT DEFAULT '[]', conditions TEXT DEFAULT '[]', "
            "order_count INTEGER, last_order_date TEXT, session_count INTEGER, "
            "last_session_date TEXT, last_contact_date TEXT, synced_at TEXT)")
        cx.commit()
    monkeypatch.setattr(app, "LOG_DB", p)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    return p


def _seed_person_row(db, email, **cols):
    keys = ["email"] + list(cols)
    vals = [email] + list(cols.values())
    with sqlite3.connect(db) as cx:
        cx.execute(f"INSERT INTO people ({','.join(keys)}) VALUES "
                   f"({','.join('?' * len(keys))})", vals)
        cx.commit()


def _seed_canon_db(db, email, **fields):
    with sqlite3.connect(db) as cx:
        ct.init_tables(cx)
        for f, vals in fields.items():
            for v in (vals if isinstance(vals, (list, tuple)) else [vals]):
                ct.set_attr(cx, email, f, v, source="test")
        cx.commit()


def test_get_people_merges_canonical_condition(app_env):
    _seed_person_row(app_env, "c@x.com", conditions=json.dumps(["glaucoma"]),
                     tags=json.dumps(["vip"]))
    _seed_canon_db(app_env, "c@x.com", conditions="ocular hypertension")
    r = app.app.test_client().get("/api/people?q=c@x.com",
                                  headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    person = next(p for p in r.get_json()["people"] if p["email"] == "c@x.com")
    assert set(json.loads(person["conditions"])) == {"glaucoma", "ocular hypertension"}
    assert json.loads(person["tags"]) == ["vip"]            # tags untouched


def test_get_people_requires_console_key(app_env):
    r = app.app.test_client().get("/api/people?q=c@x.com")
    assert r.status_code == 401


def test_get_people_canonical_failure_returns_people_unchanged(app_env, monkeypatch):
    _seed_person_row(app_env, "c@x.com", conditions=json.dumps(["glaucoma"]))
    monkeypatch.setattr(ct, "get_person",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = app.app.test_client().get("/api/people?q=c@x.com",
                                  headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    person = next(p for p in r.get_json()["people"] if p["email"] == "c@x.com")
    assert json.loads(person["conditions"]) == ["glaucoma"]


def test_get_person_by_id_merges_canonical(app_env):
    _seed_person_row(app_env, "c@x.com", conditions=json.dumps(["glaucoma"]))
    _seed_canon_db(app_env, "c@x.com", terrain_concerns="oxidative stress")
    with sqlite3.connect(app_env) as cx:
        pid = cx.execute("SELECT id FROM people WHERE email='c@x.com'").fetchone()[0]
    r = app.app.test_client().get(f"/api/people/{pid}",
                                  headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    assert json.loads(r.get_json()["terrain_concerns"]) == ["oxidative stress"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_people_canonical_merge.py -k "get_people or get_person_by_id" -v` (dummy env keys)
Expected: FAIL — `test_get_people_merges_canonical_condition` and `test_get_person_by_id_merges_canonical` fail (endpoints don't merge yet).

- [ ] **Step 3: Write the implementation**

In `app.py` `get_people` (~34063), the current tail is:

```python
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        total = cx.execute(f"SELECT COUNT(*) FROM people {where}", args).fetchone()[0]
        rows  = cx.execute(
            f"SELECT id,email,name,... FROM people {where} ORDER BY ... LIMIT ? OFFSET ?",
            args + [limit, offset]
        ).fetchall()
    return jsonify({"total": total, "people": [dict(r) for r in rows]})
```

Change the final two lines so the merge runs while `cx` is open:

```python
        rows  = cx.execute(
            f"SELECT id,email,name,... FROM people {where} ORDER BY ... LIMIT ? OFFSET ?",
            args + [limit, offset]
        ).fetchall()
        people = [_merge_canonical_into_person(cx, dict(r)) for r in rows]
    return jsonify({"total": total, "people": people})
```

(Leave the `SELECT` column list exactly as it is — do not retype it; only add the `people = [...]` line inside the `with` block and change the `return` to use `people`.)

In `app.py` `get_person` (~34100), the current body is:

```python
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
    if not row:
        return jsonify({"error":"Not found"}), 404
    return jsonify(dict(row))
```

Change it to merge while `cx` is open:

```python
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
        person = _merge_canonical_into_person(cx, dict(row)) if row else None
    if not person:
        return jsonify({"error":"Not found"}), 404
    return jsonify(person)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_people_canonical_merge.py -v` (dummy env keys)
Expected: PASS (all — 9 helper + 4 endpoint).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_people_canonical_merge.py
git commit -m "feat(cti2): /api/people GET merges canonical clinical attributes into each person"
```

---

### Task 3: Full-suite gate

- [ ] **Step 1: Run the whole gated suite**

Run: `bash ci/run-tests.sh`
Expected: PASS (ratchets against `tests/known_failures.txt`, fails only on a NEW failure; sets fake keys, unsets `DOPPLER_TOKEN`).

- [ ] **Step 2: If a NEW failure appears, fix the cause**

Most likely an existing `/api/people` test that asserts an exact clinical-field value for a person who ALSO has a canonical row (unlikely on a fresh checkout — no canonical rows exist). If one appears, confirm the merged value is correct before adjusting the assertion; never add to `known_failures.txt`.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "test(cti2): keep the CI ratchet green"
```

---

## Self-Review notes (for the implementer)

- Only `app.py` (helper + two call sites) and the one test file change. If a diff touches `dashboard/canonical_tags.py`, `biofield_narrative.py`, or `biofield_local_app.py`, it's out of scope.
- Never merge or write `tags`. Never write any `people` column.
- Preserve field shape: discrete → JSON string, scalar → plain string. When a discrete merge is empty, leave the field untouched (don't force `"[]"`).
- The merge runs inside the `with db.connect(...)` block (canonical `get_person` needs the open `cx`).

## Out of scope (Slice 3)

The gated canonical signal at the remedy-matcher review surface — its own spec after this ships. Needs recon on where `ff_matcher.generate_ff_matches` output is presented to Glen.
