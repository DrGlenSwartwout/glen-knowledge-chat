# Canonical Tagging CTI-1 — Canonical Store + Vocabulary — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the in-house authoritative store for a person's clinical attributes — controlled-vocabulary/alias mechanism, per-person canonical values, a one-time import from current `people.*`, and a regenerate-the-columns function — all dark and non-breaking.

**Architecture:** One new pure module `dashboard/canonical_tags.py` (cx-based, mirrors `dashboard/biofield_meanings.py`). Two tables: `canonical_vocab` (alias→canonical per discrete field) and `person_attributes` (per-person values with source). The store regenerates the legacy `people.*` columns one-way, so nothing downstream changes.

**Tech Stack:** Python 3.11, sqlite3, json, re, pytest.

## Global Constraints

- One new file only: `dashboard/canonical_tags.py` (+ tests). NO app.py/endpoint/route/cron change. Dark/non-breaking — no caller wired (CTI-2 does that).
- Discrete (vocab-controlled, JSON-array columns): `tags, conditions, terrain_concerns, body_systems`. Scalar (free-text columns): `challenges, goals`.
- `canonical_vocab` seeded EMPTY in CTI-1 → `resolve` falls back to the cleaned value (behaves as plain normalization until curated in CTI-4).
- `person_attributes` dedup: discrete fields many rows per (email,field) deduped by `value_norm`; scalar fields exactly one row per (email,field) (replace). `source ∈ manual|ai|ghl|rule|scan|import`.
- email is lowercased everywhere; values are resolved before storage; empty values are no-ops; never raise on bad input.
- `import_from_people` tags everything `source='import'`; idempotent on re-run.
- Pure module: only `json`, `re`, `sqlite3`, `datetime` imports. Run tests: `cd /tmp/wt-cti-82bd74c2 && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v`.

---

### Task 1: Tables + resolve + set_attr (write/vocab core)

**Files:**
- Create: `dashboard/canonical_tags.py`
- Test: `tests/test_canonical_tags_store.py`

**Interfaces:**
- Produces: module constants `DISCRETE_FIELDS`, `SCALAR_FIELDS`, `ALL_FIELDS`; `init_tables(cx)`; `_norm(s)`; `resolve(cx, field, value) -> str` (discrete: vocab canonical or cleaned; scalar: cleaned; empty→""); `set_attr(cx, email, field, value, *, source) -> bool` (discrete: INSERT OR IGNORE deduped by value_norm; scalar: replace single row; lowercases email; resolves value; no-op on empty/unknown field).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_tags_store.py
import sqlite3
from dashboard.canonical_tags import init_tables, resolve, set_attr


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_tables(cx)
    return cx


def test_discrete_dedup_by_norm(tmp_path):
    cx = _cx(tmp_path)
    assert set_attr(cx, "J@x.com", "conditions", "Eczema", source="manual") is True
    assert set_attr(cx, "j@x.com", "conditions", " eczema ", source="ai") is False   # norm-dup
    rows = cx.execute("SELECT email, value, source FROM person_attributes WHERE field='conditions'").fetchall()
    assert rows == [("j@x.com", "Eczema", "manual")]          # email lowercased, first source kept


def test_scalar_replace(tmp_path):
    cx = _cx(tmp_path)
    set_attr(cx, "j@x.com", "goals", "more energy", source="import")
    set_attr(cx, "j@x.com", "goals", "sleep better", source="manual")
    rows = cx.execute("SELECT value FROM person_attributes WHERE field='goals'").fetchall()
    assert rows == [("sleep better",)]                        # single row, replaced


def test_resolve_vocab_alias_discrete_only(tmp_path):
    cx = _cx(tmp_path)
    cx.execute("INSERT INTO canonical_vocab(field,alias_norm,canonical) VALUES('conditions','adrenal exhaustion','Adrenal Fatigue')")
    cx.commit()
    assert resolve(cx, "conditions", "Adrenal  Exhaustion") == "Adrenal Fatigue"   # alias->canonical
    assert resolve(cx, "conditions", "Unmapped Thing") == "Unmapped Thing"          # fallback cleaned
    assert resolve(cx, "goals", "adrenal exhaustion") == "adrenal exhaustion"       # scalar ignores vocab (cleaned)


def test_empty_and_bad_field_noop(tmp_path):
    cx = _cx(tmp_path)
    assert set_attr(cx, "j@x.com", "conditions", "   ", source="manual") is False
    assert set_attr(cx, "", "conditions", "x", source="manual") is False
    assert set_attr(cx, "j@x.com", "not_a_field", "x", source="manual") is False
    assert cx.execute("SELECT COUNT(*) FROM person_attributes").fetchone()[0] == 0
```

- [ ] **Step 2: Run** → FAIL (module missing).

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_canonical_tags_store.py -v`

- [ ] **Step 3: Implement**

```python
# dashboard/canonical_tags.py
"""In-house canonical store for a person's clinical attributes (CTI-1 foundation).
Authoritative store + controlled-vocabulary/alias mechanism. Pure: takes a sqlite
connection; mirrors dashboard/biofield_meanings.py. Dark/non-breaking in CTI-1 —
no caller is wired yet (CTI-2 makes it authoritative)."""
import json
import re
import sqlite3
from datetime import datetime, timezone

DISCRETE_FIELDS = ("tags", "conditions", "terrain_concerns", "body_systems")
SCALAR_FIELDS = ("challenges", "goals")
ALL_FIELDS = DISCRETE_FIELDS + SCALAR_FIELDS


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(s):
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"^[^\w]+|[^\w]+$", "", s)


def _clean(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def init_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS canonical_vocab ("
               "field TEXT, alias_norm TEXT, canonical TEXT, "
               "PRIMARY KEY(field, alias_norm))")
    cx.execute("CREATE TABLE IF NOT EXISTS person_attributes ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, field TEXT, "
               "value TEXT, value_norm TEXT, source TEXT, added_at TEXT, "
               "UNIQUE(email, field, value_norm))")
    cx.commit()


def resolve(cx, field, value):
    v = _clean(value)
    if not v:
        return ""
    if field in DISCRETE_FIELDS:
        row = cx.execute(
            "SELECT canonical FROM canonical_vocab WHERE field=? AND alias_norm=?",
            (field, _norm(v))).fetchone()
        if row and (row[0] or "").strip():
            return row[0].strip()
    return v


def set_attr(cx, email, field, value, *, source):
    init_tables(cx)
    email = (email or "").strip().lower()
    if not email or field not in ALL_FIELDS:
        return False
    canon = resolve(cx, field, value)
    if not canon:
        return False
    now, vn = _now(), _norm(canon)
    if field in SCALAR_FIELDS:
        cx.execute("DELETE FROM person_attributes WHERE email=? AND field=?", (email, field))
        cx.execute(
            "INSERT INTO person_attributes(email,field,value,value_norm,source,added_at) "
            "VALUES(?,?,?,?,?,?)", (email, field, canon, vn, source, now))
        cx.commit()
        return True
    cur = cx.execute(
        "INSERT OR IGNORE INTO person_attributes(email,field,value,value_norm,source,added_at) "
        "VALUES(?,?,?,?,?,?)", (email, field, canon, vn, source, now))
    cx.commit()
    return cur.rowcount > 0
```

- [ ] **Step 4: Run** → PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/canonical_tags.py tests/test_canonical_tags_store.py
git commit -m "feat(cti1): canonical store tables + resolve + set_attr"
```

---

### Task 2: `get_person` + `rebuild_people_columns`

**Files:**
- Modify: `dashboard/canonical_tags.py`
- Test: `tests/test_canonical_tags_rebuild.py`

**Interfaces:**
- Consumes: Task 1 functions.
- Produces: `get_person(cx, email) -> dict` (all 6 keys; discrete → sorted canonical list, scalar → value or ""); `rebuild_people_columns(cx, email)` (UPDATE the `people` row's 6 columns from `get_person` — discrete as `json.dumps` lists, scalar as text; no-op if no people row).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_tags_rebuild.py
import json
import sqlite3
from dashboard.canonical_tags import get_person, init_tables, rebuild_people_columns, set_attr


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_tables(cx)
    cx.execute("CREATE TABLE people(email TEXT, tags TEXT DEFAULT '[]', conditions TEXT DEFAULT '[]', "
               "terrain_concerns TEXT DEFAULT '[]', body_systems TEXT DEFAULT '[]', "
               "challenges TEXT DEFAULT '', goals TEXT DEFAULT '')")
    cx.execute("INSERT INTO people(email) VALUES('j@x.com')")
    cx.commit()
    return cx


def test_get_person_reconstructs(tmp_path):
    cx = _cx(tmp_path)
    set_attr(cx, "j@x.com", "conditions", "Eczema", source="manual")
    set_attr(cx, "j@x.com", "conditions", "Asthma", source="manual")
    set_attr(cx, "j@x.com", "goals", "sleep better", source="manual")
    p = get_person(cx, "J@x.com")
    assert p["conditions"] == ["Asthma", "Eczema"]           # sorted
    assert p["goals"] == "sleep better"
    assert p["tags"] == [] and p["challenges"] == ""         # all keys present
    assert get_person(cx, "nobody@x.com")["conditions"] == []


def test_rebuild_writes_people_columns(tmp_path):
    cx = _cx(tmp_path)
    set_attr(cx, "j@x.com", "conditions", "Eczema", source="manual")
    set_attr(cx, "j@x.com", "body_systems", "Liver", source="manual")
    set_attr(cx, "j@x.com", "challenges", "always tired", source="manual")
    rebuild_people_columns(cx, "j@x.com")
    row = cx.execute("SELECT conditions, body_systems, challenges, tags FROM people "
                     "WHERE email='j@x.com'").fetchone()
    assert json.loads(row[0]) == ["Eczema"] and json.loads(row[1]) == ["Liver"]
    assert row[2] == "always tired" and json.loads(row[3]) == []
```

- [ ] **Step 2: Run** → FAIL (`get_person`/`rebuild_people_columns` undefined).

- [ ] **Step 3: Implement** — append to `dashboard/canonical_tags.py`:

```python
def get_person(cx, email):
    init_tables(cx)
    cx.row_factory = sqlite3.Row
    email = (email or "").strip().lower()
    out = {f: ([] if f in DISCRETE_FIELDS else "") for f in ALL_FIELDS}
    for r in cx.execute(
            "SELECT field, value FROM person_attributes WHERE email=?", (email,)).fetchall():
        f = r["field"]
        if f in DISCRETE_FIELDS:
            out[f].append(r["value"])
        elif f in SCALAR_FIELDS:
            out[f] = r["value"]
    for f in DISCRETE_FIELDS:
        out[f] = sorted(out[f])
    return out


def rebuild_people_columns(cx, email):
    email = (email or "").strip().lower()
    p = get_person(cx, email)
    cx.execute(
        "UPDATE people SET tags=?, conditions=?, terrain_concerns=?, body_systems=?, "
        "challenges=?, goals=? WHERE lower(email)=?",
        (json.dumps(p["tags"]), json.dumps(p["conditions"]),
         json.dumps(p["terrain_concerns"]), json.dumps(p["body_systems"]),
         p["challenges"], p["goals"], email))
    cx.commit()
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_canonical_tags_rebuild.py tests/test_canonical_tags_store.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/canonical_tags.py tests/test_canonical_tags_rebuild.py
git commit -m "feat(cti1): get_person + rebuild_people_columns"
```

---

### Task 3: `import_from_people` (one-time seed)

**Files:**
- Modify: `dashboard/canonical_tags.py`
- Test: `tests/test_canonical_tags_import.py`

**Interfaces:**
- Consumes: Task 1/2 functions.
- Produces: `import_from_people(cx) -> {"persons": int, "attrs": int}` — for each `people` row with an email, parse the 4 discrete JSON-array columns + 2 scalar columns and `set_attr(..., source='import')`; idempotent (dedup); bad/empty JSON tolerated. Adds `_parse_list(s) -> list[str]` helper.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_tags_import.py
import json
import sqlite3
from dashboard.canonical_tags import import_from_people, init_tables, get_person


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_tables(cx)
    cx.execute("CREATE TABLE people(email TEXT, tags TEXT, conditions TEXT, "
               "terrain_concerns TEXT, body_systems TEXT, challenges TEXT, goals TEXT)")
    cx.execute("INSERT INTO people VALUES(?,?,?,?,?,?,?)",
               ("j@x.com", json.dumps(["type:client", "Inflammation"]),
                json.dumps(["Eczema"]), "not json", "[]", "always tired", "more energy"))
    cx.execute("INSERT INTO people VALUES(?,?,?,?,?,?,?)",
               ("", "[]", "[]", "[]", "[]", "", ""))            # blank email skipped
    cx.commit()
    return cx


def test_import_seeds_store_from_people(tmp_path):
    cx = _cx(tmp_path)
    res = import_from_people(cx)
    assert res["persons"] == 1                                 # blank-email row skipped
    p = get_person(cx, "j@x.com")
    assert set(p["tags"]) == {"type:client", "Inflammation"}
    assert p["conditions"] == ["Eczema"] and p["terrain_concerns"] == []   # bad JSON -> []
    assert p["challenges"] == "always tired" and p["goals"] == "more energy"
    # all imported with source='import'
    srcs = {r[0] for r in cx.execute("SELECT DISTINCT source FROM person_attributes").fetchall()}
    assert srcs == {"import"}


def test_import_idempotent(tmp_path):
    cx = _cx(tmp_path)
    import_from_people(cx)
    n1 = cx.execute("SELECT COUNT(*) FROM person_attributes").fetchone()[0]
    import_from_people(cx)                                     # re-run
    n2 = cx.execute("SELECT COUNT(*) FROM person_attributes").fetchone()[0]
    assert n1 == n2 and n1 > 0
```

- [ ] **Step 2: Run** → FAIL (`import_from_people` undefined).

- [ ] **Step 3: Implement** — append to `dashboard/canonical_tags.py`:

```python
def _parse_list(s):
    try:
        v = json.loads(s or "[]")
    except Exception:
        return []
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def import_from_people(cx):
    init_tables(cx)
    cx.row_factory = sqlite3.Row
    rows = cx.execute(
        "SELECT email, tags, conditions, terrain_concerns, body_systems, challenges, goals "
        "FROM people WHERE TRIM(COALESCE(email,''))<>''").fetchall()
    persons, attrs = 0, 0
    for r in rows:
        persons += 1
        for f in DISCRETE_FIELDS:
            for val in _parse_list(r[f]):
                if set_attr(cx, r["email"], f, val, source="import"):
                    attrs += 1
        for f in SCALAR_FIELDS:
            if (r[f] or "").strip() and set_attr(cx, r["email"], f, r[f], source="import"):
                attrs += 1
    return {"persons": persons, "attrs": attrs}
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_canonical_tags_import.py tests/test_canonical_tags_store.py tests/test_canonical_tags_rebuild.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/canonical_tags.py tests/test_canonical_tags_import.py
git commit -m "feat(cti1): import_from_people one-time seed"
```

---

## Self-Review

**Spec coverage:**
- Tables (canonical_vocab + person_attributes) + _norm → Task 1. ✓
- resolve (vocab discrete / passthrough scalar, empty-aware) + set_attr (discrete dedup / scalar replace) → Task 1. ✓
- get_person reconstruct + rebuild_people_columns one-way regenerate → Task 2. ✓
- import_from_people (source='import', idempotent, bad-JSON tolerant, blank-email skip) → Task 3. ✓
- Dark/non-breaking, pure module, no app.py change → all tasks (only canonical_tags.py + tests). ✓

**Placeholder scan:** No TBDs; complete code in every step.

**Type consistency:** `set_attr(cx,email,field,value,*,source)->bool` (T1) used by `import_from_people` (T3); `get_person->dict` (T2) used by `rebuild_people_columns` (T2) + import test; `resolve` (T1) used by set_attr; field-set constants shared. Consistent.

## Verification (manual, after all tasks)

`~/.venvs/deploy-chat311/bin/python -m pytest tests/test_canonical_tags_*.py -v` → all green. The module is dark (no caller); CTI-2 will run `import_from_people` on prod (console-gated, dry-run first), stop the GHL overwrite, make `rebuild_people_columns` the writer, and add push-out to GHL.
