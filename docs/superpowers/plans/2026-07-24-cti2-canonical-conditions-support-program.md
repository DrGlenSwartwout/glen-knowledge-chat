# CTI-2 Slice 1 — canonical conditions drive the eye-condition support program — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A canonical condition (`person_attributes`, including a doc-approved one) drives a client's eye-condition support program, ranked ahead of `people` data but below the operator override — with zero change to that override's precedence and no writes to `people.tags`.

**Architecture:** One new best-effort helper in `app.py`, `_condition_detect_tags(cx, email)`, that returns the ordered auto-detect input — canonical conditions first, then `people.conditions`, then `people.tags`. The two existing condition resolvers (`_client_condition_for` and `api_console_client_condition_get`) are refactored to call it instead of duplicating an inline `people` read. Read-through: the store is read as authoritative, never projected into `people`.

**Tech Stack:** Python 3, Flask (`app.py`), `dashboard/canonical_tags.py` (existing dark store), SQLite in tests / Postgres in prod via `dashboard/db.py`. Tests extend the existing `tests/test_client_conditions.py` and `tests/test_client_condition_api.py` harnesses.

## Global Constraints

- **`people.tags` is never written by this work.** It has many writers (GHL, manual tagging); canonical conditions are *read alongside* it, never merged into it.
- **The operator override (`client_conditions`) precedence is unchanged** — it wins outright, applied by the caller *above* the helper. The helper returns detection input only and knows nothing about the override.
- **Best-effort, never raises.** A canonical read failure degrades to the people-only input. `_client_condition_for`'s existing `try/except → None` contract is preserved.
- **Canonical conditions rank first** in the returned list — `_condition_key_from_tags` returns the *first* unambiguous hit, so order is load-bearing.
- **No special-casing of ambiguity.** Canonical conditions flow through the same `_condition_key_from_tags`; a bare `"glaucoma"` stays ambiguous exactly as a `glaucoma` tag does.
- **No projection/denormalization.** Do not call or modify `canonical_tags.rebuild_people_columns` / `import_from_people`.
- **Running tests:** run the two targeted files during development. Do **not** run a bare full suite from a shell with real credentials (it sends real email); use `bash ci/run-tests.sh` for the gate.

---

## File Structure

**Modify:**
- `app.py` — add `_condition_detect_tags` (beside `_client_condition_for` at line ~21555); refactor `_client_condition_for` (line ~21555) and `api_console_client_condition_get` (line ~22356) to call it.

**Modify (tests):**
- `tests/test_client_conditions.py` — add helper unit tests + resolver behavior tests (extends the existing `app_mod` / `_seed_person` harness).
- `tests/test_client_condition_api.py` — add console-API parity tests (extends the existing `app_mod` / `HDRS` harness).

No new files, no new modules — this is a surgical read-through.

---

### Task 1: The `_condition_detect_tags` helper

**Files:**
- Modify: `app.py` (insert immediately before `def _client_condition_for` at line ~21555)
- Test: `tests/test_client_conditions.py` (extend)

**Interfaces:**
- Consumes: `canonical_tags.get_person(cx, email) -> {"conditions": [...], ...}` (existing).
- Produces: `_condition_detect_tags(cx, email) -> list[str]` — canonical conditions first, then `people.conditions`, then `people.tags`. Best-effort; never raises.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_client_conditions.py` (the `app_mod` fixture and `_seed_person` already exist in this file):

```python
# --- CTI-2: canonical conditions in the detection input --------------------
from dashboard import canonical_tags as _ct_seed


def _seed_canonical(db, email, conditions):
    with sqlite3.connect(db) as cx:
        for v in conditions:
            _ct_seed.set_attr(cx, email, "conditions", v, source="test")
        cx.commit()


def test_detect_tags_puts_canonical_conditions_first(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", conditions=["dry amd"], tags=["pb:dry-eye"])
    _seed_canonical(tmp_db, "jane@example.com", ["wet amd"])
    with sqlite3.connect(tmp_db) as cx:
        out = app_mod._condition_detect_tags(cx, "jane@example.com")
    assert out[0] == "wet amd"                       # canonical is first
    assert "dry amd" in out and "pb:dry-eye" in out  # people data still present


def test_detect_tags_people_only_when_no_canonical(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", conditions=["Wet AMD"], tags=["x"])
    with sqlite3.connect(tmp_db) as cx:
        out = app_mod._condition_detect_tags(cx, "jane@example.com")
    assert out == ["Wet AMD", "x"]


def test_detect_tags_canonical_only_when_no_people_row(app_mod, tmp_db):
    _seed_canonical(tmp_db, "jane@example.com", ["ocular hypertension"])
    with sqlite3.connect(tmp_db) as cx:
        out = app_mod._condition_detect_tags(cx, "jane@example.com")
    assert out == ["ocular hypertension"]


def test_detect_tags_empty_when_nothing(app_mod, tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        assert app_mod._condition_detect_tags(cx, "nobody@x.com") == []


def test_detect_tags_degrades_when_canonical_read_raises(app_mod, tmp_db, monkeypatch):
    _seed_person(tmp_db, "jane@example.com", conditions=["Wet AMD"])
    from dashboard import canonical_tags
    monkeypatch.setattr(canonical_tags, "get_person",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with sqlite3.connect(tmp_db) as cx:
        out = app_mod._condition_detect_tags(cx, "jane@example.com")   # must not raise
    assert out == ["Wet AMD"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_client_conditions.py -k detect_tags -v`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_condition_detect_tags'`

- [ ] **Step 3: Write the implementation**

Insert into `app.py` immediately before `def _client_condition_for` (line ~21555):

```python
def _condition_detect_tags(cx, email):
    """Ordered auto-detect input for a client's eye-condition support program:
    canonical conditions first (person_attributes -- Glen-approved and
    vocabulary-canonical), then people.conditions, then people.tags.

    This is ONLY the detection input for `_condition_key_from_tags`; the operator
    override is applied by the caller, ABOVE this. Order is load-bearing: the
    matcher returns the first unambiguous hit, so canonical conditions must lead.

    Best-effort: a canonical read failure degrades to the people-only input and
    never raises, and this function never reads or writes people.tags as a
    canonical target -- it only reads it as existing detection input.
    """
    email = (email or "").strip().lower()
    tags = []
    if not email:
        return tags
    # Canonical conditions first, so they outrank people-derived hits.
    try:
        from dashboard import canonical_tags as _ct
        for v in (_ct.get_person(cx, email).get("conditions") or []):
            if str(v).strip():
                tags.append(str(v))
    except Exception:
        pass
    # Then people.conditions, then people.tags -- unchanged order and semantics.
    # Positional indexing (row[0]=conditions, row[1]=tags) so this does not
    # depend on cx.row_factory (get_person mutates and restores it).
    try:
        row = cx.execute(
            "SELECT conditions, tags FROM people WHERE lower(email)=lower(?)",
            (email,)).fetchone()
    except Exception:
        row = None
    if row:
        for cell in (row[0], row[1]):
            try:
                v = json.loads(cell or "[]")
            except Exception:
                v = []
            if isinstance(v, list):
                tags.extend(str(x) for x in v)
    return tags
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_client_conditions.py -k detect_tags -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_client_conditions.py
git commit -m "feat(cti2): _condition_detect_tags — canonical conditions in the detection input"
```

---

### Task 2: Wire `_client_condition_for` to the helper

**Files:**
- Modify: `app.py` — `_client_condition_for` (line ~21555)
- Test: `tests/test_client_conditions.py` (extend)

**Interfaces:**
- Consumes: `_condition_detect_tags` (Task 1), `_condition_key_from_tags` (existing), `client_conditions.get` (existing).
- Produces: no signature change — `_client_condition_for(email) -> str | None`, same contract, now canonical-aware.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_client_conditions.py`:

```python
def test_resolver_override_still_wins_over_canonical(app_mod, tmp_db):
    _seed_canonical(tmp_db, "jane@example.com", ["wet amd"])
    with sqlite3.connect(tmp_db) as cx:
        from dashboard import client_conditions as _cc
        _cc.init_table(cx)
        _cc.set(cx, "jane@example.com", "dry-eye", "glen")
    assert app_mod._client_condition_for("jane@example.com") == "dry-eye"


def test_resolver_canonical_drives_program_when_no_override(app_mod, tmp_db):
    _seed_canonical(tmp_db, "jane@example.com", ["ocular hypertension"])
    assert app_mod._client_condition_for("jane@example.com") == "glaucoma-elevated-iop"


def test_resolver_canonical_outranks_conflicting_people_tag(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", tags=["pb:dry-amd"])
    _seed_canonical(tmp_db, "jane@example.com", ["wet amd"])
    assert app_mod._client_condition_for("jane@example.com") == "wet-amd"


def test_resolver_ambiguous_canonical_condition_returns_none(app_mod, tmp_db):
    _seed_canonical(tmp_db, "jane@example.com", ["glaucoma"])
    assert app_mod._client_condition_for("jane@example.com") is None


def test_resolver_people_only_unchanged_regression(app_mod, tmp_db):
    # No canonical row at all -> behaves exactly as before.
    _seed_person(tmp_db, "jane@example.com", conditions=["Wet AMD"])
    assert app_mod._client_condition_for("jane@example.com") == "wet-amd"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_client_conditions.py -k "resolver_canonical or resolver_override_still or resolver_ambiguous_canonical" -v`
Expected: FAIL — `test_resolver_canonical_drives_program_when_no_override` and `test_resolver_canonical_outranks_conflicting_people_tag` fail (resolver ignores canonical); the override/ambiguous/regression ones already pass.

- [ ] **Step 3: Write the implementation**

In `app.py`, replace the body of `_client_condition_for` from the `row = cx.execute(...)` block through `return _condition_key_from_tags(tags)`. The current code is:

```python
            override = _cc.get(cx, email)
            if override:
                return override
            row = cx.execute(
                "SELECT conditions, tags FROM people WHERE lower(email)=lower(?)",
                (email,)).fetchone()
            if not row:
                return None
            tags = []
            for col in ("conditions", "tags"):
                try:
                    v = json.loads(row[col] or "[]")
                except Exception:
                    v = []
                if isinstance(v, list):
                    tags.extend(str(x) for x in v)
            return _condition_key_from_tags(tags)
```

Replace it with:

```python
            override = _cc.get(cx, email)
            if override:
                return override
            return _condition_key_from_tags(_condition_detect_tags(cx, email))
```

Leave the surrounding `def`/docstring, the `email` normalization, the `if not email` guard, the `with db.connect(LOG_DB) as cx` / `cx.row_factory = sqlite3.Row` / `_cc.init_table(cx)`, and the outer `except Exception: return None` exactly as they are. Update the docstring's "auto-detect from the client's `people.conditions` + `people.tags`" line to "auto-detect from the client's canonical conditions + `people.conditions` + `people.tags`".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_client_conditions.py -v`
Expected: PASS — all new tests plus every pre-existing resolver test in the file (regression).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_client_conditions.py
git commit -m "feat(cti2): _client_condition_for reads canonical conditions (override still wins)"
```

---

### Task 3: Wire `api_console_client_condition_get` to the helper

**Files:**
- Modify: `app.py` — `api_console_client_condition_get` (line ~22356)
- Test: `tests/test_client_condition_api.py` (extend)

**Interfaces:**
- Consumes: `_condition_detect_tags` (Task 1).
- Produces: same route `GET /api/console/client-condition`, same response keys `{email, resolved, override, auto_detected, tags}`. Behavior change: `resolved`/`auto_detected` are now canonical-aware, and `tags` now includes canonical conditions (documented in the spec).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_client_condition_api.py` (the `app_mod` fixture, `HDRS`, and `_seed_person` already exist here):

```python
from dashboard import canonical_tags as _ct_api


def _seed_canonical(db, email, conditions):
    with sqlite3.connect(db) as cx:
        for v in conditions:
            _ct_api.set_attr(cx, email, "conditions", v, source="test")
        cx.commit()


def test_get_resolved_is_canonical_aware(app_mod, tmp_db):
    _seed_canonical(tmp_db, "jane@example.com", ["ocular hypertension"])
    r = app_mod.app.test_client().get(
        "/api/console/client-condition?email=jane@example.com", headers=HDRS)
    assert r.status_code == 200
    body = r.get_json()
    assert body["resolved"] == "glaucoma-elevated-iop"
    assert body["auto_detected"] == "glaucoma-elevated-iop"
    assert "ocular hypertension" in body["tags"]


def test_get_resolved_matches_resolver_across_cases(app_mod, tmp_db):
    # override case
    _seed_canonical(tmp_db, "a@x.com", ["wet amd"])
    with sqlite3.connect(tmp_db) as cx:
        from dashboard import client_conditions as _cc
        _cc.init_table(cx)
        _cc.set(cx, "a@x.com", "dry-eye", "glen")
    # canonical-only case
    _seed_canonical(tmp_db, "b@x.com", ["ocular hypertension"])
    # people-only case
    _seed_person(tmp_db, "c@x.com", conditions=["Wet AMD"])
    client = app_mod.app.test_client()
    for email in ("a@x.com", "b@x.com", "c@x.com"):
        body = client.get(f"/api/console/client-condition?email={email}",
                          headers=HDRS).get_json()
        assert body["resolved"] == app_mod._client_condition_for(email)


def test_get_override_still_wins_in_api(app_mod, tmp_db):
    _seed_canonical(tmp_db, "jane@example.com", ["wet amd"])
    with sqlite3.connect(tmp_db) as cx:
        from dashboard import client_conditions as _cc
        _cc.init_table(cx)
        _cc.set(cx, "jane@example.com", "dry-eye", "glen")
    body = app_mod.app.test_client().get(
        "/api/console/client-condition?email=jane@example.com",
        headers=HDRS).get_json()
    assert body["resolved"] == "dry-eye"
    assert body["override"] == "dry-eye"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_client_condition_api.py -k "canonical_aware or matches_resolver" -v`
Expected: FAIL — `resolved`/`auto_detected` ignore canonical (the route still does its inline people-only read).

- [ ] **Step 3: Write the implementation**

In `app.py`, `api_console_client_condition_get`, replace the inline people-read. The current block is:

```python
    tags = []
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_table(cx)
        override = _cc.get(cx, email)
        row = cx.execute(
            "SELECT conditions, tags FROM people WHERE lower(email)=lower(?)",
            (email,)).fetchone()
        if row:
            for col in ("conditions", "tags"):
                try:
                    v = json.loads(row[col] or "[]")
                except Exception:
                    v = []
                if isinstance(v, list):
                    tags.extend(str(x) for x in v)
    auto_detected = _condition_key_from_tags(tags)
```

Replace it with:

```python
    with db.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _cc.init_table(cx)
        override = _cc.get(cx, email)
        tags = _condition_detect_tags(cx, email)
    auto_detected = _condition_key_from_tags(tags)
```

Leave the auth guard, the `email` check, the `from dashboard import client_conditions as _cc` import, the `resolved = override or auto_detected`, and the `jsonify({...})` return unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_client_condition_api.py -v`
Expected: PASS — new tests plus every pre-existing test in the file (the pure people-only GET tests still pass because the helper returns the same list when there is no canonical row).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_client_condition_api.py
git commit -m "feat(cti2): console client-condition GET is canonical-aware (parity with resolver)"
```

---

### Task 4: Full-suite gate

**Files:** none changed unless the gate reports a new failure.

- [ ] **Step 1: Run the whole gated suite**

Run: `bash ci/run-tests.sh`
Expected: PASS. The script ratchets against `tests/known_failures.txt` and fails only on a NEW failure; it sets fake `PINECONE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `CONSOLE_SECRET` and unsets `DOPPLER_TOKEN`, which is what keeps a full run from touching real services.

- [ ] **Step 2: If a NEW failure appears, fix the cause**

The most likely source is a pre-existing test that pins `_client_condition_for` or the console `tags` response for a client that ALSO happens to have canonical conditions — unlikely on a fresh checkout (no canonical rows exist), but if one appears, confirm the new behavior is correct before adjusting the assertion, and never add the test to `known_failures.txt`.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "test(cti2): keep the CI ratchet green"
```

---

## Self-Review notes (for the implementer)

- The helper is the only new code; both wirings are one-line swaps that delete duplicated inline reads. If a wiring diff is larger than a few lines, something drifted — re-check against the exact blocks quoted above.
- Do not "improve" the helper to also merge canonical `tags`/`terrain_concerns` into the detection input — only `conditions` feed the eye-condition program in this slice (spec, out-of-scope).
- Do not touch `people.tags`, `rebuild_people_columns`, or `import_from_people`.
