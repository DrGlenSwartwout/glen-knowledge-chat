# Console Core-Field Editing (E1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FMP-sourced core fields (names, common_names, par levels, source prices, recipe doses) editable in `/admin/ingredients`, with per-field override protection so FMP re-imports never clobber console edits — retiring FileMaker for day-to-day editing.

**Architecture:** Generalize the curated-vs-FMP split from whole-column to per-row/per-field via an `overrides` JSON column on the editable tables. The shared `_upsert` becomes override-aware using its existing-but-unused `conflict_update_cols` param (UPDATE only non-overridden columns). `par_level`/`par_level_unit` are promoted from `extras` JSON to first-class columns. New module functions `set_core_field`/`unlock_core_field` (allowlist-gated) + `/api/*` endpoints + editable console fields with an "overridden ⟳ unlock" indicator.

**Tech Stack:** Python 3 / Flask, SQLite (`chat_log.db`), vanilla-JS static console. Tests: pytest.

## Global Constraints

- `overrides` is a JSON array of console-owned field names, per row, on `ingredients`, `ingredient_sources`, `formulation_items`. A field in a row's `overrides` is NOT refreshed by re-import for that row.
- `_upsert` signature stays `_upsert(cx, table, fmp_cols, values, conflict_update_cols=None)`; INSERT-OR-IGNORE inserts all `fmp_cols` (new rows); UPDATE touches only `conflict_update_cols` (default = `fmp_cols`, so existing callers are unchanged). Empty update list → skip UPDATE.
- Schema changes via idempotent `ADD COLUMN`-if-absent (check `PRAGMA table_info`), safe to re-run on deploy.
- `par_level`/`par_level_unit` become real `ingredients` columns; `inventory_starting` STAYS in `extras` (seed-only, never edited).
- Core writes go through `set_core_field` with a per-table allowlist (`_ING_CORE`/`_SRC_CORE`/`_ITEM_CORE`); never let a non-allowlisted field through. `set_core_field` adds the field to `overrides`; `unlock_core_field` removes it (value untouched — unlock re-opens to FMP, does not restore a prior value).
- `common_names` is a JSON array column: UI edits a comma-separated list; module serializes `", ".join` ↔ `json.dumps([...])`.
- Endpoints `@require_console_key` + `ok`/`fail`; `ValueError`→`fail(str(e),status=400)` before generic `except`. Console uses the real `api(path,{method,body:JSON.stringify})` (returns `j.data`, throws); no `JSON.stringify` inside any onclick.
- Route tests use the Pinecone `pytest.skip` pattern.
- Numeric fields (`par_level`, `price_per_unit`, `unit_size`, `dose`) coerce via `float()`; raise `ValueError` on non-numeric.

---

### Task 1: Override-aware `_upsert` + override columns + importer preload

**Files:**
- Modify: `scripts/import_ingredients_from_fmp.py` (`_upsert`, ingredients importer)
- Modify: `scripts/import_formulations_from_fmp.py` (formulation_items importer)
- Modify: `dashboard/ingredient_catalog.py`, `dashboard/formulations.py` (schema: `overrides` columns)
- Test: `tests/test_import_override.py`

**Interfaces:**
- Produces: override-aware `_upsert(cx, table, fmp_cols, values, conflict_update_cols=None)`; `overrides` columns on the 3 tables; importers that skip overridden fields per row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_override.py
import json, sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from scripts.import_ingredients_from_fmp import import_ingredients, import_sources


def _rows_ing(name, par):
    return [{"id_pk": "i1", "name_common": name, "form": "powder", "active": "Yes",
             "par_level": par, "par_level_unit": "g"}]


def test_override_protects_field_per_row(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        import_ingredients(cx, _rows_ing("Mag", "100") + [{"id_pk": "i2", "name_common": "Lipoic", "active": "Yes", "par_level": "5", "par_level_unit": "g"}])
        cx.commit()
        # console edits ingredient i1's par_level → mark overridden
        cx.execute("UPDATE ingredients SET par_level=999, overrides=? WHERE fmp_id='i1'", (json.dumps(["par_level"]),))
        cx.commit()
        # re-import with a DIFFERENT par for both rows
        import_ingredients(cx, [{"id_pk": "i1", "name_common": "Mag", "active": "Yes", "par_level": "100", "par_level_unit": "g"},
                                {"id_pk": "i2", "name_common": "Lipoic", "active": "Yes", "par_level": "50", "par_level_unit": "g"}])
        cx.commit()
        r1 = cx.execute("SELECT par_level, name FROM ingredients WHERE fmp_id='i1'").fetchone()
        r2 = cx.execute("SELECT par_level FROM ingredients WHERE fmp_id='i2'").fetchone()
    assert r1[0] == 999.0          # overridden par survived re-import
    assert r1[1] == "Mag"          # non-overridden field on same row still refreshed
    assert r2[0] == 50.0           # different (non-overridden) row refreshed normally
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_import_override.py -q`
Expected: FAIL (no `overrides`/`par_level` column, or override not honored).

- [ ] **Step 3: Make `_upsert` override-aware** (`scripts/import_ingredients_from_fmp.py`)

```python
def _upsert(cx, table, fmp_cols, values, conflict_update_cols=None):
    # INSERT OR IGNORE all FMP cols (new rows); UPDATE only conflict_update_cols
    # (default = fmp_cols → unchanged behavior). Empty update list → skip UPDATE.
    cols = ["fmp_id"] + fmp_cols
    ph = ",".join("?" for _ in cols)
    fmp_id = values[0]
    cx.execute(f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({ph})", values)
    upd = conflict_update_cols if conflict_update_cols is not None else fmp_cols
    if not upd:
        return
    val_by_col = dict(zip(fmp_cols, values[1:]))
    setc = ", ".join(f"{c}=?" for c in upd) + ", updated_at=datetime('now')"
    cx.execute(f"UPDATE {table} SET {setc} WHERE fmp_id=?",
               (*[val_by_col[c] for c in upd], fmp_id))
```

- [ ] **Step 4: Promote par + preload overrides in the ingredients importer** (`import_ingredients`)

```python
def import_ingredients(cx, rows):
    n = 0
    fmp_cols = ["name", "form", "status", "common_names", "par_level", "par_level_unit", "extras"]
    mapped = set(fmp_cols) | {"id_pk"} | set(_NAME_FIELDS) | {"active", "form",
        "par_level", "par_level_unit",
        "inci_name", "cas_number", "hygroscopic_rating", "solubility", "stability_notes", "spec_notes", "notes"}
    ov = {r["fmp_id"]: set(json.loads(r["overrides"] or "[]"))
          for r in cx.execute("SELECT fmp_id, overrides FROM ingredients WHERE fmp_id IS NOT NULL")}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        names = [_clean(r.get(f)) for f in _NAME_FIELDS if _clean(r.get(f))]
        name = names[0] if names else f"(unnamed FMP ingredient {fid})"
        commons = json.dumps([x for x in names[1:]], ensure_ascii=False) if len(names) > 1 else None
        status = "active" if _active(r.get("active")) == 1 else "inactive"
        vals = [fid, name, _clean(r.get("form")) or None, status, commons,
                _num(r.get("par_level")), _clean(r.get("par_level_unit")) or None, _extras(r, mapped)]
        upd = [c for c in fmp_cols if c not in ov.get(fid, ())]
        _upsert(cx, "ingredients", fmp_cols, vals, upd)
        n += 1
    return n
```

Apply the same preload + `upd` pattern to `import_sources` (table `ingredient_sources`) and, in `scripts/import_formulations_from_fmp.py`, to `import_formulation_items` (table `formulation_items`). Leave materials/POs/production importers as-is (they pass `fmp_cols`, now the default).

- [ ] **Step 5: Add `overrides` + `par_level` columns in schema-init** (`dashboard/ingredient_catalog.py`)

Add a helper and call it inside `init_ingredients_schema` after the CREATE TABLEs:

```python
def _add_col(cx, table, col, decl):
    have = {r[1] for r in cx.execute(f"PRAGMA table_info({table})")}
    if col not in have:
        cx.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

# ...in init_ingredients_schema, after table creation, before commit:
_add_col(cx, "ingredients", "overrides", "TEXT")
_add_col(cx, "ingredients", "par_level", "REAL")
_add_col(cx, "ingredients", "par_level_unit", "TEXT")
_add_col(cx, "ingredient_sources", "overrides", "TEXT")
# one-time backfill of par from the extras JSON
cx.execute("""UPDATE ingredients
              SET par_level = CAST(json_extract(extras,'$.par_level') AS REAL),
                  par_level_unit = json_extract(extras,'$.par_level_unit')
              WHERE par_level IS NULL AND json_extract(extras,'$.par_level') IS NOT NULL""")
```

In `dashboard/formulations.py` `init_formulations_schema`, add `_add_col(cx, "formulation_items", "overrides", "TEXT")` (define/​import the same `_add_col` helper).

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_import_override.py tests/test_import_ingredients.py -q`
Expected: PASS (new override test + existing ingredient importer tests still green).

- [ ] **Step 7: Commit**

```bash
git add scripts/import_ingredients_from_fmp.py scripts/import_formulations_from_fmp.py dashboard/ingredient_catalog.py dashboard/formulations.py tests/test_import_override.py
git commit -m "feat(core-edit): per-field override protection in _upsert + overrides/par columns"
```

---

### Task 2: Repoint par_level readers to the column

**Files:**
- Modify: `dashboard/reorder.py`, `dashboard/inventory.py`
- Test: extend `tests/test_reorder.py`, `tests/test_inventory.py`

**Interfaces:**
- Consumes: the `ingredients.par_level` / `par_level_unit` columns from Task 1.

- [ ] **Step 1: Write/adjust the failing test** — in `tests/test_inventory.py`, change the fixture so par lives in the COLUMN, not extras, and assert `inventory_levels` still reports `below_par`:

```python
# in the _db fixture, replace the extras-based par with column writes:
cx.execute("INSERT INTO ingredients (id,name,par_level,par_level_unit) VALUES (1,'Mag L-threonate',3,'kg')")
cx.execute("INSERT INTO ingredients (id,name,par_level,par_level_unit) VALUES (2,'R-Lipoic',0.25,'kg')")
```

(Keep one extras-based row to prove the backfill path if desired.) Existing `below_par` assertions stay.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_inventory.py -q`
Expected: FAIL (reader still reads `json_extract(extras,'$.par_level')`, which is now NULL).

- [ ] **Step 3: Repoint readers**

`dashboard/inventory.py` — in `inventory_levels`, change the SELECT:
```sql
i.par_level AS par_level, i.par_level_unit AS par_level_unit
```
(was `json_extract(i.extras,'$.par_level')` / `'$.par_level_unit'`). In `get_inventory`, read `ing["par_level"]` / `ing["par_level_unit"]` (was `_json_get`). In `seed_baselines`, read `par_level_unit` from the column: `SELECT id, json_extract(extras,'$.inventory_starting') AS start, par_level_unit AS unit FROM ingredients WHERE json_extract(extras,'$.inventory_starting') IS NOT NULL` (inventory_starting stays in extras).

`dashboard/reorder.py` — `_json_get(r["extras"], "par_level")` / `(ing["extras"], "par_level")` / `(ing["extras"], "par_level_unit")` → `r["par_level"]` / `ing["par_level"]` / `ing["par_level_unit"]` at lines ~92, ~100, ~101. The candidate-set query (`SELECT id, name, extras`) should also select `par_level, par_level_unit`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_inventory.py tests/test_reorder.py tests/test_seed_inventory.py -q`
Expected: PASS (update any fixture rows that set par via extras to use the column; keep one extras row to cover backfill if you added that assertion).

- [ ] **Step 5: Commit**

```bash
git add dashboard/inventory.py dashboard/reorder.py tests/test_inventory.py tests/test_reorder.py
git commit -m "feat(core-edit): read par_level from the promoted column (reorder + inventory)"
```

---

### Task 3: `set_core_field` / `unlock_core_field` + allowlists

**Files:**
- Modify: `dashboard/ingredient_catalog.py` (ingredient + source core edits), `dashboard/formulations.py` (item core edits)
- Test: `tests/test_core_edit.py`

**Interfaces:**
- Produces: `set_ingredient_core(id, field, value, db_path=None)`, `set_source_core(id, field, value, db_path=None)`, `set_item_core(id, field, value, db_path=None)`, and matching `unlock_*` ; allowlists `_ING_CORE`/`_SRC_CORE`/`_ITEM_CORE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_edit.py
import json, sqlite3
import pytest
from dashboard.ingredient_catalog import init_ingredients_schema, set_ingredient_core, unlock_ingredient_core, get_ingredient


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        cx.execute("INSERT INTO ingredients (id,fmp_id,name) VALUES (1,'i1','Mag')")
        cx.commit()
    return db


def test_set_and_unlock_core(tmp_path):
    db = _db(tmp_path)
    set_ingredient_core(1, "par_level", "12", db_path=db)
    set_ingredient_core(1, "common_names", "Magnesium, Mag glycinate", db_path=db)
    ing = get_ingredient(1, db_path=db)
    assert ing["par_level"] == 12.0
    assert json.loads(ing["common_names"]) == ["Magnesium", "Mag glycinate"]
    assert set(json.loads(ing["overrides"])) == {"par_level", "common_names"}
    # non-allowlisted field rejected
    with pytest.raises(ValueError):
        set_ingredient_core(1, "fmp_id", "hacked", db_path=db)
    # non-numeric par rejected
    with pytest.raises(ValueError):
        set_ingredient_core(1, "par_level", "abc", db_path=db)
    # unlock removes from overrides, leaves value
    unlock_ingredient_core(1, "par_level", db_path=db)
    ing = get_ingredient(1, db_path=db)
    assert ing["par_level"] == 12.0
    assert set(json.loads(ing["overrides"])) == {"common_names"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_core_edit.py -q`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Implement core-edit helpers** (`dashboard/ingredient_catalog.py`)

```python
import json

_ING_CORE = {"name", "form", "common_names", "par_level", "par_level_unit"}
_SRC_CORE = {"price_per_unit", "unit_size", "unit_type"}
_NUMERIC_CORE = {"par_level", "price_per_unit", "unit_size"}


def _coerce_core(field, value):
    if field == "common_names":
        parts = [p.strip() for p in str(value or "").split(",") if p.strip()]
        return json.dumps(parts, ensure_ascii=False)
    if field in _NUMERIC_CORE:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be numeric")
    return value if value not in ("",) else None


def _set_core(cx_path, table, allowed, row_id, field, value):
    if field not in allowed:
        raise ValueError(f"{field} is not an editable core field of {table}")
    v = _coerce_core(field, value)
    with _connect(cx_path) as cx:
        row = cx.execute(f"SELECT overrides FROM {table} WHERE id=?", (row_id,)).fetchone()
        if not row:
            raise ValueError(f"no {table} row {row_id}")
        ov = set(json.loads(row["overrides"] or "[]")); ov.add(field)
        cx.execute(f"UPDATE {table} SET {field}=?, overrides=?, updated_at=datetime('now') WHERE id=?",
                   (v, json.dumps(sorted(ov)), row_id))
        cx.commit()


def _unlock_core(cx_path, table, row_id, field):
    with _connect(cx_path) as cx:
        row = cx.execute(f"SELECT overrides FROM {table} WHERE id=?", (row_id,)).fetchone()
        if not row:
            raise ValueError(f"no {table} row {row_id}")
        ov = set(json.loads(row["overrides"] or "[]")); ov.discard(field)
        cx.execute(f"UPDATE {table} SET overrides=?, updated_at=datetime('now') WHERE id=?",
                   (json.dumps(sorted(ov)), row_id))
        cx.commit()


def set_ingredient_core(row_id, field, value, db_path=None):
    _set_core(db_path, "ingredients", _ING_CORE, row_id, field, value)


def unlock_ingredient_core(row_id, field, db_path=None):
    _unlock_core(db_path, "ingredients", row_id, field)


def set_source_core(row_id, field, value, db_path=None):
    _set_core(db_path, "ingredient_sources", _SRC_CORE, row_id, field, value)


def unlock_source_core(row_id, field, db_path=None):
    _unlock_core(db_path, "ingredient_sources", row_id, field)
```

In `dashboard/formulations.py`, add the same pattern for items:
```python
_ITEM_CORE = {"dose", "dose_unit"}
# set_item_core(row_id, field, value, db_path=None) / unlock_item_core(...)
# coerce 'dose' via float(); use the same _set_core/_unlock_core shape (copy or factor a shared helper).
```
The `_set_core`/`_unlock_core`/`_coerce_core` helpers may be factored into a tiny shared module (e.g. `dashboard/_core_edit.py`) imported by both, or duplicated — implementer's call; keep one source of truth for `_coerce_core` if practical.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_core_edit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/ingredient_catalog.py dashboard/formulations.py tests/test_core_edit.py
git commit -m "feat(core-edit): set_/unlock_core_field with allowlists + coercion"
```

---

### Task 4: `/api/*` core-edit + unlock endpoints

**Files:**
- Modify: `app.py`
- Test: `tests/test_admin_core_edit_api.py`

**Interfaces:**
- Consumes: Task 3 functions.
- Produces: `PATCH /api/ingredients/<int:id>/core`, `PATCH /api/sources/<int:id>/core`, `PATCH /api/formulation-items/<int:id>/core`, and `POST .../unlock` for each.

- [ ] **Step 1: Write the failing test** (route-level, Pinecone-skip — mirror `tests/test_admin_inventory_api.py`)

```python
# tests/test_admin_core_edit_api.py
import importlib, json, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        cx.execute("INSERT INTO ingredients (id,fmp_id,name) VALUES (1,'i1','Mag')")
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()


def test_core_edit_and_unlock(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.patch("/api/ingredients/1/core", json={"field": "par_level", "value": "9"})
    assert r.status_code == 200
    d = c.get("/api/ingredients/1").get_json()["data"]
    assert d["par_level"] == 9.0 and "par_level" in json.loads(d["overrides"])
    assert c.patch("/api/ingredients/1/core", json={"field": "fmp_id", "value": "x"}).status_code == 400
    assert c.post("/api/ingredients/1/unlock", json={"field": "par_level"}).status_code == 200
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404) or SKIP (Pinecone). Proceed.

- [ ] **Step 3: Add endpoints in `app.py`** (beside the existing `/api/ingredients/*` block)

```python
from dashboard import ingredient_catalog as _ic
from dashboard import formulations as _ff


def _core_patch(setter, row_id):
    b = request.get_json(silent=True) or {}
    if "field" not in b:
        return fail("field required", status=400)
    setter(row_id, b["field"], b.get("value"))
    return ok({"id": row_id})


@app.route("/api/ingredients/<int:rid>/core", methods=["PATCH"])
@require_console_key
def api_ingredient_core(rid):
    try:
        return _core_patch(_ic.set_ingredient_core, rid)
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)


@app.route("/api/ingredients/<int:rid>/unlock", methods=["POST"])
@require_console_key
def api_ingredient_unlock(rid):
    try:
        _ic.unlock_ingredient_core(rid, (request.get_json(silent=True) or {}).get("field"))
        return ok({"id": rid})
    except Exception as e:
        return fail(e)
```

Repeat the pair for sources (`_ic.set_source_core`/`unlock_source_core`, route `/api/sources/<int:rid>/core` + `/unlock`) and formulation-items (`_ff.set_item_core`/`unlock_item_core`, route `/api/formulation-items/<int:rid>/core` + `/unlock`). Each `core` route catches `ValueError`→400 before the generic handler.

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_admin_core_edit_api.py -q` → PASS or SKIP on Pinecone. Smoke `python3 -c "import app"`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_admin_core_edit_api.py
git commit -m "feat(core-edit): /api core-field PATCH + unlock endpoints"
```

---

### Task 5: Console editable fields + overridden/unlock UI

**Files:**
- Modify: `static/admin-ingredients.html`

**Interfaces:**
- Consumes: the `/api/*/core` + `/unlock` endpoints; the reads now return `overrides`, `par_level`, `par_level_unit`.

- [ ] **Step 1: Read the existing Ingredients detail + Sources rows + Formulations recipe-items rendering** in `static/admin-ingredients.html`, and the existing curated-field editors (notes/preferred) as the pattern. Note the real `api(path, opts)` (returns `j.data`, throws) and `escapeHtml`.

- [ ] **Step 2: Render in-scope core fields as editable inputs** (replacing their read-only display):
  - Ingredient detail: `name`, `form`, `common_names` (comma-joined from the JSON array), `par_level` + `par_level_unit`.
  - Each source row: `price_per_unit`, `unit_size`, `unit_type`.
  - Each recipe item row (Formulations): `dose`, `dose_unit`.
  Each input saves on change/blur via the matching core PATCH, e.g. `await api("/api/ingredients/"+id+"/core", {method:"PATCH", body: JSON.stringify({field:"par_level", value: el.value})})` wrapped in try/catch (toast on error), then refresh the detail.

- [ ] **Step 3: Overridden indicator + unlock control.** For each editable field, if its name is in the row's `overrides` array, show an amber **"overridden"** badge + a small **⟳ unlock** button next to it. Unlock calls `await api("/api/<entity>/"+id+"/unlock", {method:"POST", body: JSON.stringify({field})})` then refreshes. Use a dataset/index pattern for the handler — **no `JSON.stringify` inside any onclick attribute**.

- [ ] **Step 4: Verify** HTML parses (balanced); the new inputs/ids exist and are wired; existing tabs/curated editors untouched; `escapeHtml` on all interpolated server strings; common_names round-trips (array → ", ".join in the input → server splits back).

- [ ] **Step 5: Commit**

```bash
git add static/admin-ingredients.html
git commit -m "feat(core-edit): editable core fields + overridden/unlock UI"
```

---

## Self-Review

- **Spec coverage:** override mechanism (`overrides` col + override-aware `_upsert` + importer preload) T1; par promotion + reader repoint T2; `set_/unlock_core_field` + allowlists + coercion T3; endpoints T4; console UI T5. `common_names` editable (comma↔JSON) ✓. Unlock = remove-from-overrides, value untouched ✓. Materials/POs/production importers unchanged ✓. Products.json fields out of scope ✓.
- **Placeholders:** complete code for the non-obvious pieces (the `_upsert` change, importer preload, schema/backfill, `_set_core`/`_coerce_core`, endpoints, the override test). T5 (UI) gives concrete ids/calls + adapts to existing patterns, as prior console tasks did.
- **Type consistency:** `_upsert(cx,table,fmp_cols,values,conflict_update_cols=None)` — all existing callers pass `fmp_cols` (unchanged); ingredients/formulations importers pass reduced `upd`. `set_ingredient_core`/`set_source_core`/`set_item_core` + `unlock_*` names consistent across module/endpoints/tests. `overrides` is always a JSON array of field-name strings; reads/writes via `json.loads`/`json.dumps`. Endpoints take `{field, value}` / `{field}`.
- **Highest-risk task = T1** (the protect-on-reimport invariant + par promotion); it gets the override test and careful review. **T2** carries the par blast radius (every reader); covered by existing inventory/reorder suites + adjusted fixtures.
- **Reviewer note:** confirm the `_add_col` idempotent-migration runs in the deployed schema-init path (so prod gets `overrides`/`par_level` + the backfill on deploy) before the importers rely on the columns.

## Build approach
Subagent-driven-development on a fresh branch off `main`, one PR, whole-branch review at the end. Order: T1 (foundation, adversarial review) → T2 (reader repoint) → T3 (core-edit helpers) → T4 (endpoints) → T5 (console UI). After merge + deploy: the schema-init adds the columns + backfills par on the prod DB; edit core fields in the console; any future FMP re-import refreshes only non-overridden fields.
