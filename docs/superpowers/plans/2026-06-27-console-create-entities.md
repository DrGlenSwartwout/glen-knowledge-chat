# Console Create Entities (E2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create brand-new ingredients, suppliers, sources, and recipe items in `/admin/ingredients`, with `fmp_id = NULL` so they're permanently console-owned and invisible to FMP re-imports — enabling new-vendor onboarding (HydroCurc/Pharmako) and the email-sourcing collector's "approve unmatched quote".

**Architecture:** Pure inserts of `fmp_id = NULL` rows into existing tables (no schema change). A shared `_insert_allowed` helper (mirror of E1's `_update_allowed`) gates inserted columns by an allowlist (the columns are interpolated into the INSERT, so the allowlist is the injection guard) and coerces values via E1's `_coerce_core`. New module functions + `/api/*` create/delete endpoints + console create forms.

**Tech Stack:** Python 3 / Flask, SQLite (`chat_log.db`), vanilla-JS console. Tests: pytest.

## Global Constraints

- Console-created rows have **`fmp_id = NULL`** → never matched by any importer (idempotency index is `WHERE fmp_id IS NOT NULL`; importers iterate FMP rows by fmp_id). No override flags needed. Use plain `INSERT` (not INSERT-OR-IGNORE; NULL fmp_id isn't unique-constrained).
- **Allowlist gates the inserted columns** (they're interpolated into `INSERT (cols...)`). A field not in the allowlist → `ValueError` (reject; never reaches SQL). Required fields enforced (`name` for ingredient, `company` for supplier). Reuse E1 `_coerce_core(field, value, numeric_extra=...)` for value coercion (numerics → float-or-None raising on bad input; `common_names` comma→JSON).
- Endpoints: `@require_console_key`, `ok`/`fail`; `ValueError`→`fail(str(e), status=400)` BEFORE the generic `except`.
- Console: real `api(path,{method,body:JSON.stringify})` (returns `j.data`, throws); search-to-pick index-array pattern; **NO `JSON.stringify` inside any onclick**; `escapeHtml`; `toast`.
- Route tests use the Pinecone `pytest.skip` pattern.
- No schema migration.

---

### Task 1: create_ingredient / create_supplier / create_source + `_insert_allowed`

**Files:**
- Modify: `dashboard/ingredient_catalog.py`
- Test: `tests/test_create_entities.py`

**Interfaces:**
- Produces: `create_ingredient(fields, db_path=None) -> int`; `create_supplier(fields, db_path=None) -> int`; `create_source(ingredient_id, fields, db_path=None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_create_entities.py
import sqlite3
import pytest
from dashboard.ingredient_catalog import (
    init_ingredients_schema, create_ingredient, create_supplier, create_source, get_ingredient,
    list_sources_for_ingredient,
)
from scripts.import_ingredients_from_fmp import import_ingredients


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        cx.commit()
    return db


def test_create_ingredient(tmp_path):
    db = _db(tmp_path)
    iid = create_ingredient({"name": "HydroCurc", "form": "powder",
                             "common_names": "Curcumin, LipiSperse curcumin",
                             "par_level": "5", "par_level_unit": "kg"}, db_path=db)
    ing = get_ingredient(iid, db_path=db)
    assert ing["name"] == "HydroCurc" and ing["fmp_id"] is None and ing["par_level"] == 5.0
    import json
    assert json.loads(ing["common_names"]) == ["Curcumin", "LipiSperse curcumin"]
    with pytest.raises(ValueError):
        create_ingredient({"form": "powder"}, db_path=db)            # name required
    with pytest.raises(ValueError):
        create_ingredient({"name": "X", "fmp_id": "hack"}, db_path=db)  # non-creatable field


def test_create_supplier_and_source(tmp_path):
    db = _db(tmp_path)
    iid = create_ingredient({"name": "HydroCurc"}, db_path=db)
    sup = create_supplier({"company": "Pharmako Biotechnologies", "email": "enquiries@pharmako.com.au"}, db_path=db)
    sid = create_source(iid, {"supplier_id": sup, "price_per_unit": "334", "unit_size": "1",
                              "unit_type": "kg", "minimum_order": "25", "lead_time_days": "9",
                              "preferred": 1}, db_path=db)
    srcs = list_sources_for_ingredient(iid, db_path=db)
    assert len(srcs) == 1 and srcs[0]["id"] == sid
    assert srcs[0]["price_per_unit"] == 334.0 and srcs[0]["minimum_order"] == 25.0 and srcs[0]["preferred"] == 1
    with pytest.raises(ValueError):
        create_supplier({"email": "x@y.z"}, db_path=db)              # company required
    with pytest.raises(ValueError):
        create_source(99999, {"price_per_unit": "1"}, db_path=db)   # ingredient must exist


def test_created_ingredient_survives_reimport(tmp_path):
    """The core E2 invariant: an fmp_id=NULL console-created row is untouched + not duplicated by re-import."""
    db = _db(tmp_path)
    iid = create_ingredient({"name": "HydroCurc", "par_level": "5", "par_level_unit": "kg"}, db_path=db)
    with sqlite3.connect(db) as cx:
        # re-import a real FMP ingredient (different fmp_id) — must not touch the created row
        import_ingredients(cx, [{"id_pk": "f1", "name_common": "R-Lipoic Acid", "active": "Yes"}])
        cx.commit()
        rows = cx.execute("SELECT id, name, fmp_id, par_level FROM ingredients ORDER BY id").fetchall()
    assert len(rows) == 2                          # created + imported, no duplicate
    created = [r for r in rows if r[0] == iid][0]
    assert created[1] == "HydroCurc" and created[2] is None and created[3] == 5.0   # untouched
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_create_entities.py -q`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Implement in `dashboard/ingredient_catalog.py`**

Add near the curated helpers (reuse the existing `_connect`, `_coerce_core` import, `set_preferred_source`):

```python
from dashboard._core_edit import _coerce_core  # already imported in this module for E1; reuse

_ING_CREATABLE = _ING_CORE | _ING_CURATED               # name/form/common_names/par_*/curated
_SUP_CREATABLE = {"company", "address_street", "address_city", "address_province",
                  "address_postal_code", "email", "phone_business", "phone_cell",
                  "phone_fax", "url", "notes"}
_SRC_CREATABLE = {"ingredient_id", "supplier_id", "supplier_name", "sku", "price_per_unit",
                  "unit_size", "unit_type", "shipping_quote", "preferred", "lead_time_days",
                  "minimum_order", "minimum_order_unit", "notes"}
_SRC_NUMERIC_EXTRA = {"minimum_order", "lead_time_days", "shipping_quote", "supplier_id", "ingredient_id"}


def _insert_allowed(table, fields, allowed, required, numeric_extra=None, db_path=None):
    fields = fields or {}
    for k in fields:
        if k not in allowed:
            raise ValueError(f"{k} is not a creatable field of {table}")
    for req in required:
        if str(fields.get(req) if fields.get(req) is not None else "").strip() == "":
            raise ValueError(f"{req} is required")
    cols, vals = [], []
    for k, v in fields.items():
        cols.append(k)
        vals.append(_coerce_core(k, v, numeric_extra=numeric_extra))
    if not cols:
        raise ValueError("no fields to insert")
    with _connect(db_path) as cx:
        cur = cx.execute(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", vals)
        cx.commit()
        return int(cur.lastrowid)


def create_ingredient(fields, db_path=None) -> int:
    return _insert_allowed("ingredients", fields, _ING_CREATABLE, {"name"}, db_path=db_path)


def create_supplier(fields, db_path=None) -> int:
    return _insert_allowed("suppliers", fields, _SUP_CREATABLE, {"company"}, db_path=db_path)


def create_source(ingredient_id, fields, db_path=None) -> int:
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM ingredients WHERE id=?", (ingredient_id,)).fetchone():
            raise ValueError(f"no ingredient {ingredient_id}")
    f = {**(fields or {}), "ingredient_id": ingredient_id}
    want_pref = str(f.get("preferred") or "") in ("1", "true", "True", "yes")
    sid = _insert_allowed("ingredient_sources", f, _SRC_CREATABLE, set(),
                          numeric_extra=_SRC_NUMERIC_EXTRA, db_path=db_path)
    if want_pref:
        set_preferred_source(sid, db_path=db_path)   # unset others for this ingredient
    return sid
```

Notes: `_coerce_core` turns `common_names` comma→JSON, numerics→float (raising on bad). `preferred` is coerced as a passthrough then re-asserted via `set_preferred_source`. Confirm `_coerce_core` / `_ING_CORE` / `_ING_CURATED` / `set_preferred_source` are all in scope in this module (they are from E1) — if `_coerce_core` isn't already imported at module top, add the import.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_create_entities.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/ingredient_catalog.py tests/test_create_entities.py
git commit -m "feat(create): create_ingredient/supplier/source (fmp_id NULL, importer-invisible)"
```

---

### Task 2: add / remove formulation items

**Files:**
- Modify: `dashboard/formulations.py`
- Test: `tests/test_formulation_item_crud.py`

**Interfaces:**
- Produces: `add_formulation_item(formulation_id, ingredient_id, dose, dose_unit, db_path=None) -> int`; `remove_formulation_item(item_id, db_path=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_formulation_item_crud.py
import sqlite3
import pytest
from dashboard.ingredient_catalog import init_ingredients_schema
from dashboard.formulations import (
    init_formulations_schema, add_formulation_item, remove_formulation_item, list_items_for_formulation,
)


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_formulations_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name) VALUES (1,'HydroCurc')")
        cx.execute("INSERT INTO formulations (id,name) VALUES (1,'Brain Blend')")
        cx.commit()
    return db


def test_add_and_remove_item(tmp_path):
    db = _db(tmp_path)
    item = add_formulation_item(1, 1, "500", "mg", db_path=db)
    items = list_items_for_formulation(1, db_path=db)
    assert len(items) == 1 and items[0]["id"] == item and items[0]["dose"] == 500.0 and items[0]["dose_unit"] == "mg"
    with pytest.raises(ValueError):
        add_formulation_item(999, 1, "1", "mg", db_path=db)     # formulation must exist
    with pytest.raises(ValueError):
        add_formulation_item(1, 999, "1", "mg", db_path=db)     # ingredient must exist
    with pytest.raises(ValueError):
        add_formulation_item(1, 1, "abc", "mg", db_path=db)     # dose numeric
    remove_formulation_item(item, db_path=db)
    assert list_items_for_formulation(1, db_path=db) == []
```

- [ ] **Step 2: Run to verify it fails** — FAIL (functions undefined).

- [ ] **Step 3: Implement in `dashboard/formulations.py`**

```python
def add_formulation_item(formulation_id, ingredient_id, dose, dose_unit, db_path=None) -> int:
    try:
        d = None if dose in (None, "") else float(dose)
    except (TypeError, ValueError):
        raise ValueError("dose must be numeric")
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM formulations WHERE id=?", (formulation_id,)).fetchone():
            raise ValueError(f"no formulation {formulation_id}")
        if not cx.execute("SELECT 1 FROM ingredients WHERE id=?", (ingredient_id,)).fetchone():
            raise ValueError(f"no ingredient {ingredient_id}")
        row = cx.execute("SELECT name FROM ingredients WHERE id=?", (ingredient_id,)).fetchone()
        cur = cx.execute("""
            INSERT INTO formulation_items (formulation_id, ingredient_id, ingredient_name, dose, dose_unit)
            VALUES (?, ?, ?, ?, ?)
        """, (formulation_id, ingredient_id, row["name"] if row else None, d, dose_unit or None))
        cx.commit()
        return int(cur.lastrowid)


def remove_formulation_item(item_id, db_path=None) -> None:
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM formulation_items WHERE id=?", (item_id,)).fetchone():
            raise ValueError(f"no formulation item {item_id}")
        cx.execute("DELETE FROM formulation_items WHERE id=?", (item_id,))
        cx.commit()
```

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_formulation_item_crud.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/formulations.py tests/test_formulation_item_crud.py
git commit -m "feat(create): add/remove formulation item (recipe editing)"
```

---

### Task 3: `/api/*` create + delete endpoints

**Files:**
- Modify: `app.py`
- Test: `tests/test_admin_create_api.py`

**Interfaces:**
- Produces: `POST /api/ingredients`; `POST /api/suppliers`; `POST /api/ingredients/<int:iid>/sources`; `POST /api/formulations/<int:fid>/items`; `DELETE /api/formulation-items/<int:item_id>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_admin_create_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.formulations import init_formulations_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_formulations_schema(cx)
        cx.execute("INSERT INTO formulations (id,name) VALUES (1,'Brain Blend')")
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


def test_create_flow(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    iid = c.post("/api/ingredients", json={"name": "HydroCurc", "par_level": "5"}).get_json()["data"]["id"]
    sup = c.post("/api/suppliers", json={"company": "Pharmako"}).get_json()["data"]["id"]
    src = c.post(f"/api/ingredients/{iid}/sources", json={"supplier_id": sup, "price_per_unit": "334", "unit_type": "kg"})
    assert src.status_code == 200
    item = c.post(f"/api/formulations/1/items", json={"ingredient_id": iid, "dose": "500", "dose_unit": "mg"}).get_json()["data"]["id"]
    assert c.delete(f"/api/formulation-items/{item}").status_code == 200
    assert c.post("/api/ingredients", json={"form": "powder"}).status_code == 400   # name required
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404) or SKIP (Pinecone). Proceed.

- [ ] **Step 3: Add endpoints in `app.py`** (beside the `/api/ingredients/*` block; reuse the existing `_ic`/`_ff` aliases — verify names with a grep, they were added in E1 core endpoints)

```python
@app.route("/api/ingredients", methods=["POST"])
@require_console_key
def api_create_ingredient():
    try:
        return ok({"id": _ic.create_ingredient(request.get_json(silent=True) or {})})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)


@app.route("/api/suppliers", methods=["POST"])
@require_console_key
def api_create_supplier():
    try:
        return ok({"id": _ic.create_supplier(request.get_json(silent=True) or {})})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)


@app.route("/api/ingredients/<int:iid>/sources", methods=["POST"])
@require_console_key
def api_create_source(iid):
    try:
        return ok({"id": _ic.create_source(iid, request.get_json(silent=True) or {})})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)


@app.route("/api/formulations/<int:fid>/items", methods=["POST"])
@require_console_key
def api_add_formulation_item(fid):
    try:
        b = request.get_json(silent=True) or {}
        return ok({"id": _ff.add_formulation_item(fid, b.get("ingredient_id"), b.get("dose"), b.get("dose_unit"))})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)


@app.route("/api/formulation-items/<int:item_id>", methods=["DELETE"])
@require_console_key
def api_remove_formulation_item(item_id):
    try:
        _ff.remove_formulation_item(item_id)
        return ok({"id": item_id})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)
```

(If the `_ic`/`_ff` aliases differ in app.py — E1 used `_ic = ingredient_catalog`, `_ff = formulations` — match the actual names; grep `import ingredient_catalog as` / `import formulations as` first.)

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_admin_create_api.py -q` → PASS or SKIP on Pinecone. Smoke `python3 -c "import app"`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_admin_create_api.py
git commit -m "feat(create): /api create ingredient/supplier/source/item + delete item endpoints"
```

---

### Task 4: Console create forms

**Files:**
- Modify: `static/admin-ingredients.html`

- [ ] **Step 1: Read** the Ingredients tab, Suppliers tab, ingredient detail (sources rendering), and Formulation detail (items rendering) + the E1 editable-field code as the pattern. Note the real `api()`, `escapeHtml`, `toast`, the search-to-pick index-array pattern.

- [ ] **Step 2: Add create UI:**
  - Ingredients tab: a **"+ New ingredient"** button → a small form (name [required] + form + common_names + par_level + par_level_unit) → `POST /api/ingredients` → on success open the new ingredient's detail + refresh the list.
  - Ingredient detail: an **"+ Add source"** sub-form — a supplier search-to-pick (query `/api/suppliers` if a search exists, else a "new supplier" inline: company field → `POST /api/suppliers` then use its id) + price_per_unit/unit_size/unit_type/minimum_order/lead_time_days/preferred → `POST /api/ingredients/<iid>/sources` → refresh sources.
  - Suppliers tab: a **"+ New supplier"** form (company [required] + email/phone/url) → `POST /api/suppliers`.
  - Formulation detail: an **"+ Add ingredient"** row — ingredient search-to-pick + dose + dose_unit → `POST /api/formulations/<fid>/items` → refresh items; an **✕** on each item → `confirm()` then `DELETE /api/formulation-items/<id>` → refresh.
  - All via `api(url,{method:"POST"|"DELETE", body: JSON.stringify(...)})`, try/catch + toast. **No `JSON.stringify` in any onclick** (search-to-pick uses the index-array pattern). `escapeHtml` on interpolated strings.

- [ ] **Step 3: Verify** HTML parses (balanced); new ids/handlers consistent; grep-confirm no `JSON.stringify` in any `onclick=`; existing tabs/editors untouched.

- [ ] **Step 4: Commit**

```bash
git add static/admin-ingredients.html
git commit -m "feat(create): console create forms (ingredient/supplier/source/recipe item)"
```

---

## Self-Review
- **Spec coverage:** create_ingredient/supplier/source + add/remove_formulation_item (T1,T2); endpoints (T3); console forms (T4). `fmp_id` NULL on all creates ✓. Allowlist gates inserted columns (injection-safe) ✓. The reimport-survival invariant is tested in T1 ✓. Parent-delete + new-formulation deferred ✓.
- **Placeholders:** complete code for the non-UI tasks; T4 adapts to existing patterns.
- **Type consistency:** `create_ingredient`/`create_supplier`/`create_source`/`add_formulation_item`/`remove_formulation_item` used identically across module/endpoints/tests; `_insert_allowed` mirrors `_update_allowed`; reuses E1 `_coerce_core`/`set_preferred_source`. `preferred` re-asserted via `set_preferred_source`.
- **Reviewer note (T1, security):** the inserted COLUMN names are interpolated into the `INSERT` — confirm `_insert_allowed` rejects any field not in the allowlist BEFORE building the SQL (same posture as E1's f-string guard). `table` is hardcoded by the create wrappers.

## Build approach
Subagent-driven-development, this branch (off the merged main), one PR, whole-branch review. Order T1 (foundation + the reimport-survival invariant — careful review of the insert-column injection guard) → T2 → T3 → T4. After merge: create HydroCurc + Pharmako + the $334/kg source end-to-end as the live validation; then build the email-sourcing collector (which calls these create functions).
