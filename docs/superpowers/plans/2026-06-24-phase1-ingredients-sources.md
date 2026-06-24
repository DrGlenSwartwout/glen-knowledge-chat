# Phase 1: Ingredients + Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate the FileMaker raw-material master (suppliers, ingredients, ingredient_sources) into new SQLite tables in `chat_log.db` as the authoritative source, with canonical variant clustering, an idempotent FMP importer that preserves console edits, and an `/admin/ingredients` console.

**Architecture:** A new `dashboard/ingredient_catalog.py` module mirrors `dashboard/shipping.py` exactly (idempotent `init_ingredients_schema`, `_connect`, CRUD with optional `db_path`). A standalone `scripts/import_ingredients_from_fmp.py` loads three FMP CSV exports + the canonical-clusters CSV. Console = a static `/admin/ingredients` page + `/api/ingredients/*` JSON endpoints. Schema-init wired at app.py module load.

**Tech Stack:** Python 3, Flask, sqlite3 (`chat_log.db`), pytest, vanilla-JS static HTML. Stdlib only.

## Global Constraints
- **Module name is `dashboard/ingredient_catalog.py`** (NOT `ingredients.py` — that name is the existing ingredient-PAGES resolver: slugify/resolve/formulations_with, used across app.py + topic_copy/ingredient_copy. Do not touch it). New test file: `tests/test_ingredient_catalog.py`.
- No new dependencies. The app DB (`chat_log.db`) is the authoritative source; FMP CSVs are import-only.
- **Curated-vs-FMP column split (the key invariant):** the importer writes ONLY FMP-sourced columns; on re-import it `ON CONFLICT(fmp_id) DO UPDATE` refreshing only those columns and NEVER touches console-edited curated columns. Curated cols: suppliers(`notes`); ingredients(`inci_name,cas_number,hygroscopic_rating,solubility,stability_notes,spec_notes,notes`); ingredient_sources(`preferred,lead_time_days,minimum_order,minimum_order_unit,notes`). `canonical_id` is set only by the canonical pass.
- `fmp_id` (= FMP `id_pk`) is the idempotent re-import key; partial unique index `WHERE fmp_id IS NOT NULL`.
- Money/numbers as `REAL`; booleans as `INTEGER` (FMP "Yes"→1, "No"→0, blank→NULL); timestamps `TEXT DEFAULT (datetime('now'))`; arrays/sparse → JSON `TEXT`.
- DB access via `_connect(db_path)`/`_default_db_path()` (copy from `dashboard/shipping.py:47-57`); `PRAGMA foreign_keys=ON`. Console endpoints `@require_console_key`, `ok`/`fail` from `dashboard/__init__.py`.
- **CSV field-size limit MUST be raised** (`csv.field_size_limit(sys.maxsize)`) — some FMP text fields exceed 128 KB.
- FMP source CSVs (already exported, verified): `/tmp/fmp-export/newapp/{suppliers,ingredients,ingredients_supplier}.csv` (1058 / 2363 / 3980 records). Canonical: `02 Skills/fmp-loaders/mapping/canonical_clusters.csv` (528 rows: `head_fmp_id,member_fmp_id,reason`). Path overridable by env in the importer.

---

### Task 1: `dashboard/ingredient_catalog.py` — schema + read functions + app wiring

**Files:**
- Create: `dashboard/ingredient_catalog.py`
- Modify: `app.py` (add `_init_ingredients_tables()` at module load, next to `_init_shipping_tables` ~line 922-928)
- Test: `tests/test_ingredient_catalog.py`

**Interfaces — Produces:**
- `init_ingredients_schema(cx)` — idempotent CREATE of `suppliers`, `ingredients`, `ingredient_sources` + indexes.
- `search_ingredients(q="", limit=50, offset=0, db_path=None) -> list[dict]` (name LIKE, ordered by name).
- `get_ingredient(ingredient_id, db_path=None) -> dict | None`
- `list_sources_for_ingredient(ingredient_id, db_path=None) -> list[dict]` (joined supplier company).
- `list_suppliers(q="", limit=50, offset=0, db_path=None) -> list[dict]`; `get_supplier(supplier_id, db_path=None) -> dict | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingredient_catalog.py
import sqlite3, pytest

@pytest.fixture
def db(tmp_path):
    from dashboard.ingredient_catalog import init_ingredients_schema
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx)
        init_ingredients_schema(cx)  # idempotent
    return p

def test_schema_creates_tables(db):
    with sqlite3.connect(db) as cx:
        tables = {r[0] for r in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"suppliers", "ingredients", "ingredient_sources"} <= tables

def test_search_and_get(db):
    from dashboard.ingredient_catalog import search_ingredients, get_ingredient, list_sources_for_ingredient
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO suppliers (fmp_id, company) VALUES ('s1','Acme')")
        sid = cx.execute("SELECT id FROM suppliers").fetchone()[0]
        cx.execute("INSERT INTO ingredients (fmp_id, name) VALUES ('i1','R-Lipoic Acid')")
        iid = cx.execute("SELECT id FROM ingredients").fetchone()[0]
        cx.execute("INSERT INTO ingredient_sources (fmp_id, ingredient_id, supplier_id, price_per_unit) VALUES ('x1',?,?,12.5)", (iid, sid))
        cx.commit()
    assert search_ingredients("lipoic", db_path=db)[0]["name"] == "R-Lipoic Acid"
    assert get_ingredient(iid, db_path=db)["fmp_id"] == "i1"
    srcs = list_sources_for_ingredient(iid, db_path=db)
    assert srcs[0]["price_per_unit"] == 12.5 and srcs[0]["company"] == "Acme"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-59a2725d && python3 -m pytest tests/test_ingredient_catalog.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.ingredient_catalog'`

- [ ] **Step 3: Implement `dashboard/ingredient_catalog.py`**

```python
"""Ingredients + sources catalog — FMP-migrated raw-material master in chat_log.db.
Mirrors dashboard/shipping.py conventions (idempotent schema, _connect, db_path kwarg)."""
from __future__ import annotations
import os, sqlite3
from pathlib import Path
from typing import Optional


def _default_db_path() -> str:
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    cx = sqlite3.connect(db_path or _default_db_path())
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA foreign_keys = ON")
    return cx


def init_ingredients_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS suppliers (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT, company TEXT NOT NULL,
          address_street TEXT, address_city TEXT, address_province TEXT, address_postal_code TEXT,
          email TEXT, phone_business TEXT, phone_cell TEXT, phone_fax TEXT, url TEXT,
          qb_id TEXT, active INTEGER,
          notes TEXT, extras TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_suppliers_fmp ON suppliers(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS ingredients (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT, name TEXT NOT NULL, form TEXT, status TEXT,
          common_names TEXT, canonical_id INTEGER REFERENCES ingredients(id),
          extras TEXT,
          inci_name TEXT, cas_number TEXT, hygroscopic_rating TEXT, solubility TEXT,
          stability_notes TEXT, spec_notes TEXT, notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ingredients_fmp ON ingredients(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_ingredients_canon ON ingredients(canonical_id)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS ingredient_sources (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT,
          ingredient_id INTEGER REFERENCES ingredients(id),
          supplier_id INTEGER REFERENCES suppliers(id),
          supplier_name TEXT, sku TEXT,
          price_per_unit REAL, unit_size REAL, unit_type TEXT, shipping_quote REAL,
          extras TEXT,
          preferred INTEGER DEFAULT 0, lead_time_days INTEGER,
          minimum_order REAL, minimum_order_unit TEXT, notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ingsrc_fmp ON ingredient_sources(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_ingsrc_ing ON ingredient_sources(ingredient_id)")
    cx.commit()


def search_ingredients(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT * FROM ingredients WHERE name LIKE ? ORDER BY name LIMIT ? OFFSET ?",
            (f"%{q}%", int(limit), int(offset)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_ingredient(ingredient_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("SELECT * FROM ingredients WHERE id=?", (ingredient_id,)).fetchone()
    return dict(r) if r else None


def list_sources_for_ingredient(ingredient_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT s.*, sup.company AS company
            FROM ingredient_sources s LEFT JOIN suppliers sup ON sup.id = s.supplier_id
            WHERE s.ingredient_id = ?
            ORDER BY s.preferred DESC, s.price_per_unit
        """, (ingredient_id,)).fetchall()
    return [dict(r) for r in rows]


def list_suppliers(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT * FROM suppliers WHERE company LIKE ? ORDER BY company LIMIT ? OFFSET ?",
            (f"%{q}%", int(limit), int(offset)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_supplier(supplier_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("SELECT * FROM suppliers WHERE id=?", (supplier_id,)).fetchone()
    return dict(r) if r else None
```

- [ ] **Step 4: Wire schema init at app startup** — in `app.py`, directly after the `_init_shipping_tables()` block (~line 928), add:

```python
def _init_ingredients_tables():
    """Ingredients + sources catalog (FMP-migrated raw-material master)."""
    from dashboard.ingredient_catalog import init_ingredients_schema
    with sqlite3.connect(LOG_DB) as cx:
        init_ingredients_schema(cx)

_init_ingredients_tables()
```

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_ingredient_catalog.py -q` → PASS
```bash
git add dashboard/ingredient_catalog.py app.py tests/test_ingredient_catalog.py
git commit -m "feat(ingredients): schema + read functions + app wiring"
```

---

### Task 2: Curated write functions

**Files:**
- Modify: `dashboard/ingredient_catalog.py`
- Test: `tests/test_ingredient_catalog.py`

**Interfaces — Produces:**
- `update_ingredient_curated(ingredient_id, fields: dict, db_path=None)` — only keys in `{inci_name,cas_number,hygroscopic_rating,solubility,stability_notes,spec_notes,notes}` applied.
- `update_source_curated(source_id, fields: dict, db_path=None)` — only `{lead_time_days,minimum_order,minimum_order_unit,notes}`.
- `set_preferred_source(source_id, db_path=None)` — preferred=1 on this source, preferred=0 on all other sources of the same ingredient.
- `update_supplier(supplier_id, fields: dict, db_path=None)` — only `{notes}`.

- [ ] **Step 1: Write the failing test**

```python
def test_curated_updates_and_preferred(db):
    from dashboard.ingredient_catalog import (update_ingredient_curated, update_source_curated,
        set_preferred_source, get_ingredient, list_sources_for_ingredient)
    import sqlite3
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO ingredients (fmp_id, name) VALUES ('i1','X')")
        iid = cx.execute("SELECT id FROM ingredients").fetchone()[0]
        cx.execute("INSERT INTO ingredient_sources (fmp_id, ingredient_id) VALUES ('a',?)", (iid,))
        cx.execute("INSERT INTO ingredient_sources (fmp_id, ingredient_id) VALUES ('b',?)", (iid,))
        ids = [r[0] for r in cx.execute("SELECT id FROM ingredient_sources ORDER BY id")]
        cx.commit()
    update_ingredient_curated(iid, {"inci_name": "Foo", "name": "HACK"}, db_path=db)  # name ignored
    g = get_ingredient(iid, db_path=db)
    assert g["inci_name"] == "Foo" and g["name"] == "X"
    update_source_curated(ids[0], {"lead_time_days": 14, "minimum_order": 100}, db_path=db)
    set_preferred_source(ids[1], db_path=db)
    srcs = {s["id"]: s for s in list_sources_for_ingredient(iid, db_path=db)}
    assert srcs[ids[1]]["preferred"] == 1 and srcs[ids[0]]["preferred"] == 0
    assert srcs[ids[0]]["lead_time_days"] == 14
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_ingredient_catalog.py -k curated -q`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement** (append to `dashboard/ingredient_catalog.py`)

```python
_ING_CURATED = {"inci_name","cas_number","hygroscopic_rating","solubility","stability_notes","spec_notes","notes"}
_SRC_CURATED = {"lead_time_days","minimum_order","minimum_order_unit","notes"}
_SUP_CURATED = {"notes"}


def _update_allowed(table, row_id, fields, allowed, db_path):
    cols = {k: v for k, v in (fields or {}).items() if k in allowed}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*cols.values(), row_id))
        cx.commit()


def update_ingredient_curated(ingredient_id, fields, db_path=None):
    _update_allowed("ingredients", ingredient_id, fields, _ING_CURATED, db_path)


def update_source_curated(source_id, fields, db_path=None):
    _update_allowed("ingredient_sources", source_id, fields, _SRC_CURATED, db_path)


def update_supplier(supplier_id, fields, db_path=None):
    _update_allowed("suppliers", supplier_id, fields, _SUP_CURATED, db_path)


def set_preferred_source(source_id, db_path=None):
    with _connect(db_path) as cx:
        row = cx.execute("SELECT ingredient_id FROM ingredient_sources WHERE id=?", (source_id,)).fetchone()
        if not row:
            return
        cx.execute("UPDATE ingredient_sources SET preferred=0, updated_at=datetime('now') WHERE ingredient_id=?", (row["ingredient_id"],))
        cx.execute("UPDATE ingredient_sources SET preferred=1, updated_at=datetime('now') WHERE id=?", (source_id,))
        cx.commit()
```

- [ ] **Step 4: Run tests + commit**

Run: `python3 -m pytest tests/test_ingredient_catalog.py -q` → PASS
```bash
git add dashboard/ingredient_catalog.py tests/test_ingredient_catalog.py
git commit -m "feat(ingredients): curated write functions + preferred-source toggle"
```

---

### Task 3: FMP importer (`scripts/import_ingredients_from_fmp.py`)

**Files:**
- Create: `scripts/import_ingredients_from_fmp.py`
- Test: `tests/test_import_ingredients.py`

**Interfaces — Produces (pure-ish, take an open `cx`):**
- helpers `_active(v)`, `_num(v)`, `_clean(v)`, `_extras(row, mapped)`.
- `import_suppliers(cx, rows) -> int`, `import_ingredients(cx, rows) -> int`, `import_sources(cx, rows) -> int`, `apply_canonical(cx, cluster_rows) -> dict` (counts + skipped).
- CLI: dry-run default (counts only), `--write`.

**Mapping (verified against the real CSVs):**
- suppliers: `id_pk`→fmp_id; company; address_street/city/province/postal_code; email; phone_business/cell/fax; url; qb_id; `active` (Yes/No); extras = all non-`z*`/non-audit/non-mapped cols. Curated `notes` preserved.
- ingredients: `id_pk`→fmp_id; name = first non-empty of (name_common,name_compound,name_scientific,name_species,name_favorite) with newlines→spaces, else `"(unnamed FMP ingredient <id>)"`; common_names = JSON of the OTHER non-empty name variants; form; status = active(Yes)→'active' else 'inactive'; extras = non-`z*`/non-audit/non-mapped. Curated cols + canonical_id preserved.
- ingredient_sources: `id_pk`→fmp_id; `id_fk_raw`→ingredient (via fmp_id map); `id_fk_supplier`→supplier (via fmp_id map) + supplier_name from that supplier's company; `product_id`→sku; `price`→price_per_unit; `purchase_size`→unit_size; `purchase_size_unit`→unit_type; `shipping`→shipping_quote; extras = {contact,product_link,mfg,quoted_amount,active,...} non-`z*`/non-audit/non-mapped. Curated cols preserved.
- canonical: clear all canonical_id, then for each cluster row set member.canonical_id = head's local id, skipping members/heads not present and any member that is also a head.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_ingredients.py
import sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from scripts.import_ingredients_from_fmp import (
    _active, _num, _clean, import_suppliers, import_ingredients, import_sources, apply_canonical)

def _db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx)
    return p

def test_helpers():
    assert _active("Yes") == 1 and _active("No") == 0 and _active("") is None
    assert _num("350") == 350.0 and _num("$1,000.5") == 1000.5 and _num("") is None
    assert _clean("Inositol\nFlush Free") == "Inositol Flush Free"

def test_import_and_join_and_canonical(tmp_path):
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        import_suppliers(cx, [{"id_pk":"s1","company":"Acme","active":"Yes","notes":"x","zc_junk":"drop"}])
        import_ingredients(cx, [
            {"id_pk":"5161","name_common":"CBD","active":"Yes","type":"Cannabinoid"},
            {"id_pk":"4138","name_compound":"Cannabidiol\nfull spectrum","active":"Yes"},
            {"id_pk":"i3","name_common":"","name_compound":""},   # unnamed fallback
        ])
        import_sources(cx, [{"id_pk":"src1","id_fk_raw":"5161","id_fk_supplier":"s1",
                             "product_id":"SKU9","price":"350","purchase_size":"1000","purchase_size_unit":"g"}])
        res = apply_canonical(cx, [{"head_fmp_id":"5161","member_fmp_id":"4138"}])
        cx.commit()
        ing = {r["fmp_id"]: dict(r) for r in cx.execute("SELECT * FROM ingredients")}
        src = cx.execute("SELECT s.*, sup.company FROM ingredient_sources s JOIN suppliers sup ON sup.id=s.supplier_id").fetchone()
    assert ing["i3"]["name"] == "(unnamed FMP ingredient i3)"
    assert ing["4138"]["name"] == "Cannabidiol full spectrum"
    assert ing["4138"]["canonical_id"] == ing["5161"]["id"]   # member -> head
    assert ing["5161"]["canonical_id"] is None                # head not clustered under anyone
    assert src["sku"] == "SKU9" and src["price_per_unit"] == 350.0 and src["unit_type"] == "g"
    assert src["company"] == "Acme"
    assert res["applied"] == 1

def test_reimport_preserves_curated(tmp_path):
    p = _db(tmp_path)
    from dashboard.ingredient_catalog import update_ingredient_curated, update_source_curated
    with sqlite3.connect(p) as cx:
        import_suppliers(cx, [{"id_pk":"s1","company":"Acme"}])
        import_ingredients(cx, [{"id_pk":"i1","name_common":"R-Lipoic Acid","type":"old"}])
        import_sources(cx, [{"id_pk":"src1","id_fk_raw":"i1","id_fk_supplier":"s1","price":"10"}])
        cx.commit()
        iid = cx.execute("SELECT id FROM ingredients WHERE fmp_id='i1'").fetchone()[0]
        sid = cx.execute("SELECT id FROM ingredient_sources WHERE fmp_id='src1'").fetchone()[0]
    update_ingredient_curated(iid, {"inci_name": "Thioctic Acid"}, db_path=p)
    update_source_curated(sid, {"lead_time_days": 21}, db_path=p)
    with sqlite3.connect(p) as cx:  # re-import with changed FMP data
        cx.row_factory = sqlite3.Row
        import_ingredients(cx, [{"id_pk":"i1","name_common":"R-Lipoic Acid","type":"new"}])
        import_sources(cx, [{"id_pk":"src1","id_fk_raw":"i1","id_fk_supplier":"s1","price":"99"}])
        cx.commit()
        ing = cx.execute("SELECT * FROM ingredients WHERE fmp_id='i1'").fetchone()
        src = cx.execute("SELECT * FROM ingredient_sources WHERE fmp_id='src1'").fetchone()
    assert ing["inci_name"] == "Thioctic Acid"      # curated preserved
    assert '"type": "new"' in ing["extras"]          # FMP refreshed
    assert src["lead_time_days"] == 21               # curated preserved
    assert src["price_per_unit"] == 99.0             # FMP refreshed
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_import_ingredients.py -q`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement `scripts/import_ingredients_from_fmp.py`**

```python
"""Import FileMaker raw-material data into chat_log.db (suppliers, ingredients,
ingredient_sources) + apply canonical clustering. Idempotent by fmp_id; preserves
console-edited curated columns. Dry-run default; --write to persist."""
from __future__ import annotations
import argparse, csv, json, os, re, sqlite3, sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

EXPORT_DIR = os.environ.get("FMP_EXPORT_DIR", "/tmp/fmp-export/newapp")
CANON_CSV = os.environ.get("FMP_CANONICAL_CSV",
    str(Path(__file__).resolve().parent.parent.parent / "AI-Training" / "02 Skills" / "fmp-loaders" / "mapping" / "canonical_clusters.csv"))

_AUDIT = {"PrimaryKey","CreationTimestamp","CreatedBy","ModificationTimestamp","ModifiedBy","id_pk"}


def _active(v):
    s = (v or "").strip().lower()
    if s in ("yes","1","true","y"): return 1
    if s in ("no","0","false","n"): return 0
    return None


def _num(v):
    if v is None: return None
    s = str(v).replace(",", "").strip()
    if not s: return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _clean(v):
    if not v: return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def _extras(row, mapped):
    out = {}
    for k, v in row.items():
        if k in mapped or k in _AUDIT or k.startswith("z") or not (v or "").strip():
            continue
        out[k] = v
    return json.dumps(out, ensure_ascii=False) if out else None


def _upsert(cx, table, fmp_cols, values, conflict_update_cols):
    cols = ["fmp_id"] + fmp_cols
    ph = ",".join("?" for _ in cols)
    setc = ", ".join(f"{c}=excluded.{c}" for c in fmp_cols) + ", updated_at=datetime('now')"
    cx.execute(
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({ph}) "
        f"ON CONFLICT(fmp_id) DO UPDATE SET {setc}",
        values,
    )


def import_suppliers(cx, rows):
    n = 0
    fmp_cols = ["company","address_street","address_city","address_province","address_postal_code",
                "email","phone_business","phone_cell","phone_fax","url","qb_id","active","extras"]
    mapped = set(fmp_cols) | {"id_pk"} | {"notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid: continue
        vals = [fid, _clean(r.get("company")) or "(unknown company)",
                r.get("address_street"), r.get("address_city"), r.get("address_province"), r.get("address_postal_code"),
                r.get("email"), r.get("phone_business"), r.get("phone_cell"), r.get("phone_fax"), r.get("url"),
                r.get("qb_id"), _active(r.get("active")), _extras(r, mapped)]
        _upsert(cx, "suppliers", fmp_cols, vals, fmp_cols)
        n += 1
    return n


_NAME_FIELDS = ["name_common","name_compound","name_scientific","name_species","name_favorite"]


def import_ingredients(cx, rows):
    n = 0
    fmp_cols = ["name","form","status","common_names","extras"]
    mapped = set(fmp_cols) | {"id_pk"} | set(_NAME_FIELDS) | {"active","form",
        "inci_name","cas_number","hygroscopic_rating","solubility","stability_notes","spec_notes","notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid: continue
        names = [_clean(r.get(f)) for f in _NAME_FIELDS if _clean(r.get(f))]
        name = names[0] if names else f"(unnamed FMP ingredient {fid})"
        commons = json.dumps([x for x in names[1:]], ensure_ascii=False) if len(names) > 1 else None
        status = "active" if _active(r.get("active")) == 1 else "inactive"
        vals = [fid, name, _clean(r.get("form")) or None, status, commons, _extras(r, mapped)]
        _upsert(cx, "ingredients", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def import_sources(cx, rows):
    ing = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM ingredients WHERE fmp_id IS NOT NULL")}
    sup = {r["fmp_id"]: (r["id"], r["company"]) for r in cx.execute("SELECT id, fmp_id, company FROM suppliers WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = ["ingredient_id","supplier_id","supplier_name","sku","price_per_unit","unit_size","unit_type","shipping_quote","extras"]
    mapped = set(fmp_cols) | {"id_pk","id_fk_raw","id_fk_supplier","product_id","price","purchase_size","purchase_size_unit","shipping",
        "preferred","lead_time_days","minimum_order","minimum_order_unit","notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid: continue
        iid = ing.get((r.get("id_fk_raw") or "").strip())
        s = sup.get((r.get("id_fk_supplier") or "").strip())
        sid, sname = (s[0], s[1]) if s else (None, None)
        vals = [fid, iid, sid, sname, _clean(r.get("product_id")) or None,
                _num(r.get("price")), _num(r.get("purchase_size")), _clean(r.get("purchase_size_unit")) or None,
                _num(r.get("shipping")), _extras(r, mapped)]
        _upsert(cx, "ingredient_sources", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def apply_canonical(cx, cluster_rows):
    idmap = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM ingredients WHERE fmp_id IS NOT NULL")}
    heads = {str(r["head_fmp_id"]).strip() for r in cluster_rows}
    cx.execute("UPDATE ingredients SET canonical_id=NULL")
    applied, skipped = 0, []
    for r in cluster_rows:
        h, m = str(r["head_fmp_id"]).strip(), str(r["member_fmp_id"]).strip()
        if h not in idmap or m not in idmap:
            skipped.append((h, m, "missing")); continue
        if m in heads:
            skipped.append((h, m, "member-is-head")); continue
        cx.execute("UPDATE ingredients SET canonical_id=? WHERE id=?", (idmap[h], idmap[m]))
        applied += 1
    return {"applied": applied, "skipped": len(skipped)}


def _read_csv(name):
    p = os.path.join(EXPORT_DIR, name)
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    suppliers = _read_csv("suppliers.csv")
    ingredients = _read_csv("ingredients.csv")
    sources = _read_csv("ingredients_supplier.csv")
    with open(CANON_CSV, newline="") as f:
        clusters = list(csv.DictReader(f))
    print(f"suppliers={len(suppliers)} ingredients={len(ingredients)} sources={len(sources)} clusters={len(clusters)}")
    if not args.write:
        print("(dry run — pass --write to import)")
        return 0
    from dashboard.ingredient_catalog import _default_db_path, init_ingredients_schema
    cx = sqlite3.connect(args.db or _default_db_path())
    cx.row_factory = sqlite3.Row
    init_ingredients_schema(cx)
    ns = import_suppliers(cx, suppliers)
    ni = import_ingredients(cx, ingredients)
    nsrc = import_sources(cx, sources)
    canon = apply_canonical(cx, clusters)
    cx.commit(); cx.close()
    print(f"wrote suppliers={ns} ingredients={ni} sources={nsrc} canonical_applied={canon['applied']} skipped={canon['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, then a real dry-run + a real --write into a temp DB**

Run: `python3 -m pytest tests/test_import_ingredients.py -q` → PASS
Run: `python3 scripts/import_ingredients_from_fmp.py` → prints `suppliers=1058 ingredients=2363 sources=3980 clusters=528`
Run: `python3 scripts/import_ingredients_from_fmp.py --write --db /tmp/ing_test.db` → prints written counts; assert `python3 -c "import sqlite3;c=sqlite3.connect('/tmp/ing_test.db');print(c.execute('SELECT count(*) FROM ingredients').fetchone(),c.execute('SELECT count(*) FROM ingredient_sources WHERE ingredient_id IS NOT NULL').fetchone(),c.execute('SELECT count(*) FROM ingredients WHERE canonical_id IS NOT NULL').fetchone())"` — capture in report. (Do NOT touch the repo/prod `chat_log.db`.)

- [ ] **Step 5: Commit**

```bash
git add scripts/import_ingredients_from_fmp.py tests/test_import_ingredients.py
git commit -m "feat(ingredients): FMP importer (suppliers/ingredients/sources + canonical, curated-preserving)"
```

---

### Task 4: Console API endpoints + search registration

**Files:**
- Modify: `app.py` (page route + `/api/ingredients/*` endpoints near the `/api/shipping/*` block ~line 20207+)
- Modify: `static/console-search-index.json`
- Test: `tests/test_admin_ingredients_api.py`

**Interfaces — Consumes Tasks 1-2:** `search_ingredients,get_ingredient,list_sources_for_ingredient,list_suppliers,update_ingredient_curated,update_source_curated,set_preferred_source,update_supplier`.

- [ ] **Step 1: Write the failing test** (route-level, Pinecone-skip pattern)

```python
# tests/test_admin_ingredients_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx)
        cx.execute("INSERT INTO ingredients (fmp_id,name) VALUES ('i1','R-Lipoic Acid')")
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')")
        cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client(), db

def test_search_and_curated(tmp_path, monkeypatch):
    c, db = _client(tmp_path, monkeypatch)
    r = c.get("/api/ingredients/search?q=lipoic").get_json()
    assert r["data"][0]["name"] == "R-Lipoic Acid"
    iid = r["data"][0]["id"]
    assert c.patch(f"/api/ingredients/{iid}", json={"inci_name":"Thioctic Acid"}).status_code == 200
    d = c.get(f"/api/ingredients/{iid}").get_json()["data"]
    assert d["ingredient"]["inci_name"] == "Thioctic Acid" and "sources" in d
    assert c.get("/api/ingredients/suppliers").get_json()["data"][0]["company"] == "Acme"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_admin_ingredients_api.py -q`
Expected: FAIL (404 / endpoints missing) — or SKIP if app import needs Pinecone locally (runs in CI).

- [ ] **Step 3: Implement endpoints** (in `app.py`, beside the shipping API block)

```python
from dashboard import ingredient_catalog as _ingredients

@app.route("/admin/ingredients")
def admin_ingredients_page():
    resp = send_from_directory(STATIC, "admin-ingredients.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.route("/api/ingredients/search", methods=["GET"])
@require_console_key
def api_ingredients_search():
    try:
        q = request.args.get("q", ""); limit = int(request.args.get("limit", 50)); offset = int(request.args.get("offset", 0))
        return ok(_ingredients.search_ingredients(q, limit, offset))
    except Exception as e: return fail(e)

@app.route("/api/ingredients/suppliers", methods=["GET"])
@require_console_key
def api_ingredients_suppliers():
    try: return ok(_ingredients.list_suppliers(request.args.get("q", ""), int(request.args.get("limit", 50))))
    except Exception as e: return fail(e)

@app.route("/api/ingredients/suppliers/<int:sid>", methods=["PATCH"])
@require_console_key
def api_ingredients_supplier_patch(sid):
    try:
        _ingredients.update_supplier(sid, request.get_json(silent=True) or {})
        return ok(_ingredients.get_supplier(sid))
    except Exception as e: return fail(e)

@app.route("/api/ingredients/<int:iid>", methods=["GET"])
@require_console_key
def api_ingredients_get(iid):
    try:
        ing = _ingredients.get_ingredient(iid)
        if not ing: return fail("not found", status=404)
        return ok({"ingredient": ing, "sources": _ingredients.list_sources_for_ingredient(iid)})
    except Exception as e: return fail(e)

@app.route("/api/ingredients/<int:iid>", methods=["PATCH"])
@require_console_key
def api_ingredients_patch(iid):
    try:
        _ingredients.update_ingredient_curated(iid, request.get_json(silent=True) or {})
        return ok(_ingredients.get_ingredient(iid))
    except Exception as e: return fail(e)

@app.route("/api/ingredients/sources/<int:src_id>", methods=["PATCH"])
@require_console_key
def api_ingredients_source_patch(src_id):
    try:
        _ingredients.update_source_curated(src_id, request.get_json(silent=True) or {})
        return ok({"id": src_id})
    except Exception as e: return fail(e)

@app.route("/api/ingredients/sources/<int:src_id>/preferred", methods=["POST"])
@require_console_key
def api_ingredients_source_preferred(src_id):
    try:
        _ingredients.set_preferred_source(src_id)
        return ok({"id": src_id})
    except Exception as e: return fail(e)
```

- [ ] **Step 4: Register in console search** — add to `static/console-search-index.json`:

```json
{ "title": "Ingredients & Sources", "page": "Products", "url": "/admin/ingredients", "keywords": ["ingredient","ingredients","sources","supplier","raw","cost","sourcing","canonical"] }
```

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_admin_ingredients_api.py -q` → PASS (or SKIP locally)
```bash
git add app.py static/console-search-index.json tests/test_admin_ingredients_api.py
git commit -m "feat(ingredients): console API endpoints + search registration"
```

---

### Task 5: Console page `static/admin-ingredients.html`

**Files:**
- Create: `static/admin-ingredients.html`

**Interfaces — Consumes Task 4 endpoints.** Mirror `static/admin-shipping.html` markup/JS style (same header, fonts, `X-Console-Key`-via-`?key=` handling). Don't restyle.

- [ ] **Step 1: Read `static/admin-shipping.html`** to copy its shell (page chrome, fetch helper that forwards `?key=`, CSS classes).

- [ ] **Step 2: Build the page** — three regions, vanilla JS calling the Task 4 endpoints:
  1. **Search bar** → `GET /api/ingredients/search?q=` → list of ingredient names (click to open detail). Debounced; server-side search (2,363 rows, never load all).
  2. **Ingredient detail** → `GET /api/ingredients/<id>` → show FMP fields (name, form, status, common_names, parsed `extras`) **read-only**; editable curated inputs (`inci_name,cas_number,hygroscopic_rating,solubility,stability_notes,spec_notes,notes`) that `PATCH /api/ingredients/<id>` on save; a **Sources** table (supplier company, sku, price_per_unit, unit_size+unit_type, shipping_quote) with editable curated cols (`lead_time_days,minimum_order,minimum_order_unit,notes` → `PATCH /api/ingredients/sources/<id>`) and a **Preferred** radio/toggle (`POST .../preferred`).
  3. **Suppliers tab** → `GET /api/ingredients/suppliers?q=` list; editable `notes` (`PATCH /api/ingredients/suppliers/<id>`).

- [ ] **Step 3: Manual verify** (endpoint data path already covered by Task 4 tests)

Run the app, open `/admin/ingredients?key=$CONSOLE_SECRET`, search "lipoic", open it, edit `inci_name`, save, reload → persists; toggle a preferred source.

- [ ] **Step 4: Commit**

```bash
git add static/admin-ingredients.html
git commit -m "feat(ingredients): /admin/ingredients console page"
```

---

## Self-Review
**Spec coverage:** suppliers+ingredients+ingredient_sources schema (T1); curated writes + preferred (T2); FMP importer with canonical + curated-preserve (T3); console endpoints (T4) + page (T5); canonicalization included (T3); curated-vs-FMP split enforced (T3 `_upsert` excludes curated cols; T2 whitelists). ✓
**Placeholder scan:** complete code in T1-T4; T5 is HTML mirroring an existing page with explicit endpoint contracts. ✓
**Type consistency:** `_connect`/`db_path` kwarg, `_upsert(cx,table,fmp_cols,values,...)`, `import_*`/`apply_canonical(cx,...)` signatures, endpoint→module function names all consistent across tasks. ✓
**Verification (post-build):** real `--write` into `/tmp/ing_test.db`, assert ~2363 ingredients / ~3900 linked sources / canonical_id set on the CBD cluster members; edit a curated field, re-run import, confirm preserved.

## Operational note (post-merge, Glen)
The importer's real `--write` against prod `chat_log.db` is an operator step (like the bottle-type `--write`): run dry-run, then `--write`, on the deployed instance. The FMP CSVs must be re-exported (AppleScript extractor) before each refresh.
