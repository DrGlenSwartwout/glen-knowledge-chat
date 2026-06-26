# Phase 3a: Materials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate the FileMaker materials master (production inputs + packaging) + its supplier links into `chat_log.db`, with an importer, console, and server-side import — mirroring Phase 1/2.

**Architecture:** New `dashboard/materials_catalog.py` (schema + reads + curated writes, reuses `_connect` from `ingredient_catalog`). Importer `scripts/import_materials_from_fmp.py` reuses Phase-1 helpers. Console = `/api/materials/*` + a Materials tab in `static/admin-ingredients.html`. Server-side import endpoint mirrors `/api/ingredients/import`.

**Tech Stack:** Python 3, Flask, sqlite3, pytest, vanilla-JS. Stdlib only.

## Global Constraints
- New module is **`dashboard/materials_catalog.py`** (verified free; `ls` before create). Reuses `_connect` from `dashboard.ingredient_catalog`. References Phase-1 `suppliers` (FK) and Phase-2 products.json `fmp_id` (for product_suppliers link).
- **Reuse, don't duplicate:** `from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert`.
- Curated-vs-FMP split: importer writes only FMP cols; curated = `materials.notes`; `material_suppliers`/`product_suppliers` `preferred,notes`. `_upsert` enforces it.
- `fmp_id` (= FMP id_pk) idempotent key; partial unique index `WHERE fmp_id IS NOT NULL`. csv.field_size_limit(sys.maxsize). INTEGER bools, REAL money, JSON extras, timestamps `TEXT DEFAULT (datetime('now'))`.
- Console `@require_console_key`, `ok`/`fail`, `?key=`; server import mirrors `/api/ingredients/import` (multipart, dry-run/write, raises field_size_limit, lazy imports, cx try/finally).
- FMP source (exported, verified): `/tmp/fmp-export/newapp/{materials,materials_supplier,products_supplier}.csv` (49/56/210). Fields — materials: `id_pk,material_name,type,active,notes,par_level,inventory_starting,purchase_count`. materials_supplier: `id_pk,id_fk_material,id_fk_supplier,product_id,price,purchase_size,purchase_size_unit,mfg,contact,product_link,active`. products_supplier: same but `id_fk_product` instead of id_fk_material.

---

### Task 1: `dashboard/materials_catalog.py` — schema + reads + curated writes + wiring

**Files:** Create `dashboard/materials_catalog.py`; Modify `app.py` (`_init_materials_tables()` after `_init_formulations_tables()`); Test `tests/test_materials_catalog.py`.

**Produces:** `init_materials_schema(cx)`; `search_materials(q,limit,offset,db_path)`; `get_material(id,db_path)`; `list_suppliers_for_material(material_id,db_path)` (LEFT JOIN suppliers.company); `list_product_suppliers(fmp_product_id,db_path)`; `update_material_curated(id,fields,db_path)` (only `notes`); `update_material_supplier_curated(id,fields,db_path)` (`preferred,notes`); `set_preferred_material_supplier(id,db_path)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_materials_catalog.py
import sqlite3, pytest

@pytest.fixture
def db(tmp_path):
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx)   # suppliers table (material_suppliers FK)
        init_materials_schema(cx); init_materials_schema(cx)  # idempotent
    return p

def test_schema_reads_curated(db):
    from dashboard.materials_catalog import (search_materials, get_material,
        list_suppliers_for_material, update_material_curated, set_preferred_material_supplier)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')")
        sid = cx.execute("SELECT id FROM suppliers").fetchone()[0]
        cx.execute("INSERT INTO materials (fmp_id,name) VALUES ('m1','Pullulan Caps')")
        mid = cx.execute("SELECT id FROM materials").fetchone()[0]
        cx.execute("INSERT INTO material_suppliers (fmp_id,material_id,supplier_id,price) VALUES ('a',?,?,10)", (mid,sid))
        cx.execute("INSERT INTO material_suppliers (fmp_id,material_id,supplier_id,price) VALUES ('b',?,?,20)", (mid,sid))
        ids=[r[0] for r in cx.execute("SELECT id FROM material_suppliers ORDER BY id")]; cx.commit()
    assert search_materials("pullulan", db_path=db)[0]["name"] == "Pullulan Caps"
    sup = list_suppliers_for_material(mid, db_path=db)
    assert sup[0]["company"] == "Acme"
    update_material_curated(mid, {"notes":"x","name":"HACK"}, db_path=db)
    assert get_material(mid, db_path=db)["notes"]=="x" and get_material(mid, db_path=db)["name"]=="Pullulan Caps"
    set_preferred_material_supplier(ids[1], db_path=db)
    pref={r["id"]:r for r in list_suppliers_for_material(mid, db_path=db)}
    assert pref[ids[1]]["preferred"]==1 and pref[ids[0]]["preferred"]==0
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_materials_catalog.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement `dashboard/materials_catalog.py`**

```python
"""Materials (production inputs + packaging) catalog — FMP-migrated.
Mirrors dashboard/ingredient_catalog.py; references Phase-1 suppliers."""
from __future__ import annotations
import sqlite3
from dashboard.ingredient_catalog import _connect


def init_materials_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS materials (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT, name TEXT NOT NULL, type TEXT, status TEXT,
          extras TEXT, notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_materials_fmp ON materials(fmp_id) WHERE fmp_id IS NOT NULL")
    for tbl, link in (("material_suppliers", "material_id INTEGER REFERENCES materials(id)"),
                      ("product_suppliers", "fmp_product_id TEXT")):
        cx.execute(f"""
            CREATE TABLE IF NOT EXISTS {tbl} (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              fmp_id TEXT, {link},
              supplier_id INTEGER REFERENCES suppliers(id),
              supplier_name TEXT, sku TEXT, price REAL,
              purchase_size REAL, purchase_size_unit TEXT, mfg TEXT, contact TEXT, product_link TEXT,
              extras TEXT, preferred INTEGER DEFAULT 0, notes TEXT,
              created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
            )""")
        cx.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{tbl}_fmp ON {tbl}(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_matsup_mat ON material_suppliers(material_id)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_prodsup_fpid ON product_suppliers(fmp_product_id)")
    cx.commit()


def search_materials(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("SELECT * FROM materials WHERE name LIKE ? ORDER BY name LIMIT ? OFFSET ?",
                          (f"%{q}%", int(limit), int(offset))).fetchall()
    return [dict(r) for r in rows]


def get_material(material_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("SELECT * FROM materials WHERE id=?", (material_id,)).fetchone()
    return dict(r) if r else None


def list_suppliers_for_material(material_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT ms.*, sup.company AS company FROM material_suppliers ms
            LEFT JOIN suppliers sup ON sup.id = ms.supplier_id
            WHERE ms.material_id = ? ORDER BY ms.preferred DESC, ms.price
        """, (material_id,)).fetchall()
    return [dict(r) for r in rows]


def list_product_suppliers(fmp_product_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT ps.*, sup.company AS company FROM product_suppliers ps
            LEFT JOIN suppliers sup ON sup.id = ps.supplier_id
            WHERE ps.fmp_product_id = ? ORDER BY ps.preferred DESC, ps.price
        """, (str(fmp_product_id),)).fetchall()
    return [dict(r) for r in rows]


_MAT_CURATED = {"notes"}
_SUP_CURATED = {"preferred", "notes"}


def _update_allowed(table, row_id, fields, allowed, db_path):
    cols = {k: v for k, v in (fields or {}).items() if k in allowed}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*cols.values(), row_id))
        cx.commit()


def update_material_curated(material_id, fields, db_path=None):
    _update_allowed("materials", material_id, fields, _MAT_CURATED, db_path)


def update_material_supplier_curated(ms_id, fields, db_path=None):
    _update_allowed("material_suppliers", ms_id, fields, _SUP_CURATED, db_path)


def set_preferred_material_supplier(ms_id, db_path=None):
    with _connect(db_path) as cx:
        row = cx.execute("SELECT material_id FROM material_suppliers WHERE id=?", (ms_id,)).fetchone()
        if not row:
            return
        cx.execute("UPDATE material_suppliers SET preferred=0, updated_at=datetime('now') WHERE material_id=?", (row["material_id"],))
        cx.execute("UPDATE material_suppliers SET preferred=1, updated_at=datetime('now') WHERE id=?", (ms_id,))
        cx.commit()
```

- [ ] **Step 4: Wire at app load** — in `app.py` after `_init_formulations_tables()`:

```python
def _init_materials_tables():
    from dashboard.materials_catalog import init_materials_schema
    with sqlite3.connect(LOG_DB) as cx:
        init_materials_schema(cx)

_init_materials_tables()
```

- [ ] **Step 5: Run + commit** — `python3 -m pytest tests/test_materials_catalog.py -q` → PASS.
```bash
git add dashboard/materials_catalog.py app.py tests/test_materials_catalog.py
git commit -m "feat(materials): schema + reads + curated writes + app wiring"
```

---

### Task 2: FMP materials importer

**Files:** Create `scripts/import_materials_from_fmp.py`; Test `tests/test_import_materials.py`.

**Produces:** `import_materials(cx, rows) -> int`; `import_material_suppliers(cx, rows) -> int` (resolve material_id via id_fk_material→materials.fmp_id, supplier_id via id_fk_supplier→suppliers.fmp_id); `import_product_suppliers(cx, rows) -> int` (store id_fk_product as fmp_product_id; supplier_id via id_fk_supplier); CLI dry-run/`--write`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_materials.py
import sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from dashboard.materials_catalog import init_materials_schema
from scripts.import_materials_from_fmp import import_materials, import_material_suppliers, import_product_suppliers

def _db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')"); cx.commit()
    return p

def test_import(tmp_path):
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        nm = import_materials(cx, [{"id_pk":"m1","material_name":"Pullulan Caps","type":"Capsule","active":"Yes","par_level":"500"}])
        nms = import_material_suppliers(cx, [{"id_pk":"ms1","id_fk_material":"m1","id_fk_supplier":"s1","product_id":"SKU1","price":"10","purchase_size":"1000","purchase_size_unit":"ea"}])
        nps = import_product_suppliers(cx, [{"id_pk":"ps1","id_fk_product":"5161","id_fk_supplier":"s1","product_id":"SKU2","price":"20"}])
        cx.commit()
        m = cx.execute("SELECT * FROM materials WHERE fmp_id='m1'").fetchone()
        ms = cx.execute("SELECT ms.*, sup.company FROM material_suppliers ms JOIN suppliers sup ON sup.id=ms.supplier_id").fetchone()
        ps = cx.execute("SELECT * FROM product_suppliers WHERE fmp_id='ps1'").fetchone()
    assert nm==1 and m["name"]=="Pullulan Caps" and m["status"]=="active" and '"par_level": "500"' in m["extras"]
    assert nms==1 and ms["material_id"]==m["id"] and ms["sku"]=="SKU1" and ms["price"]==10.0 and ms["company"]=="Acme"
    assert nps==1 and ps["fmp_product_id"]=="5161" and ps["price"]==20.0

def test_reimport_preserves_curated(tmp_path):
    p = _db(tmp_path)
    from dashboard.materials_catalog import update_material_curated
    with sqlite3.connect(p) as cx:
        import_materials(cx, [{"id_pk":"m1","material_name":"Caps","active":"Yes"}]); cx.commit()
        mid = cx.execute("SELECT id FROM materials WHERE fmp_id='m1'").fetchone()[0]
    update_material_curated(mid, {"notes":"keep"}, db_path=p)
    with sqlite3.connect(p) as cx:
        import_materials(cx, [{"id_pk":"m1","material_name":"Caps RENAMED","active":"No"}]); cx.commit()
        cx.row_factory=sqlite3.Row
        m = cx.execute("SELECT * FROM materials WHERE fmp_id='m1'").fetchone()
    assert m["name"]=="Caps RENAMED" and m["notes"]=="keep"
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_import_materials.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement `scripts/import_materials_from_fmp.py`**

```python
"""Import FileMaker materials + material/product supplier links into chat_log.db.
Idempotent by fmp_id; curated-preserving. Dry-run default; --write."""
from __future__ import annotations
import argparse, csv, os, sqlite3, sys
csv.field_size_limit(sys.maxsize)
from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert

EXPORT_DIR = os.environ.get("FMP_EXPORT_DIR", "/tmp/fmp-export/newapp")


def import_materials(cx, rows):
    n = 0
    fmp_cols = ["name", "type", "status", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "material_name", "active", "notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        name = _clean(r.get("material_name")) or f"(unnamed FMP material {fid})"
        status = "active" if _active(r.get("active")) == 1 else "inactive"
        _upsert(cx, "materials", fmp_cols, [fid, name, _clean(r.get("type")) or None, status, _extras(r, mapped)], fmp_cols)
        n += 1
    return n


def _import_supplier_links(cx, rows, table, link_col, link_resolver):
    sup = {r["fmp_id"]: (r["id"], r["company"]) for r in cx.execute("SELECT id, fmp_id, company FROM suppliers WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = [link_col, "supplier_id", "supplier_name", "sku", "price", "purchase_size", "purchase_size_unit", "mfg", "contact", "product_link", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_material", "id_fk_product", "id_fk_supplier", "product_id", "active", "preferred", "notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        link_val = link_resolver(r)
        s = sup.get((r.get("id_fk_supplier") or "").strip())
        sid, sname = (s[0], s[1]) if s else (None, None)
        vals = [fid, link_val, sid, sname, _clean(r.get("product_id")) or None, _num(r.get("price")),
                _num(r.get("purchase_size")), _clean(r.get("purchase_size_unit")) or None, _clean(r.get("mfg")) or None,
                _clean(r.get("contact")) or None, _clean(r.get("product_link")) or None, _extras(r, mapped)]
        _upsert(cx, table, fmp_cols, vals, fmp_cols)
        n += 1
    return n


def import_material_suppliers(cx, rows):
    mat = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM materials WHERE fmp_id IS NOT NULL")}
    return _import_supplier_links(cx, rows, "material_suppliers", "material_id",
                                  lambda r: mat.get((r.get("id_fk_material") or "").strip()))


def import_product_suppliers(cx, rows):
    return _import_supplier_links(cx, rows, "product_suppliers", "fmp_product_id",
                                  lambda r: (r.get("id_fk_product") or "").strip() or None)


def _read(name):
    with open(os.path.join(EXPORT_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    materials = _read("materials.csv"); msup = _read("materials_supplier.csv"); psup = _read("products_supplier.csv")
    print(f"materials={len(materials)} material_suppliers={len(msup)} product_suppliers={len(psup)}")
    if not args.write:
        print("(dry run — pass --write)")
        return 0
    from dashboard.ingredient_catalog import _default_db_path, init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    cx = sqlite3.connect(args.db or _default_db_path()); cx.row_factory = sqlite3.Row
    init_ingredients_schema(cx); init_materials_schema(cx)
    nm = import_materials(cx, materials); nms = import_material_suppliers(cx, msup); nps = import_product_suppliers(cx, psup)
    cx.commit(); cx.close()
    print(f"wrote materials={nm} material_suppliers={nms} product_suppliers={nps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests + real --write into a temp DB**

Run: `python3 -m pytest tests/test_import_materials.py -q` → PASS.
Run: `python3 scripts/import_materials_from_fmp.py --write --db /tmp/p3.db` (after seeding suppliers: first run `python3 scripts/import_ingredients_from_fmp.py --write --db /tmp/p3.db`). Expected `materials=49 material_suppliers=56 product_suppliers=210`. Capture in report.

- [ ] **Step 5: Commit**
```bash
git add scripts/import_materials_from_fmp.py tests/test_import_materials.py
git commit -m "feat(materials): FMP importer (materials + material/product suppliers)"
```

---

### Task 3: Materials console (endpoints + tab)

**Files:** Modify `app.py` (`/api/materials/*`), `static/admin-ingredients.html` (Materials tab); Test `tests/test_admin_materials_api.py`.

**Consumes Task 1 reads/writes.** Endpoints (all `@require_console_key`, `ok`/`fail`, `from dashboard import materials_catalog as _materials`): GET `/api/materials/search`; GET `/api/materials/<int:mid>` → `{material, suppliers}` (404 if None); PATCH `/api/materials/<int:mid>` (update_material_curated); PATCH `/api/materials/suppliers/<int:ms_id>` (update_material_supplier_curated); POST `/api/materials/suppliers/<int:ms_id>/preferred` (set_preferred_material_supplier).

- [ ] **Step 1: Write the failing test** (route-level, Pinecone-skip, mirror `tests/test_admin_formulations_api.py`)

```python
# tests/test_admin_materials_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        cx.execute("INSERT INTO materials (fmp_id,name) VALUES ('m1','Pullulan Caps')"); cx.commit()
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod.app.test_client()

def test_search_get_patch(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/materials/search?q=caps").get_json()
    mid = r["data"][0]["id"]
    d = c.get(f"/api/materials/{mid}").get_json()["data"]
    assert d["material"]["name"]=="Pullulan Caps" and "suppliers" in d
    assert c.patch(f"/api/materials/{mid}", json={"notes":"x"}).status_code == 200
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404) or SKIP.

- [ ] **Step 3: Implement endpoints** (in `app.py`, beside `/api/formulations/*`)

```python
from dashboard import materials_catalog as _materials

@app.route("/api/materials/search", methods=["GET"])
@require_console_key
def api_materials_search():
    try:
        return ok(_materials.search_materials(request.args.get("q",""), int(request.args.get("limit",50)), int(request.args.get("offset",0))))
    except Exception as e: return fail(e)

@app.route("/api/materials/<int:mid>", methods=["GET"])
@require_console_key
def api_materials_get(mid):
    try:
        m = _materials.get_material(mid)
        if not m: return fail("not found", status=404)
        return ok({"material": m, "suppliers": _materials.list_suppliers_for_material(mid)})
    except Exception as e: return fail(e)

@app.route("/api/materials/<int:mid>", methods=["PATCH"])
@require_console_key
def api_materials_patch(mid):
    try:
        _materials.update_material_curated(mid, request.get_json(silent=True) or {})
        return ok(_materials.get_material(mid))
    except Exception as e: return fail(e)

@app.route("/api/materials/suppliers/<int:ms_id>", methods=["PATCH"])
@require_console_key
def api_materials_supplier_patch(ms_id):
    try:
        _materials.update_material_supplier_curated(ms_id, request.get_json(silent=True) or {})
        return ok({"id": ms_id})
    except Exception as e: return fail(e)

@app.route("/api/materials/suppliers/<int:ms_id>/preferred", methods=["POST"])
@require_console_key
def api_materials_supplier_preferred(ms_id):
    try:
        _materials.set_preferred_material_supplier(ms_id)
        return ok({"id": ms_id})
    except Exception as e: return fail(e)
```

- [ ] **Step 4: Materials tab in `static/admin-ingredients.html`** — read the page first; add `"materials"` to the `labels` array + a tab button + panel, mirroring the Ingredients tab: debounced search `GET /api/materials/search?q=` → list; on click `GET /api/materials/<id>` → material name/type/status read-only + editable curated `notes` (`PATCH /api/materials/<id>`); suppliers table (company, sku, price, purchase_size+unit, mfg) with editable `notes` (`PATCH /api/materials/suppliers/<id>`) + preferred toggle (`POST .../preferred`). Reuse `api()`, `escapeHtml`, `showTab`. Verify HTML parses; ids exist; existing tabs untouched.

- [ ] **Step 5: Run tests + commit**
```bash
git add app.py static/admin-ingredients.html tests/test_admin_materials_api.py
git commit -m "feat(materials): console endpoints + materials tab"
```

---

### Task 4: Server-side materials import endpoint + UI section

**Files:** Modify `app.py` (`POST /api/materials/import`), `static/admin-ingredients.html` (import section); Test `tests/test_admin_materials_import.py`.

Mirror `/api/ingredients/import` exactly. Multipart files `materials`, `materials_supplier`, `products_supplier` + `write` flag. Dry-run → counts. Write → open LOG_DB, init_ingredients_schema + init_materials_schema, run import_materials + import_material_suppliers + import_product_suppliers, commit, close (try/finally), return counts. Raise field_size_limit; lazy imports; missing any file → 400. UI: a "Materials" section in the Import tab with 3 file pickers + Dry-run/Import (mirror `fmpImport`); hint "import Ingredients first (supplier links resolve via the supplier catalog)".

- [ ] **Step 1–5:** Mirror Task 4 of the Phase-1 activation (`tests/test_admin_ingredients_import.py` is the test template; the `/api/ingredients/import` handler is the endpoint template; the `fmpImport`/`formImport` JS is the UI template). TDD; route test skips locally on Pinecone. Commit `feat(materials): server-side materials import endpoint + UI section`.

---

## Self-Review
**Coverage:** materials+material_suppliers+product_suppliers schema/reads/curated (T1); importer reusing Phase-1 helpers (T2); console (T3); server-side import (T4). ✓
**Placeholders:** complete code T1-T3; T4 explicitly mirrors the Phase-1 activation templates (named files). ✓
**Type consistency:** `_connect`/`db_path`; reused `_active/_num/_clean/_extras/_upsert`; `import_materials/import_material_suppliers/import_product_suppliers`; endpoint→module fn names; `materials_catalog as _materials` alias unique. ✓

## Operational note
Activation (after merge): console Import tab → import Ingredients/Suppliers first, then Materials (3 CSVs). Re-export FMP materials tables via the AppleScript extractor before each refresh.
