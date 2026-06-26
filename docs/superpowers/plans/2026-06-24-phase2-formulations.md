# Phase 2: Formulations (recipes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate FileMaker formulation recipes into `chat_log.db` (referencing Phase-1 ingredients), add a recipe console, and generate the customer-facing `products.json` ingredient panels from the DB — retiring `scripts/refresh_ingredients_from_fmp.py`.

**Architecture:** New `dashboard/formulations.py` (schema + reads + curated writes, mirrors `dashboard/ingredient_catalog.py`). Importer `scripts/import_formulations_from_fmp.py` reuses Phase-1 helpers. A product↔FMP matcher writes a stable `fmp_id` onto `products.json`. A panel generator joins formulations→products by `fmp_id` and writes the panels. Console = `/api/formulations/*` + a tab in `admin-ingredients.html`.

**Tech Stack:** Python 3, Flask, sqlite3 (`chat_log.db`), pytest, vanilla-JS. Stdlib only.

## Global Constraints
- Builds on Phase 1 (merged): `dashboard/ingredient_catalog.py` (tables `suppliers`/`ingredients`/`ingredient_sources`, `_connect`/`_default_db_path`, `init_ingredients_schema`). New module is **`dashboard/formulations.py`** (verified no existing file of that name — confirm with `ls dashboard/formulations.py` before creating, per the grep-before-create lesson).
- **Reuse, don't duplicate:** import helpers from the Phase-1 scripts — `from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert`; matcher helpers `from scripts.populate_bottle_types import _norm, _build_fmp_index`. Do NOT reimplement.
- **Curated-vs-FMP split:** importer refreshes only FMP columns; never curated (`formulations.notes`; `formulation_items.notes`). `_upsert` already enforces this (curated cols never in `fmp_cols`).
- `fmp_id` (= FMP `id_pk`) is the idempotent key; partial unique index `WHERE fmp_id IS NOT NULL`. `csv.field_size_limit(sys.maxsize)`. INTEGER bools, REAL numbers, JSON `TEXT` extras, `TEXT DEFAULT (datetime('now'))`.
- Console: `@require_console_key`, `ok`/`fail`, `?key=` forwarding — same as the existing `/api/ingredients/*` block.
- FMP source (already exported): `/tmp/fmp-export/newapp/products.csv` (formulations = rows where `type='Functional Formulation'`, **181**) + `products_items.csv` (recipe lines; **1,684** belong to FF, **1,274** carry `id_fk_raw`). Storefront catalog: `data/products.json`.
- Decisions (confirmed): (A) store `fmp_id` on `products.json` for a stable recipe↔product link; (B) generate panels now and retire the refresher.

---

### Task 1: `dashboard/formulations.py` — schema + reads + curated writes + app wiring

**Files:**
- Create: `dashboard/formulations.py`
- Modify: `app.py` (add `_init_formulations_tables()` at module load, after `_init_ingredients_tables()`)
- Test: `tests/test_formulations.py`

**Interfaces — Produces:**
- `init_formulations_schema(cx)` — idempotent CREATE of `formulations`, `formulation_items` + indexes.
- `search_formulations(q="", limit=50, offset=0, db_path=None) -> list[dict]`
- `get_formulation(formulation_id, db_path=None) -> dict | None`
- `list_items_for_formulation(formulation_id, db_path=None) -> list[dict]` (LEFT JOIN ingredients for the canonical name + the ingredient's preferred-source price via a correlated subquery — see code).
- `update_formulation_curated(id, fields, db_path=None)` (only `{notes}`); `update_item_curated(id, fields, db_path=None)` (only `{notes}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_formulations.py
import sqlite3, pytest

@pytest.fixture
def db(tmp_path):
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.formulations import init_formulations_schema
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx)           # Phase-1 tables (formulation_items FKs ingredients)
        init_formulations_schema(cx)
        init_formulations_schema(cx)           # idempotent
    return p

def test_schema_and_reads(db):
    from dashboard.formulations import (search_formulations, get_formulation,
        list_items_for_formulation, update_formulation_curated, update_item_curated)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO ingredients (fmp_id, name) VALUES ('r1','R-Lipoic Acid')")
        iid = cx.execute("SELECT id FROM ingredients").fetchone()[0]
        cx.execute("INSERT INTO formulations (fmp_id, name) VALUES ('f1','Nerve Pulse')")
        fid = cx.execute("SELECT id FROM formulations").fetchone()[0]
        cx.execute("INSERT INTO formulation_items (fmp_id, formulation_id, ingredient_id, ingredient_name, dose, dose_unit) "
                   "VALUES ('it1',?,?, 'R-Lipoic Acid', 100, 'mg')", (fid, iid))
        cx.commit()
    assert search_formulations("nerve", db_path=db)[0]["name"] == "Nerve Pulse"
    items = list_items_for_formulation(fid, db_path=db)
    assert items[0]["dose"] == 100 and items[0]["ingredient_name"] == "R-Lipoic Acid"
    assert items[0]["ingredient_canonical"] == "R-Lipoic Acid"   # from the JOIN
    update_formulation_curated(fid, {"notes": "test", "name": "HACK"}, db_path=db)
    assert get_formulation(fid, db_path=db)["notes"] == "test"
    assert get_formulation(fid, db_path=db)["name"] == "Nerve Pulse"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-59a2725d && python3 -m pytest tests/test_formulations.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.formulations'`

- [ ] **Step 3: Implement `dashboard/formulations.py`**

```python
"""Formulations (recipes) — FMP-migrated, references Phase-1 ingredients.
Mirrors dashboard/ingredient_catalog.py conventions."""
from __future__ import annotations
import sqlite3
from dashboard.ingredient_catalog import _connect  # reuse Phase-1 connection helper


def init_formulations_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS formulations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT, name TEXT NOT NULL, status TEXT,
          product_slug TEXT, extras TEXT,
          notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_formulations_fmp ON formulations(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS formulation_items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT,
          formulation_id INTEGER REFERENCES formulations(id),
          ingredient_id INTEGER REFERENCES ingredients(id),
          ingredient_name TEXT, dose REAL, dose_unit TEXT, raw_text TEXT,
          extras TEXT, notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_formitems_fmp ON formulation_items(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_formitems_form ON formulation_items(formulation_id)")
    cx.commit()


def search_formulations(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT * FROM formulations WHERE name LIKE ? ORDER BY name LIMIT ? OFFSET ?",
            (f"%{q}%", int(limit), int(offset)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_formulation(formulation_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("SELECT * FROM formulations WHERE id=?", (formulation_id,)).fetchone()
    return dict(r) if r else None


def list_items_for_formulation(formulation_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT fi.*, ing.name AS ingredient_canonical,
              (SELECT price_per_unit FROM ingredient_sources s
               WHERE s.ingredient_id = fi.ingredient_id
               ORDER BY s.preferred DESC, s.price_per_unit LIMIT 1) AS preferred_price
            FROM formulation_items fi
            LEFT JOIN ingredients ing ON ing.id = fi.ingredient_id
            WHERE fi.formulation_id = ?
            ORDER BY fi.id
        """, (formulation_id,)).fetchall()
    return [dict(r) for r in rows]


_FORM_CURATED = {"notes"}
_ITEM_CURATED = {"notes"}


def _update_allowed(table, row_id, fields, allowed, db_path):
    cols = {k: v for k, v in (fields or {}).items() if k in allowed}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*cols.values(), row_id))
        cx.commit()


def update_formulation_curated(formulation_id, fields, db_path=None):
    _update_allowed("formulations", formulation_id, fields, _FORM_CURATED, db_path)


def update_item_curated(item_id, fields, db_path=None):
    _update_allowed("formulation_items", item_id, fields, _ITEM_CURATED, db_path)
```

- [ ] **Step 4: Wire schema init at app load** — in `app.py`, after the `_init_ingredients_tables()` call, add:

```python
def _init_formulations_tables():
    """Formulations (recipes) — FMP-migrated, references Phase-1 ingredients."""
    from dashboard.formulations import init_formulations_schema
    with sqlite3.connect(LOG_DB) as cx:
        init_formulations_schema(cx)

_init_formulations_tables()
```

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_formulations.py -q` → PASS
```bash
git add dashboard/formulations.py app.py tests/test_formulations.py
git commit -m "feat(formulations): schema + reads + curated writes + app wiring"
```

---

### Task 2: FMP formulations importer

**Files:**
- Create: `scripts/import_formulations_from_fmp.py`
- Test: `tests/test_import_formulations.py`

**Interfaces:**
- Consumes Phase-1 helpers `_active,_num,_clean,_extras,_upsert` (from `scripts.import_ingredients_from_fmp`); Phase-1 `ingredients.fmp_id` rows.
- Produces: `import_formulations(cx, product_rows) -> int` (only `type=='Functional Formulation'`); `import_formulation_items(cx, item_rows, ff_product_ids: set) -> dict` (counts incl. `unresolved`); CLI dry-run/`--write`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_formulations.py
import sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from dashboard.formulations import init_formulations_schema
from scripts.import_formulations_from_fmp import import_formulations, import_formulation_items

def _db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx); init_formulations_schema(cx)
        cx.execute("INSERT INTO ingredients (fmp_id, name) VALUES ('r1','R-Lipoic Acid')")
        cx.commit()
    return p

def test_import_formulations_and_items(tmp_path):
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        nprod = import_formulations(cx, [
            {"id_pk":"f1","product_name":"Nerve Pulse","type":"Functional Formulation","active":"Yes"},
            {"id_pk":"x9","product_name":"Not A Formula","type":"Product","active":"Yes"},  # skipped
        ])
        res = import_formulation_items(cx, [
            {"id_pk":"it1","id_fk_product":"f1","id_fk_raw":"r1","zc_raw_display":"100mg - R-Lipoic Acid","zc_mg":"100","qty":"1","unit_measurement":"ea."},
            {"id_pk":"it2","id_fk_product":"f1","id_fk_raw":"UNKNOWN","zc_raw_display":"50mg - Mystery","zc_mg":"50"},
        ], ff_product_ids={"f1"})
        cx.commit()
        forms = cx.execute("SELECT * FROM formulations").fetchall()
        items = {r["fmp_id"]: dict(r) for r in cx.execute("SELECT * FROM formulation_items")}
    assert nprod == 1 and forms[0]["name"] == "Nerve Pulse"
    assert items["it1"]["ingredient_id"] is not None and items["it1"]["dose"] == 100.0 and items["it1"]["dose_unit"] == "mg"
    assert items["it1"]["ingredient_name"] == "R-Lipoic Acid"     # from zc_raw_display after " - "
    assert items["it2"]["ingredient_id"] is None                 # unresolved id_fk_raw kept, not dropped
    assert res["unresolved"] == 1

def test_reimport_preserves_curated(tmp_path):
    p = _db(tmp_path)
    from dashboard.formulations import update_formulation_curated
    with sqlite3.connect(p) as cx:
        import_formulations(cx, [{"id_pk":"f1","product_name":"Nerve Pulse","type":"Functional Formulation","active":"Yes"}]); cx.commit()
        fid = cx.execute("SELECT id FROM formulations WHERE fmp_id='f1'").fetchone()[0]
    update_formulation_curated(fid, {"notes":"keep me"}, db_path=p)
    with sqlite3.connect(p) as cx:
        import_formulations(cx, [{"id_pk":"f1","product_name":"Nerve Pulse RENAMED","type":"Functional Formulation","active":"No"}]); cx.commit()
        r = cx.execute("SELECT * FROM formulations WHERE fmp_id='f1'").fetchone()
    assert r[2] == "Nerve Pulse RENAMED"  # name (FMP) refreshed; col order: id,fmp_id,name
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        r = cx.execute("SELECT * FROM formulations WHERE fmp_id='f1'").fetchone()
    assert r["notes"] == "keep me"        # curated preserved
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_import_formulations.py -q`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement `scripts/import_formulations_from_fmp.py`**

```python
"""Import FileMaker formulation recipes into chat_log.db: formulations (FF products)
+ formulation_items (recipe lines referencing Phase-1 ingredients). Idempotent by
fmp_id; curated-preserving. Dry-run default; --write."""
from __future__ import annotations
import argparse, csv, os, re, sqlite3, sys
csv.field_size_limit(sys.maxsize)

from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert

EXPORT_DIR = os.environ.get("FMP_EXPORT_DIR", "/tmp/fmp-export/newapp")


def _name_after_dash(raw):
    # zc_raw_display like "100mg - R-Lipoic Acid" -> "R-Lipoic Acid"; "" if no name
    s = raw or ""
    return _clean(s.split(" - ", 1)[1]) if " - " in s else ""


def import_formulations(cx, product_rows):
    n = 0
    fmp_cols = ["name", "status", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "product_name", "type", "active",
                              "product_slug", "notes"}
    for r in product_rows:
        if (r.get("type") or "").strip() != "Functional Formulation":
            continue
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        name = _clean(r.get("product_name")) or f"(unnamed FMP formulation {fid})"
        status = "active" if _active(r.get("active")) == 1 else "inactive"
        _upsert(cx, "formulations", fmp_cols, [fid, name, status, _extras(r, mapped)], fmp_cols)
        n += 1
    return n


def import_formulation_items(cx, item_rows, ff_product_ids):
    ing = {r[1]: r[0] for r in cx.execute("SELECT id, fmp_id FROM ingredients WHERE fmp_id IS NOT NULL")}
    form = {r[1]: r[0] for r in cx.execute("SELECT id, fmp_id FROM formulations WHERE fmp_id IS NOT NULL")}
    n, unresolved = 0, 0
    fmp_cols = ["formulation_id", "ingredient_id", "ingredient_name", "dose", "dose_unit", "raw_text", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_product", "id_fk_raw", "id_fk_material",
                              "qty", "unit_measurement", "zc_mg", "zc_raw_display", "notes"}
    for r in item_rows:
        pid = (r.get("id_fk_product") or "").strip()
        if pid not in ff_product_ids:
            continue
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        form_id = form.get(pid)
        raw_fk = (r.get("id_fk_raw") or "").strip()
        ing_id = ing.get(raw_fk)
        if raw_fk and ing_id is None:
            unresolved += 1
        mg = _num(r.get("zc_mg"))
        if mg and mg > 0:
            dose, unit = mg, "mg"
        else:
            dose, unit = _num(r.get("qty")), (_clean(r.get("unit_measurement")) or None)
        name = _name_after_dash(r.get("zc_raw_display")) or None
        vals = [fid, form_id, ing_id, name, dose, unit, _clean(r.get("zc_raw_display")) or None, _extras(r, mapped)]
        _upsert(cx, "formulation_items", fmp_cols, vals, fmp_cols)
        n += 1
    return {"items": n, "unresolved": unresolved}


def _read(name):
    with open(os.path.join(EXPORT_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    products = _read("products.csv")
    items = _read("products_items.csv")
    ff_ids = {(p.get("id_pk") or "").strip() for p in products if (p.get("type") or "").strip() == "Functional Formulation"}
    print(f"FF formulations={len(ff_ids)} products_items={len(items)}")
    if not args.write:
        print("(dry run — pass --write to import)")
        return 0
    from dashboard.ingredient_catalog import _default_db_path, init_ingredients_schema
    from dashboard.formulations import init_formulations_schema
    cx = sqlite3.connect(args.db or _default_db_path()); cx.row_factory = sqlite3.Row
    init_ingredients_schema(cx); init_formulations_schema(cx)
    nf = import_formulations(cx, products)
    res = import_formulation_items(cx, items, ff_ids)
    cx.commit(); cx.close()
    print(f"wrote formulations={nf} items={res['items']} unresolved_ingredient_links={res['unresolved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests + real dry-run/--write into a temp DB**

Run: `python3 -m pytest tests/test_import_formulations.py -q` → PASS
Run (real data; the ingredient catalog must be present, so import Phase-1 first into the same temp db):
```bash
FMP_CANONICAL_CSV="$HOME/AI-Training/02 Skills/fmp-loaders/mapping/canonical_clusters.csv" python3 scripts/import_ingredients_from_fmp.py --write --db /tmp/ph2.db
python3 scripts/import_formulations_from_fmp.py --write --db /tmp/ph2.db
```
Expected: `wrote formulations=181 items=1684 unresolved_ingredient_links=<~410>`. Capture in report.

- [ ] **Step 5: Commit**

```bash
git add scripts/import_formulations_from_fmp.py tests/test_import_formulations.py
git commit -m "feat(formulations): FMP recipe importer (FF products + items, ingredient-linked)"
```

---

### Task 3: Product ↔ FMP matcher (write `fmp_id` onto products.json)

**Files:**
- Create: `scripts/match_products_to_fmp.py`
- Test: `tests/test_match_products.py`

**Interfaces:**
- Consumes `from scripts.populate_bottle_types import _norm, _build_fmp_index`; `dashboard.products.load_products`.
- Produces: `match_products(products: dict, fmp_by_name: dict) -> {"matched": {slug: fmp_id}, "review": [...]}` (exact `_norm` → suffix-strip → difflib ≥0.92; never overwrite an existing `fmp_id`; unmatched → review). CLI dry-run/`--write` (patches `products.json` adding `fmp_id`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_match_products.py
from scripts.match_products_to_fmp import match_products

def test_match_exact_and_fuzzy_and_review():
    products = {
        "nerve-pulse": {"name": "Nerve Pulse"},
        "msm-syntropy": {"name": "MSM Synergy"},   # alias → FMP "MSM Syntropy"
        "mystery": {"name": "Zzz Unknown Tonic"},
        "already": {"name": "Foo", "fmp_id": "999"},
    }
    fmp_by_name = {  # built by _build_fmp_index in real use; here pre-normalized keys
        "nerve pulse": {"id_pk": "1104"},
        "msm synergy": {"id_pk": "606"},           # _norm maps syntropy->synergy on both sides
    }
    m = match_products(products, fmp_by_name)
    assert m["matched"]["nerve-pulse"] == "1104"
    assert m["matched"]["msm-syntropy"] == "606"
    assert "already" not in m["matched"]            # never overwrite existing fmp_id
    assert any(r["slug"] == "mystery" for r in m["review"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_match_products.py -q`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement `scripts/match_products_to_fmp.py`**

```python
"""Match storefront products (products.json) to FMP products by name, writing a
stable fmp_id onto each matched product. Reuses the bottle-type matcher. Idempotent
(never overwrites an existing fmp_id). Dry-run default; --write."""
from __future__ import annotations
import argparse, csv, difflib, json, os, sys
csv.field_size_limit(sys.maxsize)

from scripts.populate_bottle_types import _norm, _build_fmp_index, _SUFFIX_WORDS

FMP_PRODUCTS = os.environ.get("FMP_PRODUCTS_CSV", "/tmp/fmp-export/newapp/products.csv")


def match_products(products, fmp_by_name):
    keys = list(fmp_by_name.keys())
    matched, review = {}, []
    for slug, p in products.items():
        if p.get("fmp_id"):
            continue
        nm = _norm(p.get("name"))
        row = fmp_by_name.get(nm)
        if row is None:
            stripped = _SUFFIX_WORDS.sub("", nm).strip()
            row = fmp_by_name.get(stripped)
        if row is None:
            close = difflib.get_close_matches(nm, keys, n=1, cutoff=0.92)
            if close:
                row = fmp_by_name[close[0]]
        fid = (row or {}).get("id_pk")
        if fid:
            matched[slug] = str(fid).strip()
        else:
            review.append({"slug": slug, "name": p.get("name", "")})
    return {"matched": matched, "review": review}


def _products_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    path = _products_path()
    with open(path) as f:
        doc = json.load(f)
    products = doc.get("products", {})
    with open(FMP_PRODUCTS, newline="") as f:
        fmp_by_name = _build_fmp_index(csv.DictReader(f))
    m = match_products(products, fmp_by_name)
    print(f"{len(m['matched'])} matched; {len(m['review'])} need review (of {len(products)})")
    for r in m["review"][:40]:
        print(f"  REVIEW {r['slug']}: {r['name']!r}")
    if args.write:
        for slug, fid in m["matched"].items():
            products[slug]["fmp_id"] = fid
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"wrote fmp_id to {len(m['matched'])} products")
    else:
        print("(dry run — pass --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests + dry-run over real catalog**

Run: `python3 -m pytest tests/test_match_products.py -q` → PASS
Run: `python3 scripts/match_products_to_fmp.py` → prints matched/review counts; no file change. Capture counts in report (do NOT `--write` — operator step).

- [ ] **Step 5: Commit**

```bash
git add scripts/match_products_to_fmp.py tests/test_match_products.py
git commit -m "feat(formulations): product->FMP matcher (writes stable fmp_id to products.json)"
```

---

### Task 4: Panel generator (DB → products.json) + retire the refresher

**Files:**
- Create: `scripts/generate_panels_from_db.py`
- Delete: `scripts/refresh_ingredients_from_fmp.py`, `tests/test_refresh_ingredients_from_fmp.py`
- Test: `tests/test_generate_panels.py`

**Interfaces:**
- Consumes: `dashboard.formulations` reads; `dashboard.ingredient_catalog._default_db_path`; `products.json` `fmp_id` (Task 3).
- Produces: `build_panel(items) -> (ingredients_list | None, reason)` — `[{name,dose}]` from formulation_items (name = `ingredient_canonical` or `ingredient_name`; dose = `"<dose> <unit>"`); returns `(None, reason)` if INCOMPLETE (any item dosed-but-unnamed → review, don't overwrite). `build_assignments(formulations_with_items, fmp_to_slug) -> {"panels": {slug: ingredients}, "review": [...]}`. CLI dry-run/`--write` (patch products.json, set `ingredients_source="db-formulations-<date>"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_panels.py
from scripts.generate_panels_from_db import build_panel

def test_build_panel_complete_and_incomplete():
    ok, reason = build_panel([
        {"ingredient_canonical": "R-Lipoic Acid", "ingredient_name": None, "dose": 100, "dose_unit": "mg"},
        {"ingredient_canonical": None, "ingredient_name": "Benfotiamine", "dose": 100, "dose_unit": "mg"},
    ])
    assert reason is None
    assert {"name": "R-Lipoic Acid", "dose": "100 mg"} in ok
    assert {"name": "Benfotiamine", "dose": "100 mg"} in ok
    bad, reason2 = build_panel([
        {"ingredient_canonical": None, "ingredient_name": None, "dose": 400, "dose_unit": "mg"},  # dosed, unnamed
    ])
    assert bad is None and "incomplete" in reason2.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_generate_panels.py -q`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement `scripts/generate_panels_from_db.py`**

```python
"""Generate customer-facing products.json ingredient panels FROM the DB formulation
recipes (single source of truth). One-way DB -> products.json. Completeness guard:
a recipe with a dosed-but-unnamed line is held for review, never overwrites a panel.
Dry-run default; --write. Supersedes scripts/refresh_ingredients_from_fmp.py."""
from __future__ import annotations
import argparse, json, os, sqlite3, sys


def _dose_str(dose, unit):
    if dose is None:
        return ""
    d = int(dose) if float(dose) == int(dose) else dose
    return f"{d} {unit}".strip() if unit else f"{d}"


def build_panel(items):
    out = []
    for it in items:
        name = (it.get("ingredient_canonical") or it.get("ingredient_name") or "").strip()
        has_dose = it.get("dose") is not None
        if not name:
            if has_dose:
                return None, "incomplete: dosed line with no ingredient name"
            continue  # nameless, doseless (packaging) — skip
        out.append({"name": name, "dose": _dose_str(it.get("dose"), it.get("dose_unit"))})
    if not out:
        return None, "no named ingredients"
    return out, None


def build_assignments(formulations_with_items, fmp_to_slug):
    panels, review = {}, []
    for f in formulations_with_items:
        slug = fmp_to_slug.get(str(f["fmp_id"]).strip())
        if not slug:
            review.append({"fmp_id": f["fmp_id"], "name": f["name"], "reason": "no matched product"})
            continue
        panel, reason = build_panel(f["items"])
        if panel is None:
            review.append({"slug": slug, "name": f["name"], "reason": reason})
        else:
            panels[slug] = panel
    return {"panels": panels, "review": review}


def _products_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, "products.json")):
        return os.path.join(d, "products.json")
    return os.path.join(here, "data", "products.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--db", default=None)
    ap.add_argument("--source-date", default="2026-06-24")  # stamp; pass run date
    args = ap.parse_args(argv)
    from dashboard.ingredient_catalog import _default_db_path
    from dashboard.formulations import search_formulations, list_items_for_formulation
    path = _products_path()
    with open(path) as f:
        doc = json.load(f)
    products = doc.get("products", {})
    fmp_to_slug = {str(p["fmp_id"]).strip(): slug for slug, p in products.items() if p.get("fmp_id")}
    db = args.db or _default_db_path()
    forms = search_formulations("", limit=100000, db_path=db)
    fwi = [{"fmp_id": f["fmp_id"], "name": f["name"],
            "items": list_items_for_formulation(f["id"], db_path=db)} for f in forms]
    m = build_assignments(fwi, fmp_to_slug)
    print(f"{len(m['panels'])} panels to write; {len(m['review'])} in review")
    for r in m["review"][:40]:
        print(f"  REVIEW {r.get('slug') or r.get('fmp_id')}: {r['name']!r} ({r['reason']})")
    if args.write:
        for slug, panel in m["panels"].items():
            products[slug]["ingredients"] = panel
            products[slug]["ingredients_source"] = f"db-formulations-{args.source_date}"
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"wrote {len(m['panels'])} panels")
    else:
        print("(dry run — pass --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Retire the refresher** — confirm nothing imports it, then delete:

```bash
grep -rn "refresh_ingredients_from_fmp" --include="*.py" . | grep -v "tests/test_refresh_ingredients_from_fmp"
git rm scripts/refresh_ingredients_from_fmp.py tests/test_refresh_ingredients_from_fmp.py
```
(If the grep shows a non-test importer, STOP and report — do not delete.)

- [ ] **Step 5: Run tests + commit**

Run: `python3 -m pytest tests/test_generate_panels.py -q` → PASS
```bash
git add scripts/generate_panels_from_db.py tests/test_generate_panels.py
git commit -m "feat(formulations): DB->products.json panel generator; retire FMP refresher"
```

---

### Task 5: Formulation console endpoints

**Files:**
- Modify: `app.py` (add `/api/formulations/*` beside `/api/ingredients/*`)
- Test: `tests/test_admin_formulations_api.py`

**Interfaces — Consumes Task 1 reads/writes.**

- [ ] **Step 1: Write the failing test** (route-level, Pinecone-skip pattern like `tests/test_admin_ingredients_api.py`)

```python
# tests/test_admin_formulations_api.py
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
        cx.execute("INSERT INTO formulations (fmp_id,name) VALUES ('f1','Nerve Pulse')")
        cx.commit()
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
    r = c.get("/api/formulations/search?q=nerve").get_json()
    fid = r["data"][0]["id"]
    d = c.get(f"/api/formulations/{fid}").get_json()["data"]
    assert d["formulation"]["name"] == "Nerve Pulse" and "items" in d
    assert c.patch(f"/api/formulations/{fid}", json={"notes":"x"}).status_code == 200
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_admin_formulations_api.py -q`
Expected: FAIL (404) or SKIP (Pinecone).

- [ ] **Step 3: Implement endpoints** (in `app.py`, beside `/api/ingredients/*`)

```python
from dashboard import formulations as _formulations

@app.route("/api/formulations/search", methods=["GET"])
@require_console_key
def api_formulations_search():
    try:
        return ok(_formulations.search_formulations(
            request.args.get("q",""), int(request.args.get("limit",50)), int(request.args.get("offset",0))))
    except Exception as e: return fail(e)

@app.route("/api/formulations/<int:fid>", methods=["GET"])
@require_console_key
def api_formulations_get(fid):
    try:
        f = _formulations.get_formulation(fid)
        if not f: return fail("not found", status=404)
        return ok({"formulation": f, "items": _formulations.list_items_for_formulation(fid)})
    except Exception as e: return fail(e)

@app.route("/api/formulations/<int:fid>", methods=["PATCH"])
@require_console_key
def api_formulations_patch(fid):
    try:
        _formulations.update_formulation_curated(fid, request.get_json(silent=True) or {})
        return ok(_formulations.get_formulation(fid))
    except Exception as e: return fail(e)

@app.route("/api/formulations/items/<int:item_id>", methods=["PATCH"])
@require_console_key
def api_formulations_item_patch(item_id):
    try:
        _formulations.update_item_curated(item_id, request.get_json(silent=True) or {})
        return ok({"id": item_id})
    except Exception as e: return fail(e)
```

- [ ] **Step 4: Run tests + commit**

Run: `python3 -m pytest tests/test_admin_formulations_api.py -q` → PASS or SKIP
```bash
git add app.py tests/test_admin_formulations_api.py
git commit -m "feat(formulations): console API endpoints"
```

---

### Task 6: Formulations console tab

**Files:**
- Modify: `static/admin-ingredients.html` (add a "Formulations" tab)

**Interfaces — Consumes Task 5 endpoints.** Mirror the existing tabs' markup/JS (search → list → detail; the `api()` helper, `?key=` forwarding, `escapeHtml`, `showTab`).

- [ ] **Step 1: Read `static/admin-ingredients.html`** to reuse its tab system (`labels` array, `showTab`), `api()`, `escapeHtml`, `toast`.

- [ ] **Step 2: Build the tab** — add `"formulations"` to the `labels` array + a tab button + panel:
  - Debounced search `GET /api/formulations/search?q=` → clickable list.
  - On click `GET /api/formulations/<id>` → show name/status (read-only) + a recipe table (each item: `ingredient_canonical || ingredient_name`, `dose` + `dose_unit`, `preferred_price` if present) + editable curated `notes` per item (`PATCH /api/formulations/items/<id>`) and per formulation (`PATCH /api/formulations/<id>`).
  - Use `escapeHtml` on all server strings; FMP fields read-only; only `notes` editable.

- [ ] **Step 3: Verify** — `python3 -c "import html.parser,pathlib; html.parser.HTMLParser().feed(pathlib.Path('static/admin-ingredients.html').read_text()); print('ok')"`; confirm fetch URLs/methods match Task 5; element ids exist.

- [ ] **Step 4: Commit**

```bash
git add static/admin-ingredients.html
git commit -m "feat(formulations): console formulations tab"
```

---

## Self-Review
**Spec coverage:** formulations+items schema (T1); importer with ingredient link + unresolved handling (T2); products.json fmp_id matcher = Decision A (T3); panel generator + retire refresher = Decision B (T4); console endpoints (T5) + tab (T6). ✓
**Placeholder scan:** complete code T1-T5; T6 is HTML mirroring existing tabs with explicit endpoint contracts. ✓
**Type consistency:** `_connect`/`db_path`; reused `_active/_num/_clean/_extras/_upsert` (exact names from Phase-1 script); `_norm/_build_fmp_index/_SUFFIX_WORDS` from populate_bottle_types; `build_panel`/`build_assignments`/`match_products`/`import_formulations`/`import_formulation_items` signatures consistent across tasks; endpoint→module fn names match. ✓
**Note on `_SUFFIX_WORDS` reuse (T3):** confirmed it exists in `scripts/populate_bottle_types.py`. If the import fails, fall back to defining the same regex locally.

## Operational note (post-merge, Glen)
Activation order: (1) ensure Phase-1 ingredients imported (the formulations console import already does this); (2) run formulations importer; (3) run `match_products_to_fmp.py --write` (review the unmatched), commit; (4) run `generate_panels_from_db.py --write` (review), commit. A server-side import endpoint for formulations (like the Phase-1 one) can be added if needed — deferred.
