# Phase 3c-2 Production + Consumption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Production runs consume ingredients — import FMP production history + log new runs, each posting negative `consumption` entries to the 3c-1 inventory ledger, completing on_hand = baseline + receipts − consumption ± recounts.

**Architecture:** Two new tables (`production_runs`, `production_run_items`) mirroring FMP `production`/`production_items`. A `dashboard/production.py` module (reads + curated writes + `log_run` + `post_consumption`). Consumption posts into the EXISTING `inventory_txns` ledger via INSERT-OR-IGNORE on `source_ref='prod_item:<id>'` (the same partial-unique idempotency mechanism as 3c-1). A console Production tab + `/api/production/*` endpoints + a server-side import. Consumption posting is mode-controlled (`all` | `from_date` | `record_only`) so historical-vs-going-forward is an activation choice, not hard-coded.

**Tech Stack:** Python 3 / Flask, SQLite (`chat_log.db`), vanilla-JS static console. Tests: pytest.

## Global Constraints

- Authoritative DB `chat_log.db`; module fns take optional `db_path`, use `with _connect(db_path)`. Reuse `_connect`/`_default_db_path` from `dashboard/inventory.py`/`purchase_orders.py`.
- Idempotency everywhere: `fmp_id` partial-unique for table re-import (INSERT-OR-IGNORE + UPDATE of fmp-cols only — NEVER `ON CONFLICT` on a partial index); consumption ledger rows keyed on `source_ref='prod_item:'||production_run_items.id` against the existing `idx_invtxn_source` partial-unique index. Counts use `cur.rowcount` (real inserts), never `len(candidates)`.
- Consumption sign is NEGATIVE: `qty = -abs(qty_used)`. Only `item_type='ingredient'` rows with non-null `ingredient_id` and non-zero `qty_used` post; material lines are recorded but DO NOT post to the ledger.
- Curated write surface is `notes` ONLY (on runs and items). Importer refreshes FMP cols only, never `notes`.
- Reuse FMP helpers: `from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert`. Set `cx.row_factory = sqlite3.Row` in importer/seed functions (Phase-3b lesson).
- Endpoints: `@require_console_key`, `ok`/`fail` from `dashboard/__init__.py`, alias `from dashboard import production as _prod`. `ValueError`→`fail(str(e), status=400)`, caught BEFORE the generic `except Exception`.
- Console mirrors the existing PO/Inventory tabs in `static/admin-ingredients.html`. The page's helper is `api(path, opts={})` — it RETURNS `j.data` directly and THROWS on error (no `.data`/`.ok` on the result). Multipart imports use raw `fetch` + `FormData` with `headers:{"X-Console-Key":KEY}` (NOT `api()`). **Add `display:none` initial-state CSS for any new detail panel + empty div.** Append to the `labels` array; don't touch existing tabs.
- `import_production_items` returns a DICT `{"items": n}`; `import_production_runs` and `post_consumption` return ints. Unwrap `["items"]` at call sites.
- Route tests use the Pinecone `pytest.skip` pattern (mirror `tests/test_admin_inventory_api.py`).
- Module name `dashboard/production.py` confirmed collision-free.

---

### Task 1: `dashboard/production.py` — schema, reads, `log_run`, `post_consumption`

**Files:**
- Create: `dashboard/production.py`
- Test: `tests/test_production.py`

**Interfaces:**
- Produces: `init_production_schema(cx)`; `search_production_runs(q="",limit=50,offset=0,db_path=None)`; `get_production_run(run_id,db_path=None)`; `list_run_items(run_id,db_path=None)`; `update_run_curated(run_id,fields,db_path=None)`; `update_run_item_curated(item_id,fields,db_path=None)`; `post_consumption(cx,run_id=None,mode="all",cutoff_date=None)->int`; `log_run(formulation_id,run_date,quantity_units,items,batch_number=None,db_path=None)->int`; `recipe_prefill(formulation_id,db_path=None)->list`.
- Consumes: 3c-1 `inventory_txns` (writes `consumption` rows); `formulations`/`formulation_items` (read for prefill); `ingredients`/`materials` (FK + name resolution).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_production.py
import sqlite3
import pytest
from dashboard import production as prod
from dashboard import inventory as inv


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, name TEXT, extras TEXT)")
        cx.execute("CREATE TABLE materials (id INTEGER PRIMARY KEY, name TEXT)")
        cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
        cx.execute("""CREATE TABLE formulation_items (id INTEGER PRIMARY KEY, fmp_id TEXT,
            formulation_id INTEGER, ingredient_id INTEGER, ingredient_name TEXT,
            dose REAL, dose_unit TEXT, raw_text TEXT, extras TEXT, notes TEXT)""")
        inv.init_inventory_schema(cx)
        prod.init_production_schema(cx)
        cx.execute("INSERT INTO ingredients VALUES (1,'Mag',NULL)")
        cx.execute("INSERT INTO ingredients VALUES (2,'Lipoic',NULL)")
        cx.execute("INSERT INTO materials VALUES (5,'Capsule')")
        cx.execute("INSERT INTO formulations VALUES (1,'f1','Brain Blend')")
        cx.execute("INSERT INTO formulation_items (id,formulation_id,ingredient_id,ingredient_name,dose,dose_unit) VALUES (1,1,1,'Mag',2.0,'kg')")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',10.0)")
        cx.commit()
    return db


def test_log_run_posts_negative_consumption(tmp_path):
    db = _db(tmp_path)
    rid = prod.log_run(1, "2026-02-01", 100, [{"ingredient_id": 1, "qty_used": 3.0, "unit": "kg"}],
                       batch_number="B1", db_path=db)
    assert isinstance(rid, int) and rid > 0
    assert inv.on_hand(1, db) == 7.0                 # 10 baseline − 3 consumed
    items = prod.list_run_items(rid, db)
    assert items[0]["posted"] == 1 and items[0]["qty_used"] == 3.0
    with pytest.raises(ValueError):
        prod.log_run(999, "2026-02-01", 1, [{"ingredient_id": 1, "qty_used": 1}], db_path=db)
    with pytest.raises(ValueError):
        prod.log_run(1, "2026-02-01", 1, [], db_path=db)   # no items


def test_post_consumption_idempotent(tmp_path):
    db = _db(tmp_path)
    rid = prod.log_run(1, "2026-02-01", 100, [{"ingredient_id": 1, "qty_used": 3.0, "unit": "kg"}], db_path=db)
    inv.add_adjustment(1, -1.0, db_path=db)          # manual recount after the run
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        again = prod.post_consumption(cx, run_id=rid, mode="all")
        cx.commit()
    assert again == 0                                 # already posted → no double
    assert inv.on_hand(1, db) == 6.0                  # 7.0 − 1.0 recount, unchanged by re-post


def test_mode_record_only_and_from_date(tmp_path):
    db = _db(tmp_path)
    # two manual runs on different dates, but post nothing at creation by inserting rows directly
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        cx.execute("INSERT INTO production_runs (id,formulation_id,run_date,quantity_units,source_kind) VALUES (10,1,'2025-01-01',50,'manual')")
        cx.execute("INSERT INTO production_runs (id,formulation_id,run_date,quantity_units,source_kind) VALUES (11,1,'2026-05-01',50,'manual')")
        cx.execute("INSERT INTO production_run_items (id,production_run_id,item_type,ingredient_id,qty_used,unit) VALUES (100,10,'ingredient',1,2.0,'kg')")
        cx.execute("INSERT INTO production_run_items (id,production_run_id,item_type,ingredient_id,qty_used,unit) VALUES (101,11,'ingredient',1,4.0,'kg')")
        cx.execute("INSERT INTO production_run_items (id,production_run_id,item_type,material_id,qty_used,unit) VALUES (102,11,'material',5,9.0,'ea')")
        cx.commit()
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        assert prod.post_consumption(cx, mode="record_only") == 0
        cx.commit()
    assert inv.on_hand(1, db) == 10.0                 # nothing posted
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        n = prod.post_consumption(cx, mode="from_date", cutoff_date="2026-01-01")
        cx.commit()
    assert n == 1                                     # only run 11's ingredient line (run 10 pre-cutoff; material skipped)
    assert inv.on_hand(1, db) == 6.0                  # 10 − 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_production.py -q`
Expected: FAIL (`ModuleNotFoundError`/`AttributeError`).

- [ ] **Step 3: Write `dashboard/production.py`**

```python
"""Production runs + consumption posting (Phase 3c-2)."""
import os
import sqlite3
from pathlib import Path
from typing import Optional


def _default_db_path() -> str:
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    cx = sqlite3.connect(db_path or _default_db_path())
    cx.row_factory = sqlite3.Row
    cx.execute("PRAGMA foreign_keys=ON")
    return cx


def init_production_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS production_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT,
          formulation_id INTEGER REFERENCES formulations(id),
          batch_number TEXT,
          run_date TEXT,
          quantity_units REAL,
          status TEXT,
          source_kind TEXT,
          extras TEXT,
          notes TEXT,
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prodrun_fmp ON production_runs(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_prodrun_form ON production_runs(formulation_id)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS production_run_items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT,
          production_run_id INTEGER REFERENCES production_runs(id),
          item_type TEXT,
          ingredient_id INTEGER REFERENCES ingredients(id),
          material_id INTEGER REFERENCES materials(id),
          item_label TEXT,
          qty_used REAL,
          unit TEXT,
          extras TEXT,
          notes TEXT,
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prunitems_fmp ON production_run_items(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_prunitems_run ON production_run_items(production_run_id)")
    cx.commit()


def search_production_runs(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT pr.*, f.name AS formulation_name
            FROM production_runs pr LEFT JOIN formulations f ON f.id = pr.formulation_id
            WHERE pr.batch_number LIKE ? OR f.name LIKE ?
            ORDER BY pr.run_date DESC, pr.id DESC LIMIT ? OFFSET ?
        """, (f"%{q}%", f"%{q}%", int(limit), int(offset))).fetchall()
    return [dict(r) for r in rows]


def get_production_run(run_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("""
            SELECT pr.*, f.name AS formulation_name
            FROM production_runs pr LEFT JOIN formulations f ON f.id = pr.formulation_id
            WHERE pr.id=?
        """, (run_id,)).fetchone()
    return dict(r) if r else None


def list_run_items(run_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT it.*, ing.name AS ingredient_canonical, mat.name AS material_name,
              CASE WHEN EXISTS (SELECT 1 FROM inventory_txns t
                                WHERE t.source_ref = 'prod_item:' || it.id) THEN 1 ELSE 0 END AS posted
            FROM production_run_items it
            LEFT JOIN ingredients ing ON ing.id = it.ingredient_id
            LEFT JOIN materials mat ON mat.id = it.material_id
            WHERE it.production_run_id = ? ORDER BY it.id
        """, (run_id,)).fetchall()
    return [dict(r) for r in rows]


_RUN_CURATED = {"notes"}
_ITEM_CURATED = {"notes"}


def _update_allowed(table, row_id, fields, allowed, db_path):
    cols = {k: v for k, v in (fields or {}).items() if k in allowed}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*cols.values(), row_id))
        cx.commit()


def update_run_curated(run_id, fields, db_path=None):
    _update_allowed("production_runs", run_id, fields, _RUN_CURATED, db_path)


def update_run_item_curated(item_id, fields, db_path=None):
    _update_allowed("production_run_items", item_id, fields, _ITEM_CURATED, db_path)


def post_consumption(cx, run_id=None, mode="all", cutoff_date=None) -> int:
    """Post negative consumption ledger rows for run items. mode: all|from_date|record_only.
    Takes an OPEN connection; caller commits. Idempotent via source_ref='prod_item:<item.id>'."""
    cx.row_factory = sqlite3.Row
    if mode == "record_only":
        return 0
    where = ["it.item_type = 'ingredient'", "it.ingredient_id IS NOT NULL",
             "it.qty_used IS NOT NULL", "it.qty_used <> 0"]
    params = []
    if run_id is not None:
        where.append("it.production_run_id = ?")
        params.append(run_id)
    if mode == "from_date" and cutoff_date:
        where.append("pr.run_date >= ?")
        params.append(cutoff_date)
    sql = """
        SELECT it.id AS item_id, it.ingredient_id, it.qty_used, it.unit, pr.run_date
        FROM production_run_items it
        JOIN production_runs pr ON pr.id = it.production_run_id
        WHERE """ + " AND ".join(where)
    n = 0
    for r in cx.execute(sql, params).fetchall():
        cur = cx.execute("""
            INSERT OR IGNORE INTO inventory_txns
              (ingredient_id, txn_type, qty, unit, txn_date, source_kind, source_ref)
            VALUES (?, 'consumption', ?, ?, ?, 'production_run', ?)
        """, (r["ingredient_id"], -abs(float(r["qty_used"])), r["unit"], r["run_date"],
              f"prod_item:{r['item_id']}"))
        n += cur.rowcount
    return n


def log_run(formulation_id, run_date, quantity_units, items, batch_number=None, db_path=None) -> int:
    if not items:
        raise ValueError("a production run needs at least one item")
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM formulations WHERE id=?", (formulation_id,)).fetchone():
            raise ValueError(f"no formulation {formulation_id}")
        cur = cx.execute("""
            INSERT INTO production_runs (formulation_id, batch_number, run_date, quantity_units, status, source_kind)
            VALUES (?, ?, ?, ?, 'completed', 'manual')
        """, (formulation_id, batch_number, run_date, quantity_units))
        rid = int(cur.lastrowid)
        for it in items:
            ing_id = it.get("ingredient_id")
            label = None
            if ing_id is not None:
                row = cx.execute("SELECT name FROM ingredients WHERE id=?", (ing_id,)).fetchone()
                label = row["name"] if row else None
            cx.execute("""
                INSERT INTO production_run_items
                  (production_run_id, item_type, ingredient_id, item_label, qty_used, unit)
                VALUES (?, 'ingredient', ?, ?, ?, ?)
            """, (rid, ing_id, label, it.get("qty_used"), it.get("unit")))
        post_consumption(cx, run_id=rid, mode="all")
        cx.commit()
        return rid


def recipe_prefill(formulation_id, db_path=None):
    from dashboard.formulations import list_items_for_formulation
    out = []
    for fi in list_items_for_formulation(formulation_id, db_path=db_path):
        out.append({
            "ingredient_id": fi.get("ingredient_id"),
            "item_label": fi.get("ingredient_canonical") or fi.get("ingredient_name"),
            "qty_used": fi.get("dose"),
            "unit": fi.get("dose_unit"),
        })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_production.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/production.py tests/test_production.py
git commit -m "feat(production): runs/items schema + log_run + consumption posting"
```

---

### Task 2: FMP importer + CLI

**Files:**
- Create: `scripts/import_production_from_fmp.py`
- Test: `tests/test_import_production.py`

**Interfaces:**
- Produces: `import_production_runs(cx, rows) -> int`; `import_production_items(cx, rows) -> dict` (`{"items": n}`); CLI `main()`.
- Consumes: Task-1 `post_consumption`; the FMP helper set from `scripts.import_ingredients_from_fmp`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_production.py
import sqlite3
from dashboard import production as prod
from dashboard import inventory as inv
from scripts.import_production_from_fmp import import_production_runs, import_production_items


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT, extras TEXT)")
        cx.execute("CREATE TABLE materials (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
        cx.execute("CREATE TABLE formulations (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT)")
        inv.init_inventory_schema(cx)
        prod.init_production_schema(cx)
        cx.execute("INSERT INTO ingredients VALUES (1,'r1','Mag',NULL)")
        cx.execute("INSERT INTO materials VALUES (5,'m1','Capsule')")
        cx.execute("INSERT INTO formulations VALUES (1,'p1','Brain Blend')")
        cx.commit()
    return db


def test_import_runs_items_and_consume(tmp_path):
    db = _db(tmp_path)
    runs = [{"id_pk": "900", "id_fk_product": "p1", "production_date": "2026-03-01", "qty": "100", "label": "B7"}]
    items = [
        {"id_pk": "9000", "id_fk_production": "900", "id_fk_raw": "r1", "qty": "2.5", "unit_measurement": "kg"},
        {"id_pk": "9001", "id_fk_production": "900", "id_fk_material": "m1", "qty": "100", "unit_measurement": "ea"},
    ]
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        nr = import_production_runs(cx, runs)
        ni = import_production_items(cx, items)
        n_consumed = prod.post_consumption(cx, mode="all")
        cx.commit()
    assert nr == 1 and ni == {"items": 2}
    assert n_consumed == 1                            # only the ingredient line consumes; material does not
    assert inv.on_hand(1, db) == -2.5                 # no baseline here, just the consumption
    # run resolved to formulation, item to ingredient/material
    run = prod.search_production_runs(db_path=db)[0]
    assert run["formulation_name"] == "Brain Blend" and run["batch_number"] == "B7"
    its = prod.list_run_items(run["id"], db)
    kinds = sorted(i["item_type"] for i in its)
    assert kinds == ["ingredient", "material"]


def test_reimport_preserves_curated_and_idempotent(tmp_path):
    db = _db(tmp_path)
    runs = [{"id_pk": "900", "id_fk_product": "p1", "production_date": "2026-03-01", "qty": "100", "label": "B7"}]
    items = [{"id_pk": "9000", "id_fk_production": "900", "id_fk_raw": "r1", "qty": "2.5", "unit_measurement": "kg"}]
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        import_production_runs(cx, runs); import_production_items(cx, items)
        prod.post_consumption(cx, mode="all"); cx.commit()
    rid = prod.search_production_runs(db_path=db)[0]["id"]
    prod.update_run_curated(rid, {"notes": "keep me"}, db_path=db)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        import_production_runs(cx, runs); import_production_items(cx, items)
        n2 = prod.post_consumption(cx, mode="all"); cx.commit()
    assert n2 == 0                                    # consumption idempotent
    assert prod.get_production_run(rid, db)["notes"] == "keep me"   # curated preserved
    assert inv.on_hand(1, db) == -2.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_import_production.py -q`
Expected: FAIL (`ModuleNotFoundError: scripts.import_production_from_fmp`).

- [ ] **Step 3: Write `scripts/import_production_from_fmp.py`**

```python
#!/usr/bin/env python3
"""Import FMP production + production_items → production_runs/items, then post consumption."""
import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.import_ingredients_from_fmp import _num, _clean, _extras, _upsert  # noqa: E402
from dashboard import production as prod  # noqa: E402
from dashboard import inventory as inv  # noqa: E402

csv.field_size_limit(sys.maxsize)


def import_production_runs(cx, rows) -> int:
    cx.row_factory = sqlite3.Row
    fmap = {r["fmp_id"]: r["id"] for r in cx.execute(
        "SELECT id, fmp_id FROM formulations WHERE fmp_id IS NOT NULL")}
    fmp_cols = ["formulation_id", "batch_number", "run_date", "quantity_units", "status", "source_kind", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_product", "production_date", "qty", "label", "notes"}
    n = 0
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        form_id = fmap.get((r.get("id_fk_product") or "").strip())
        vals = [fid, form_id, _clean(r.get("label")) or None, _clean(r.get("production_date")) or None,
                _num(r.get("qty")), "completed", "fmp", _extras(r, mapped)]
        _upsert(cx, "production_runs", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def import_production_items(cx, rows) -> dict:
    cx.row_factory = sqlite3.Row
    runmap = {r["fmp_id"]: r["id"] for r in cx.execute(
        "SELECT id, fmp_id FROM production_runs WHERE fmp_id IS NOT NULL")}
    ingmap = {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute(
        "SELECT id, fmp_id, name FROM ingredients WHERE fmp_id IS NOT NULL")}
    matmap = {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute(
        "SELECT id, fmp_id, name FROM materials WHERE fmp_id IS NOT NULL")}
    fmp_cols = ["production_run_id", "item_type", "ingredient_id", "material_id", "item_label",
                "qty_used", "unit", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_production", "id_fk_raw", "id_fk_material",
                              "qty", "unit_measurement", "notes"}
    n = 0
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        run_id = runmap.get((r.get("id_fk_production") or "").strip())
        raw = (r.get("id_fk_raw") or "").strip()
        mat = (r.get("id_fk_material") or "").strip()
        if raw and raw in ingmap:
            kind, iid, mid, label = "ingredient", ingmap[raw][0], None, ingmap[raw][1]
        elif mat and mat in matmap:
            kind, iid, mid, label = "material", None, matmap[mat][0], matmap[mat][1]
        else:
            kind, iid, mid, label = ("ingredient" if raw else "material" if mat else None), None, None, None
        vals = [fid, run_id, kind, iid, mid, label, _num(r.get("qty")),
                _clean(r.get("unit_measurement")) or None, _extras(r, mapped)]
        _upsert(cx, "production_run_items", fmp_cols, vals, fmp_cols)
        n += 1
    return {"items": n}


def _read(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _db_path():
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_db_path())
    ap.add_argument("--dir", default="/tmp/fmp-export/newapp")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--consumption", choices=["all", "record_only"], default="all")
    ap.add_argument("--consumption-from", default=None, help="YYYY-MM-DD; only runs on/after post consumption")
    args = ap.parse_args()
    runs = _read(str(Path(args.dir) / "production.csv"))
    items = _read(str(Path(args.dir) / "production_items.csv"))
    cx = sqlite3.connect(args.db)
    cx.row_factory = sqlite3.Row
    try:
        inv.init_inventory_schema(cx)
        prod.init_production_schema(cx)
        nr = import_production_runs(cx, runs)
        ni = import_production_items(cx, items)
        mode = "from_date" if args.consumption_from else args.consumption
        nc = prod.post_consumption(cx, mode=mode, cutoff_date=args.consumption_from)
        if args.write:
            cx.commit()
            print(f"WROTE runs={nr} items={ni['items']} consumption={nc}")
        else:
            cx.rollback()
            print(f"DRY-RUN runs={nr} items={ni['items']} consumption={nc} (rolled back)")
    finally:
        cx.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_import_production.py tests/test_production.py -q`
Expected: PASS (Task-1 tests still green + 2 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/import_production_from_fmp.py tests/test_import_production.py
git commit -m "feat(production): FMP importer + consumption-mode CLI"
```

---

### Task 3: `/api/production/*` endpoints + server-side import

**Files:**
- Modify: `app.py`
- Test: `tests/test_admin_production_api.py`

**Interfaces:**
- Consumes: Task 1 + Task 2 via `from dashboard import production as _prod`.
- Produces: `GET /api/production/search`; `GET /api/production/<int:run_id>`; `GET /api/production/recipe/<int:formulation_id>`; `POST /api/production/log`; `PATCH /api/production/<int:run_id>`; `PATCH /api/production/items/<int:item_id>`; `POST /api/production/import`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_admin_production_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.formulations import init_formulations_schema
    from dashboard.inventory import init_inventory_schema
    from dashboard.production import init_production_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        init_formulations_schema(cx); init_inventory_schema(cx); init_production_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name) VALUES (1,'Mag')")
        cx.execute("INSERT INTO formulations (id,fmp_id,name) VALUES (1,'p1','Brain Blend')")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',10.0)")
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


def test_log_and_read(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/production/log", json={"formulation_id": 1, "run_date": "2026-03-01",
               "quantity_units": 100, "items": [{"ingredient_id": 1, "qty_used": 3.0, "unit": "kg"}]})
    assert r.status_code == 200
    rid = r.get_json()["data"]["id"]
    d = c.get(f"/api/production/{rid}").get_json()["data"]
    assert d["run"]["formulation_name"] == "Brain Blend" and len(d["items"]) == 1
    assert c.get("/api/production/search?q=Brain").get_json()["data"][0]["id"] == rid
    assert c.get("/api/production/999").status_code == 404
    bad = c.post("/api/production/log", json={"formulation_id": 1, "run_date": "x", "quantity_units": 1, "items": []})
    assert bad.status_code == 400
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_admin_production_api.py -q`
Expected: FAIL (404) or SKIP (Pinecone). Proceed to implement.

- [ ] **Step 3: Wire schema init + endpoints in `app.py`**

After `_init_inventory_tables()` (added in 3c-1), add:

```python
def _init_production_tables():
    from dashboard.production import init_production_schema
    cx = sqlite3.connect(str(LOG_DB))
    try:
        init_production_schema(cx)
    finally:
        cx.close()

_init_production_tables()
```

Beside the `/api/inventory/*` block, add:

```python
from dashboard import production as _prod


@app.route("/api/production/search", methods=["GET"])
@require_console_key
def api_production_search():
    try:
        return ok(_prod.search_production_runs(request.args.get("q",""),
                                               int(request.args.get("limit",100)),
                                               int(request.args.get("offset",0))))
    except Exception as e:
        return fail(e)


@app.route("/api/production/<int:run_id>", methods=["GET"])
@require_console_key
def api_production_get(run_id):
    try:
        r = _prod.get_production_run(run_id)
        if not r:
            return fail("not found", status=404)
        return ok({"run": r, "items": _prod.list_run_items(run_id)})
    except Exception as e:
        return fail(e)


@app.route("/api/production/recipe/<int:formulation_id>", methods=["GET"])
@require_console_key
def api_production_recipe(formulation_id):
    try:
        return ok(_prod.recipe_prefill(formulation_id))
    except Exception as e:
        return fail(e)


@app.route("/api/production/log", methods=["POST"])
@require_console_key
def api_production_log():
    try:
        b = request.get_json(silent=True) or {}
        rid = _prod.log_run(b.get("formulation_id"), b.get("run_date"), b.get("quantity_units"),
                            b.get("items") or [], b.get("batch_number"))
        return ok({"id": rid})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)


@app.route("/api/production/<int:run_id>", methods=["PATCH"])
@require_console_key
def api_production_patch(run_id):
    try:
        _prod.update_run_curated(run_id, request.get_json(silent=True) or {})
        return ok(_prod.get_production_run(run_id))
    except Exception as e:
        return fail(e)


@app.route("/api/production/items/<int:item_id>", methods=["PATCH"])
@require_console_key
def api_production_item_patch(item_id):
    try:
        _prod.update_run_item_curated(item_id, request.get_json(silent=True) or {})
        return ok({"id": item_id})
    except Exception as e:
        return fail(e)


@app.route("/api/production/import", methods=["POST"])
@require_console_key
def api_production_import():
    import csv as _csv, sys as _sys, io as _io
    _csv.field_size_limit(_sys.maxsize)
    try:
        from scripts.import_production_from_fmp import import_production_runs, import_production_items
        from dashboard.ingredient_catalog import init_ingredients_schema
        from dashboard.materials_catalog import init_materials_schema
        from dashboard.formulations import init_formulations_schema
        from dashboard.inventory import init_inventory_schema
        from dashboard.production import init_production_schema, post_consumption
    except Exception as e:
        return fail(f"import error: {e}")
    try:
        f_runs = request.files.get("production")
        f_items = request.files.get("production_items")
        if not all([f_runs, f_items]):
            return fail("upload both: production, production_items", status=400)
        runs = list(_csv.DictReader(_io.StringIO(f_runs.read().decode("utf-8", errors="replace"))))
        items = list(_csv.DictReader(_io.StringIO(f_items.read().decode("utf-8", errors="replace"))))
        write = request.form.get("write", "").lower() in ("1","true","yes")
        cons = request.form.get("consumption", "all")
        cons_from = request.form.get("consumption_from") or None
        cx = sqlite3.connect(str(LOG_DB))
        cx.row_factory = sqlite3.Row
        try:
            init_ingredients_schema(cx); init_materials_schema(cx); init_formulations_schema(cx)
            init_inventory_schema(cx); init_production_schema(cx)
            nr = import_production_runs(cx, runs)
            ni = import_production_items(cx, items)
            mode = "from_date" if cons_from else cons
            nc = post_consumption(cx, mode=mode, cutoff_date=cons_from)
            if write:
                cx.commit()
            else:
                cx.rollback()
        finally:
            cx.close()
        return ok({"mode": "write" if write else "dry_run", "runs": nr, "items": ni["items"], "consumption": nc})
    except Exception as e:
        app.logger.exception("production import error")
        return fail(str(e))
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_admin_production_api.py -q`
Expected: PASS or SKIP locally on Pinecone. Smoke: `python3 -c "import app"`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_admin_production_api.py
git commit -m "feat(production): console endpoints + server-side import"
```

---

### Task 4: Production console tab + Import section + search index

**Files:**
- Modify: `static/admin-ingredients.html`
- Modify: `static/console-search-index.json`

**Interfaces:**
- Consumes: `/api/production/*` endpoints from Task 3.

- [ ] **Step 1: Read the existing Inventory + PO tabs in `static/admin-ingredients.html`** — note the real `api(path, opts={})` (returns `j.data`, throws), the `escapeHtml`/`showTab` helpers, the `labels` array, the `display:none` CSS for `#inv-detail-panel`/`#po-detail-panel`, and the raw-`fetch`+`FormData` import pattern (`fmpImport`/`poImport`/`invSeed`). The Production tab mirrors these.

- [ ] **Step 2: Append `"production"` to the `labels` array** in `showTab`.

- [ ] **Step 3: Add the Production tab button + panel** (mirror the Inventory tab). Panel: a search box (`id="prod-search"`, oninput → `doProdSearch`), a run list (`id="prod-list"`) + empty div (`id="prod-empty"`), a detail panel (`id="prod-detail-panel"`) with a header (`id="prod-detail-head"`) + a line-items table (`id="prod-items"`, each row shows item_label/type, qty_used+unit, a "consumed ✓/—" badge from `posted`, editable `notes`), and a "Log a production run" form: a formulation `<select id="prod-log-form">` (populate via `GET /api/formulations/search` or the existing formulations loader — reuse whatever the Formulations tab uses; if none, a numeric formulation-id input is acceptable), a `run_date` input, a `quantity_units` input, an editable lines table (`id="prod-log-lines"`) with an "↻ Pull recipe" button (`GET /api/production/recipe/<formId>` → fill lines), and a submit (→ `POST /api/production/log`).

- [ ] **Step 4: Add CSS (mirror the inventory rules) — REQUIRED initial `display:none`:**

```css
#prod-empty { color: var(--muted); font-size: 13px; padding: 16px 0; display: none; }
#prod-detail-panel { display: none; }
#prod-detail-panel.active { display: block; }
```

- [ ] **Step 5: Add the JS (use the REAL `api()` — returns data, throws; reuse `escapeHtml`):**

```javascript
let prodCurrentId = null;
let prodLogLines = [];

async function doProdSearch() {
  const q = document.getElementById("prod-search").value.trim();
  const rows = await api("/api/production/search?q=" + encodeURIComponent(q));
  const el = document.getElementById("prod-list");
  if (!rows.length) { el.innerHTML = ""; document.getElementById("prod-empty").style.display = "block"; return; }
  document.getElementById("prod-empty").style.display = "none";
  el.innerHTML = rows.map(function (x) {
    return '<tr onclick="openProd(' + x.id + ')"><td>' + escapeHtml(x.batch_number || "—") +
      '</td><td>' + escapeHtml(x.formulation_name || "") + '</td><td>' + escapeHtml(x.run_date || "") +
      '</td><td>' + (x.quantity_units != null ? x.quantity_units : "") + '</td></tr>';
  }).join("");
}

async function openProd(id) {
  prodCurrentId = id;
  const d = await api("/api/production/" + id);
  document.getElementById("prod-detail-panel").classList.add("active");
  document.getElementById("prod-detail-head").innerHTML =
    "<strong>" + escapeHtml(d.run.formulation_name || "") + "</strong> — " +
    escapeHtml(d.run.batch_number || "") + " · " + escapeHtml(d.run.run_date || "") +
    " · qty " + (d.run.quantity_units != null ? d.run.quantity_units : "—");
  document.getElementById("prod-items").innerHTML = (d.items || []).map(function (it) {
    var name = escapeHtml(it.item_label || it.ingredient_canonical || it.material_name || "");
    var badge = it.posted ? '<span style="color:#080">consumed ✓</span>' : '<span style="color:var(--muted)">—</span>';
    return "<tr><td>" + name + "</td><td>" + escapeHtml(it.item_type || "") + "</td><td>" +
      (it.qty_used != null ? it.qty_used : "") + " " + escapeHtml(it.unit || "") + "</td><td>" + badge + "</td></tr>";
  }).join("");
}

async function prodPullRecipe() {
  const fid = parseInt(document.getElementById("prod-log-form").value, 10);
  if (!fid) { toast("Pick a formulation first", "error"); return; }
  prodLogLines = await api("/api/production/recipe/" + fid);
  renderProdLogLines();
}

function renderProdLogLines() {
  document.getElementById("prod-log-lines").innerHTML = prodLogLines.map(function (l, i) {
    return '<tr><td>' + escapeHtml(l.item_label || "") + '</td><td><input type="number" step="any" value="' +
      (l.qty_used != null ? l.qty_used : "") + '" onchange="prodLogLines[' + i + '].qty_used=parseFloat(this.value)"></td><td>' +
      escapeHtml(l.unit || "") + '</td></tr>';
  }).join("");
}

async function submitProdLog() {
  const fid = parseInt(document.getElementById("prod-log-form").value, 10);
  const body = {
    formulation_id: fid,
    run_date: document.getElementById("prod-log-date").value || null,
    quantity_units: parseFloat(document.getElementById("prod-log-qty").value) || null,
    items: prodLogLines.filter(function (l) { return l.ingredient_id && l.qty_used; }),
  };
  if (!fid || !body.items.length) { toast("Pick a formulation and pull/enter at least one line", "error"); return; }
  try {
    const r = await api("/api/production/log", { method: "POST", body: JSON.stringify(body) });
    toast("Run logged — consumption posted");
    prodLogLines = []; renderProdLogLines();
    openProd(r.id); doProdSearch();
  } catch (e) { toast("Log failed: " + e.message, "error"); }
}

async function prodImport(write) {
  const fRuns = document.getElementById("fmp-file-production").files[0];
  const fItems = document.getElementById("fmp-file-production-items").files[0];
  if (!fRuns || !fItems) { toast("Choose both production CSVs", "error"); return; }
  const fd = new FormData();
  fd.append("production", fRuns); fd.append("production_items", fItems);
  fd.append("consumption", document.getElementById("prod-consumption-mode").value);
  const cfrom = document.getElementById("prod-consumption-from").value;
  if (cfrom) fd.append("consumption_from", cfrom);
  if (write) fd.append("write", "1");
  const res = await fetch("/api/production/import", { method: "POST", headers: { "X-Console-Key": KEY }, body: fd });
  const j = await res.json();
  if (res.status === 401) { document.getElementById("auth-warn").style.display = "block"; return; }
  if (!j.ok) { toast(j.error || "Import failed", "error"); return; }
  toast((j.data.mode === "write" ? "Imported" : "Dry run") + " — runs=" + j.data.runs +
        " items=" + j.data.items + " consumption=" + j.data.consumption);
}
```

Wire the search `oninput`, the "Pull recipe" / submit buttons, the import buttons, and a consumption-mode `<select id="prod-consumption-mode">` (options `all` = "Post all", `record_only` = "Record only") + optional `<input id="prod-consumption-from" type="date">` in the Production import section. Reuse the Formulations tab's loader to populate `#prod-log-form` if one exists; otherwise a numeric formulation-id input is acceptable (note which you chose in the report). Verify HTML parses, ids exist, existing tabs untouched.

- [ ] **Step 6: Register in `static/console-search-index.json`**

Add: `{ "title": "Production Runs", "page": "Products", "url": "/admin/ingredients", "keywords": ["production","run","batch","made","consumption","manufacture","produced"] }`

- [ ] **Step 7: Commit**

```bash
git add static/admin-ingredients.html static/console-search-index.json
git commit -m "feat(production): admin Production tab + import section + search index"
```

---

## Self-Review

- **Spec coverage:** runs/items schema + log_run + post_consumption (T1); FMP importer + mode CLI (T2); endpoints + server-side import (T3); console tab + import section + search index (T4). Consumption modes all/from_date/record_only ✓. Material lines recorded but never post to ledger ✓. Idempotency by fmp_id (tables) + source_ref (ledger) ✓. Curated = notes only ✓.
- **Placeholders:** none — full code in every code step.
- **Type consistency:** `init_production_schema`/`search_production_runs`/`get_production_run`/`list_run_items`/`log_run`/`post_consumption`/`recipe_prefill`/`import_production_runs`/`import_production_items` used identically across module, importer, endpoints, tests. `_prod` alias. `post_consumption(cx, run_id=, mode=, cutoff_date=)` signature identical in T1 def, T2 CLI, T3 endpoint. `import_production_items` returns `{"items": n}` consistently; unwrapped `["items"]` in CLI + endpoint. `source_ref='prod_item:'||id` identical in `post_consumption` (write) and `list_run_items` (posted check).
- **Reviewer note:** the console `api()` calls use the REAL `api(path, opts)` shape (returns data, throws). The import call uses raw `fetch`+`FormData` (the seed/import pattern), NOT `api()`. The implementer must confirm the Formulations-tab loader for the run form's formulation picker (or fall back to a numeric id input) and report which.
