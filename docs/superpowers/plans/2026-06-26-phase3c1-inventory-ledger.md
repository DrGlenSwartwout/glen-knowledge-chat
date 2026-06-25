# Phase 3c-1 Inventory Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A persisted inventory ledger (`inventory_txns`) giving a live on-hand balance per ingredient, seeded from the Phase-1 baseline + Phase-3b receipts, with a console to view levels-vs-par and post manual recounts.

**Architecture:** One append-only ledger table; on-hand = signed `SUM(qty)`. A `dashboard/inventory.py` module (reads + one curated write). Machine-generated entries (baseline, receipt) carry a stable `source_ref` and seed via INSERT-OR-IGNORE on a partial unique index, so seeding is re-runnable. An "Inventory" tab in the existing `/admin/ingredients` console + `/api/inventory/*` endpoints, mirroring the Phase-3b Purchase-Orders feature exactly.

**Tech Stack:** Python 3 / Flask, SQLite (`chat_log.db`), vanilla-JS static console. Tests: pytest.

## Global Constraints

- Authoritative DB is `chat_log.db`; all module functions take an optional `db_path` and use `with _connect(db_path)`. Reuse `_connect`/`_default_db_path` shape from `dashboard/purchase_orders.py`.
- Idempotency: SQLite cannot use a partial unique index as an `ON CONFLICT` target → use **INSERT OR IGNORE** keyed on `source_ref` (partial unique `WHERE source_ref IS NOT NULL`).
- Seeding/read functions that read by column name must set `cx.row_factory = sqlite3.Row` defensively (Phase-3b lesson).
- All endpoints: `@require_console_key`, return via `ok()` / `fail()` from `dashboard/__init__.py`. Import alias `from dashboard import inventory as _inv`.
- Curated write surface is `notes` ONLY (on a txn); never let any other column through a PATCH.
- Console: mirror the existing Purchase-Orders tab in `static/admin-ingredients.html`. **Add `display:none` initial-state CSS for the new detail panel + empty div** (recurring gotcha — a prior task shipped a panel visible on load). Existing tabs/routes untouched. Reuse `api()`/`escapeHtml`/`showTab`.
- Route tests use the Pinecone `pytest.skip` pattern (skip locally, run in CI), mirroring `tests/test_admin_po_api.py`.
- Sign convention: `baseline`/`receipt` positive; `consumption` negative (3c-2, not written here); `adjustment` signed. On-hand is the plain sum.
- Module name `dashboard/inventory.py` confirmed collision-free.

---

### Task 1: `dashboard/inventory.py` — schema, reads, curated write

**Files:**
- Create: `dashboard/inventory.py`
- Test: `tests/test_inventory.py`

**Interfaces:**
- Produces: `init_inventory_schema(cx)`; `on_hand(ingredient_id, db_path=None) -> float`; `inventory_levels(q="", limit=50, offset=0, db_path=None) -> list[dict]`; `get_inventory(ingredient_id, db_path=None) -> dict|None`; `list_txns(ingredient_id, db_path=None) -> list[dict]`; `add_adjustment(ingredient_id, qty, unit=None, txn_date=None, notes=None, db_path=None) -> int`; `update_txn_curated(txn_id, fields, db_path=None)`. (`seed_baselines`/`seed_receipts` are added in Task 2.)
- Consumes: the Phase-1 `ingredients` table (`extras` JSON holds `inventory_starting`, `par_level`, `par_level_unit`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inventory.py
import json, sqlite3
import pytest
from dashboard import inventory as inv


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("""CREATE TABLE ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fmp_id TEXT, name TEXT, extras TEXT)""")
        inv.init_inventory_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name,extras) VALUES (1,'Mag L-threonate',?)",
                   (json.dumps({"inventory_starting": "1.0", "par_level": "3", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO ingredients (id,name,extras) VALUES (2,'R-Lipoic',?)",
                   (json.dumps({"par_level": "0.25", "par_level_unit": "kg"}),))
        cx.commit()
    return db


def test_on_hand_sums_signed(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',1.0)")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'receipt',5.0)")
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'consumption',-2.0)")
        cx.commit()
    assert inv.on_hand(1, db) == 4.0
    assert inv.on_hand(2, db) == 0.0


def test_levels_below_par(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',4.0)")
        cx.commit()
    rows = {r["id"]: r for r in inv.inventory_levels(db_path=db)}
    assert rows[1]["on_hand"] == 4.0 and rows[1]["below_par"] == 0      # 4 >= 3
    assert rows[2]["on_hand"] == 0.0 and rows[2]["below_par"] == 1      # 0 < 0.25
    assert rows[1]["par_level_unit"] == "kg"


def test_add_adjustment_shifts_on_hand(tmp_path):
    db = _db(tmp_path)
    tid = inv.add_adjustment(1, -0.5, unit="kg", notes="recount", db_path=db)
    assert isinstance(tid, int) and tid > 0
    assert inv.on_hand(1, db) == -0.5
    with pytest.raises(ValueError):
        inv.add_adjustment(999, 1.0, db_path=db)          # no such ingredient


def test_update_txn_curated_notes_only(tmp_path):
    db = _db(tmp_path)
    tid = inv.add_adjustment(1, 1.0, db_path=db)
    inv.update_txn_curated(tid, {"notes": "x", "qty": 999, "txn_type": "hack"}, db_path=db)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        r = cx.execute("SELECT * FROM inventory_txns WHERE id=?", (tid,)).fetchone()
    assert r["notes"] == "x" and r["qty"] == 1.0 and r["txn_type"] == "adjustment"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_inventory.py -q`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError: init_inventory_schema`).

- [ ] **Step 3: Write `dashboard/inventory.py`**

```python
"""Inventory ledger — persisted on-hand balance per ingredient (Phase 3c-1)."""
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


def init_inventory_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS inventory_txns (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ingredient_id INTEGER REFERENCES ingredients(id),
          txn_type TEXT NOT NULL,
          qty REAL NOT NULL,
          unit TEXT,
          txn_date TEXT,
          source_kind TEXT,
          source_ref TEXT,
          notes TEXT,
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_invtxn_source ON inventory_txns(source_ref) WHERE source_ref IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_invtxn_ing ON inventory_txns(ingredient_id)")
    cx.commit()


def on_hand(ingredient_id, db_path=None) -> float:
    with _connect(db_path) as cx:
        r = cx.execute("SELECT COALESCE(SUM(qty),0) AS oh FROM inventory_txns WHERE ingredient_id=?",
                       (ingredient_id,)).fetchone()
    return float(r["oh"] or 0)


def inventory_levels(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT i.id, i.name,
                   COALESCE((SELECT SUM(qty) FROM inventory_txns t WHERE t.ingredient_id=i.id),0) AS on_hand,
                   json_extract(i.extras,'$.par_level')      AS par_level,
                   json_extract(i.extras,'$.par_level_unit') AS par_level_unit
            FROM ingredients i
            WHERE i.name LIKE ?
            ORDER BY i.name
            LIMIT ? OFFSET ?
        """, (f"%{q}%", int(limit), int(offset))).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["on_hand"] = float(d["on_hand"] or 0)
        par = _to_num(d.get("par_level"))
        d["below_par"] = 1 if (par is not None and d["on_hand"] < par) else 0
        out.append(d)
    # below-par first, then name
    out.sort(key=lambda d: (-d["below_par"], (d["name"] or "").lower()))
    return out


def get_inventory(ingredient_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("SELECT * FROM ingredients WHERE id=?", (ingredient_id,)).fetchone()
        if not r:
            return None
        ing = dict(r)
        oh = cx.execute("SELECT COALESCE(SUM(qty),0) AS oh FROM inventory_txns WHERE ingredient_id=?",
                        (ingredient_id,)).fetchone()["oh"]
        txns = [dict(x) for x in cx.execute(
            "SELECT * FROM inventory_txns WHERE ingredient_id=? ORDER BY txn_date DESC, id DESC",
            (ingredient_id,)).fetchall()]
    par = _json_get(ing.get("extras"), "par_level")
    par_unit = _json_get(ing.get("extras"), "par_level_unit")
    return {"ingredient": ing, "on_hand": float(oh or 0),
            "par_level": par, "par_level_unit": par_unit, "txns": txns}


def list_txns(ingredient_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute(
            "SELECT * FROM inventory_txns WHERE ingredient_id=? ORDER BY txn_date DESC, id DESC",
            (ingredient_id,)).fetchall()
    return [dict(r) for r in rows]


def add_adjustment(ingredient_id, qty, unit=None, txn_date=None, notes=None, db_path=None) -> int:
    try:
        q = float(qty)
    except (TypeError, ValueError):
        raise ValueError("qty must be numeric")
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM ingredients WHERE id=?", (ingredient_id,)).fetchone():
            raise ValueError(f"no ingredient {ingredient_id}")
        cur = cx.execute("""
            INSERT INTO inventory_txns (ingredient_id, txn_type, qty, unit, txn_date, source_kind, notes)
            VALUES (?, 'adjustment', ?, ?, ?, 'manual', ?)
        """, (ingredient_id, q, unit, txn_date, notes))
        cx.commit()
        return int(cur.lastrowid)


_TXN_CURATED = {"notes"}


def update_txn_curated(txn_id, fields, db_path=None) -> None:
    cols = {k: v for k, v in (fields or {}).items() if k in _TXN_CURATED}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE inventory_txns SET {sets} WHERE id=?", (*cols.values(), txn_id))
        cx.commit()


def _to_num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _json_get(extras, key):
    import json
    if not extras:
        return None
    try:
        return json.loads(extras).get(key)
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_inventory.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/inventory.py tests/test_inventory.py
git commit -m "feat(inventory): ledger schema + reads + curated write"
```

---

### Task 2: Seeding — `seed_baselines` / `seed_receipts` + CLI

**Files:**
- Modify: `dashboard/inventory.py` (add `seed_baselines`, `seed_receipts`)
- Create: `scripts/seed_inventory_ledger.py`
- Test: `tests/test_seed_inventory.py`

**Interfaces:**
- Produces: `seed_baselines(cx) -> int`; `seed_receipts(cx) -> int` (both take an open connection, caller commits). CLI `scripts/seed_inventory_ledger.py [--write]`.
- Consumes: `ingredients.extras.inventory_starting` / `.par_level_unit`; Phase-3b `po_receiving` + `po_items` + `purchase_orders`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed_inventory.py
import json, sqlite3
from dashboard import inventory as inv


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE ingredients (id INTEGER PRIMARY KEY, fmp_id TEXT, name TEXT, extras TEXT)")
        cx.execute("CREATE TABLE purchase_orders (id INTEGER PRIMARY KEY, po_date TEXT, posted_date TEXT)")
        cx.execute("CREATE TABLE po_items (id INTEGER PRIMARY KEY, po_id INTEGER, ingredient_id INTEGER, material_id INTEGER)")
        cx.execute("CREATE TABLE po_receiving (id INTEGER PRIMARY KEY, po_id INTEGER, po_item_id INTEGER, qty_received REAL, received_size TEXT, created_at TEXT)")
        inv.init_inventory_schema(cx)
        cx.execute("INSERT INTO ingredients VALUES (1,'f1','Mag',?)",
                   (json.dumps({"inventory_starting": "1.0", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO ingredients VALUES (2,'f2','Lipoic',?)", (json.dumps({"par_level": "0.25"}),))  # no baseline
        cx.execute("INSERT INTO purchase_orders VALUES (10,'2026-01-01','2026-01-05')")
        cx.execute("INSERT INTO po_items VALUES (100,10,1,NULL)")        # ingredient line
        cx.execute("INSERT INTO po_items VALUES (101,10,NULL,7)")        # material-only line
        cx.execute("INSERT INTO po_receiving VALUES (1000,10,100,5.0,'kg','2026-01-06')")   # → ingredient 1
        cx.execute("INSERT INTO po_receiving VALUES (1001,10,101,9.0,'ea','2026-01-06')")   # material-only, skip
        cx.commit()
    return db


def test_seed_and_idempotent(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        nb = inv.seed_baselines(cx)
        nr = inv.seed_receipts(cx)
        cx.commit()
    assert nb == 1 and nr == 1                       # one baseline (ing 1), one receipt (ing 1 only)
    assert inv.on_hand(1, db) == 6.0                 # 1.0 baseline + 5.0 received
    assert inv.on_hand(2, db) == 0.0
    # a manual adjustment must survive a re-seed
    inv.add_adjustment(1, -0.5, db_path=db)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        nb2 = inv.seed_baselines(cx)
        nr2 = inv.seed_receipts(cx)
        cx.commit()
    assert nb2 == 0 and nr2 == 0                      # idempotent: nothing re-inserted
    assert inv.on_hand(1, db) == 5.5                  # 6.0 − 0.5, unchanged by re-seed


def test_receipt_date_from_po(tmp_path):
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        inv.seed_receipts(cx); cx.commit()
    rows = inv.list_txns(1, db)
    rec = [t for t in rows if t["txn_type"] == "receipt"][0]
    assert rec["txn_date"] == "2026-01-05"            # posted_date preferred over po_date
    assert rec["source_ref"] == "po_receiving:1000" and rec["unit"] == "kg"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_seed_inventory.py -q`
Expected: FAIL (`AttributeError: seed_baselines`).

- [ ] **Step 3: Add seeding functions to `dashboard/inventory.py`**

```python
def seed_baselines(cx) -> int:
    cx.row_factory = sqlite3.Row
    rows = cx.execute("""
        SELECT id, json_extract(extras,'$.inventory_starting') AS start,
                   json_extract(extras,'$.par_level_unit')      AS unit
        FROM ingredients
        WHERE json_extract(extras,'$.inventory_starting') IS NOT NULL
    """).fetchall()
    n = 0
    for r in rows:
        qty = _to_num(r["start"])
        if qty is None:
            continue
        cur = cx.execute("""
            INSERT OR IGNORE INTO inventory_txns
                (ingredient_id, txn_type, qty, unit, txn_date, source_kind, source_ref)
            VALUES (?, 'baseline', ?, ?, NULL, 'fmp_baseline', ?)
        """, (r["id"], qty, r["unit"], f"baseline:{r['id']}"))
        n += cur.rowcount
    return n


def seed_receipts(cx) -> int:
    cx.row_factory = sqlite3.Row
    rows = cx.execute("""
        SELECT rec.id AS rec_id, pi.ingredient_id AS ingredient_id,
               rec.qty_received AS qty, rec.received_size AS unit,
               COALESCE(po.posted_date, po.po_date, date(rec.created_at)) AS txn_date
        FROM po_receiving rec
        JOIN po_items pi ON pi.id = rec.po_item_id
        LEFT JOIN purchase_orders po ON po.id = rec.po_id
        WHERE pi.ingredient_id IS NOT NULL
          AND rec.qty_received IS NOT NULL AND rec.qty_received <> 0
    """).fetchall()
    n = 0
    for r in rows:
        cur = cx.execute("""
            INSERT OR IGNORE INTO inventory_txns
                (ingredient_id, txn_type, qty, unit, txn_date, source_kind, source_ref)
            VALUES (?, 'receipt', ?, ?, ?, 'po_receiving', ?)
        """, (r["ingredient_id"], float(r["qty"]), r["unit"], r["txn_date"], f"po_receiving:{r['rec_id']}"))
        n += cur.rowcount
    return n
```

- [ ] **Step 4: Write `scripts/seed_inventory_ledger.py`**

```python
#!/usr/bin/env python3
"""Seed the inventory ledger from FMP baseline (ingredients.extras) + Phase-3b receipts."""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import inventory as inv  # noqa: E402


def _db_path():
    base = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / "chat_log.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_db_path())
    ap.add_argument("--write", action="store_true", help="commit (default: dry-run, rolled back)")
    args = ap.parse_args()
    cx = sqlite3.connect(args.db)
    cx.row_factory = sqlite3.Row
    try:
        inv.init_inventory_schema(cx)
        nb = inv.seed_baselines(cx)
        nr = inv.seed_receipts(cx)
        if args.write:
            cx.commit()
            print(f"WROTE baselines={nb} receipts={nr}")
        else:
            cx.rollback()
            print(f"DRY-RUN would insert baselines={nb} receipts={nr} (rolled back)")
    finally:
        cx.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_seed_inventory.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add dashboard/inventory.py scripts/seed_inventory_ledger.py tests/test_seed_inventory.py
git commit -m "feat(inventory): re-runnable baseline + receipt seeding"
```

---

### Task 3: `/api/inventory/*` endpoints + server-side seed

**Files:**
- Modify: `app.py` (`_init_inventory_tables()` wiring + `/api/inventory/*` routes)
- Test: `tests/test_admin_inventory_api.py`

**Interfaces:**
- Consumes: Task 1 + Task 2 functions via `from dashboard import inventory as _inv`.
- Produces: `GET /api/inventory/levels`; `GET /api/inventory/<int:ingredient_id>`; `POST /api/inventory/<int:ingredient_id>/adjust`; `PATCH /api/inventory/txns/<int:txn_id>`; `POST /api/inventory/seed`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_admin_inventory_api.py
import importlib, json, sqlite3, sys
from pathlib import Path
import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    from dashboard.inventory import init_inventory_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        init_purchase_orders_schema(cx); init_inventory_schema(cx)
        cx.execute("INSERT INTO ingredients (id,name,extras) VALUES (1,'Mag',?)",
                   (json.dumps({"par_level": "3", "par_level_unit": "kg"}),))
        cx.execute("INSERT INTO inventory_txns (ingredient_id,txn_type,qty) VALUES (1,'baseline',1.0)")
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


def test_levels_get_adjust_patch(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    lv = c.get("/api/inventory/levels?q=Mag").get_json()["data"]
    assert lv[0]["id"] == 1 and lv[0]["on_hand"] == 1.0 and lv[0]["below_par"] == 1
    d = c.get("/api/inventory/1").get_json()["data"]
    assert d["on_hand"] == 1.0 and "txns" in d
    r = c.post("/api/inventory/1/adjust", json={"qty": 2.5, "notes": "recount"}).get_json()
    assert r["data"]["on_hand"] == 3.5
    tid = c.get("/api/inventory/1").get_json()["data"]["txns"][0]["id"]
    assert c.patch(f"/api/inventory/txns/{tid}", json={"notes": "z"}).status_code == 200
    assert c.get("/api/inventory/999").status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_admin_inventory_api.py -q`
Expected: FAIL (404 on the routes) or SKIP (Pinecone import). If SKIP locally, that is the accepted pattern — proceed to implement.

- [ ] **Step 3: Wire schema init + add endpoints in `app.py`**

Find the purchase-orders schema-init wiring (`_init_purchase_orders_tables()` and its call) and add an inventory init right after it:

```python
def _init_inventory_tables():
    from dashboard.inventory import init_inventory_schema
    cx = sqlite3.connect(str(LOG_DB))
    try:
        init_inventory_schema(cx)
    finally:
        cx.close()

_init_inventory_tables()
```

Add the routes beside the `/api/po/*` block:

```python
from dashboard import inventory as _inv


@app.route("/api/inventory/levels", methods=["GET"])
@require_console_key
def api_inventory_levels():
    try:
        return ok(_inv.inventory_levels(request.args.get("q",""),
                                        int(request.args.get("limit",200)),
                                        int(request.args.get("offset",0))))
    except Exception as e:
        return fail(e)


@app.route("/api/inventory/<int:ingredient_id>", methods=["GET"])
@require_console_key
def api_inventory_get(ingredient_id):
    try:
        d = _inv.get_inventory(ingredient_id)
        if not d:
            return fail("not found", status=404)
        return ok(d)
    except Exception as e:
        return fail(e)


@app.route("/api/inventory/<int:ingredient_id>/adjust", methods=["POST"])
@require_console_key
def api_inventory_adjust(ingredient_id):
    try:
        b = request.get_json(silent=True) or {}
        _inv.add_adjustment(ingredient_id, b.get("qty"), b.get("unit"),
                            b.get("txn_date"), b.get("notes"))
        return ok({"on_hand": _inv.on_hand(ingredient_id)})
    except ValueError as e:
        return fail(str(e), status=400)
    except Exception as e:
        return fail(e)


@app.route("/api/inventory/txns/<int:txn_id>", methods=["PATCH"])
@require_console_key
def api_inventory_txn_patch(txn_id):
    try:
        _inv.update_txn_curated(txn_id, request.get_json(silent=True) or {})
        return ok({"id": txn_id})
    except Exception as e:
        return fail(e)


@app.route("/api/inventory/seed", methods=["POST"])
@require_console_key
def api_inventory_seed():
    try:
        from dashboard.ingredient_catalog import init_ingredients_schema
        from dashboard.purchase_orders import init_purchase_orders_schema
        from dashboard.inventory import init_inventory_schema, seed_baselines, seed_receipts
    except Exception as e:
        return fail(f"import error: {e}")
    try:
        write = request.form.get("write", "").lower() in ("1","true","yes")
        cx = sqlite3.connect(str(LOG_DB))
        cx.row_factory = sqlite3.Row
        try:
            init_ingredients_schema(cx)
            init_purchase_orders_schema(cx)
            init_inventory_schema(cx)
            nb = seed_baselines(cx)
            nr = seed_receipts(cx)
            if write:
                cx.commit()
            else:
                cx.rollback()
        finally:
            cx.close()
        return ok({"mode": "write" if write else "dry_run", "baselines": nb, "receipts": nr})
    except Exception as e:
        app.logger.exception("inventory seed error")
        return fail(str(e))
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_admin_inventory_api.py -q`
Expected: PASS, or SKIP locally on the Pinecone guard (runs in CI). Also smoke: `python3 -c "import app"` (may skip on Pinecone — note it).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_admin_inventory_api.py
git commit -m "feat(inventory): console endpoints + server-side seed"
```

---

### Task 4: Inventory console tab

**Files:**
- Modify: `static/admin-ingredients.html` (Inventory tab)
- Modify: `static/console-search-index.json`

**Interfaces:**
- Consumes: `/api/inventory/*` endpoints from Task 3.

- [ ] **Step 1: Read the existing Purchase-Orders tab in `static/admin-ingredients.html`** — locate the `labels` array, the PO tab button + panel, the PO `<style>` rules (`#po-detail-panel { display:none }`, `#po-empty { ... display:none }`), and the `openPo`/`savePo*` JS helpers. The Inventory tab mirrors them.

- [ ] **Step 2: Add `"inventory"` to the `labels` array** (append after the last entry, e.g. `"po"`).

- [ ] **Step 3: Add the tab button + panel markup** (mirror the PO tab). The panel contains: a search box (`id="inv-search"`), a "Seed from baseline + receipts" button (`id="inv-seed-btn"`), a levels list container (`id="inv-list"`), an empty-state div (`id="inv-empty"`), and a detail panel (`id="inv-detail-panel"`) with an ingredient header (name, on-hand, par), an "Add adjustment / recount" form (qty + optional unit/date/notes + submit), and a ledger table (`id="inv-txns"`).

- [ ] **Step 4: Add CSS (in the `<style>` block, mirroring the PO rules) — REQUIRED initial `display:none`:**

```css
#inv-empty { color: var(--muted); font-size: 13px; padding: 16px 0; display: none; }
#inv-detail-panel { display: none; }
#inv-detail-panel.active { display: block; }
```

- [ ] **Step 5: Add the JS (mirror the PO helpers, reuse `api()`/`escapeHtml`/`showTab`):**

```javascript
let invCurrentId = null;

async function doInvSearch() {
  const q = document.getElementById("inv-search").value.trim();
  const r = await api("/api/inventory/levels?q=" + encodeURIComponent(q));
  const rows = (r && r.data) || [];
  const el = document.getElementById("inv-list");
  if (!rows.length) { el.innerHTML = ""; document.getElementById("inv-empty").style.display = "block"; return; }
  document.getElementById("inv-empty").style.display = "none";
  el.innerHTML = rows.map(function (x) {
    var par = x.par_level != null ? (x.par_level + " " + (x.par_level_unit || "")) : "—";
    var cls = x.below_par ? ' style="color:#b00;font-weight:600"' : "";
    return '<tr' + cls + ' onclick="openInv(' + x.id + ')"><td>' + escapeHtml(x.name || "") +
           '</td><td>' + x.on_hand + '</td><td>' + escapeHtml(par) + '</td></tr>';
  }).join("");
}

async function openInv(id) {
  invCurrentId = id;
  const r = await api("/api/inventory/" + id);
  const d = r && r.data;
  if (!d) return;
  const panel = document.getElementById("inv-detail-panel");
  panel.classList.add("active");
  const par = d.par_level != null ? (d.par_level + " " + (d.par_level_unit || "")) : "—";
  document.getElementById("inv-detail-head").innerHTML =
    "<strong>" + escapeHtml(d.ingredient.name || "") + "</strong> — on hand: " + d.on_hand + " · par: " + escapeHtml(par);
  document.getElementById("inv-txns").innerHTML = (d.txns || []).map(function (t) {
    return "<tr><td>" + escapeHtml(t.txn_date || "—") + "</td><td>" + escapeHtml(t.txn_type) +
           "</td><td>" + t.qty + "</td><td>" + escapeHtml(t.unit || "") + "</td><td>" + escapeHtml(t.notes || "") + "</td></tr>";
  }).join("");
}

async function submitInvAdjust() {
  if (!invCurrentId) return;
  const qty = parseFloat(document.getElementById("inv-adj-qty").value);
  if (isNaN(qty)) { alert("Enter a numeric qty (use − to subtract)"); return; }
  const body = { qty: qty, unit: document.getElementById("inv-adj-unit").value || null,
                 notes: document.getElementById("inv-adj-notes").value || null };
  const r = await api("/api/inventory/" + invCurrentId + "/adjust", "POST", body);
  if (r && r.ok) { document.getElementById("inv-adj-qty").value = ""; openInv(invCurrentId); doInvSearch(); }
}

async function invSeed() {
  if (!confirm("Seed baseline + receipt entries from existing data? (safe to re-run)")) return;
  const fd = new FormData(); fd.append("write", "1");
  const res = await fetch("/api/inventory/seed", { method: "POST", headers: { "X-Console-Key": KEY }, body: fd });
  const j = await res.json();
  alert("Seeded baselines=" + (j.data ? j.data.baselines : "?") + " receipts=" + (j.data ? j.data.receipts : "?"));
  doInvSearch();
}
```

Wire `oninput` (debounced like the PO search if the page debounces; otherwise direct) on `#inv-search` → `doInvSearch`, the seed button → `invSeed`, the adjust submit → `submitInvAdjust`. **Match exactly how the PO tab sends the console key in `api()` and how `matImport`/`poImport` send `X-Console-Key`** — if the page's `api()` helper already injects the key, use `api()` for the seed call too instead of raw `fetch`. Verify HTML parses, ids exist, existing tabs untouched.

- [ ] **Step 6: Register in `static/console-search-index.json`**

Add: `{ "title": "Inventory & Stock", "page": "Products", "url": "/admin/ingredients", "keywords": ["inventory","stock","on hand","on-hand","par","reorder","balance","ledger"] }`

- [ ] **Step 7: Commit**

```bash
git add static/admin-ingredients.html static/console-search-index.json
git commit -m "feat(inventory): admin Inventory tab + search index"
```

---

## Self-Review

- **Spec coverage:** ledger table + on-hand (T1); seeding baseline+receipt idempotent (T2); endpoints + server-side seed (T3); console tab + search index (T4). `consumption` type defined, unwritten (3c-2) ✓. Materials skipped in receipts ✓. Curated = notes only ✓. Idempotency via `source_ref` partial unique + INSERT-OR-IGNORE ✓.
- **Placeholders:** none — every step has full code.
- **Type consistency:** `init_inventory_schema`/`on_hand`/`inventory_levels`/`get_inventory`/`list_txns`/`add_adjustment`/`update_txn_curated`/`seed_baselines`/`seed_receipts` used identically in module, CLI, endpoints, tests; `_inv` alias; `from dashboard import inventory`. `below_par` is 0/1 in both module and test. `source_ref` format `baseline:<id>` / `po_receiving:<id>` consistent across seeding + tests.
- **Reviewer note:** the `api(url, method, body)` JS signature in Task 4 assumes the page's existing helper shape — the implementer must confirm against the real `api()`/PO helpers and match it (do not invent a new signature).
