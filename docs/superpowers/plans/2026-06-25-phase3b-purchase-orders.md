# Phase 3b: Purchase Orders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate FileMaker purchase-order history (POs + line items + receiving) into `chat_log.db`, with an importer, console view, and server-side import — line items reference the Phase 1/2/3a ingredients, materials, and products.

**Architecture:** New `dashboard/purchase_orders.py` (schema + reads + curated writes, reuses `_connect`). Importer `scripts/import_purchase_orders_from_fmp.py` reuses Phase-1 helpers + resolves line links. Console = `/api/po/*` + a Purchase Orders tab. Server-side import mirrors `/api/materials/import`.

**Tech Stack:** Python 3, Flask, sqlite3, pytest, vanilla-JS. Stdlib only.

## Global Constraints
- New module **`dashboard/purchase_orders.py`** (verified free; `ls` before create). Reuses `_connect` from `dashboard.ingredient_catalog`. FKs Phase-1 `suppliers`/`ingredients`, Phase-3a `materials`; `fmp_product_id` joins Phase-2 products.json `fmp_id`.
- **Reuse:** `from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert`.
- Curated-vs-FMP split: importer writes only FMP cols; curated = `purchase_orders.notes`, `po_items.notes`. `_upsert` enforces it.
- `fmp_id` (= FMP id_pk) idempotent key; partial unique index `WHERE fmp_id IS NOT NULL`. csv.field_size_limit(sys.maxsize). REAL money, JSON extras, timestamps.
- Console `@require_console_key`, `ok`/`fail`, `?key=`; server import mirrors `/api/materials/import`.
- This is read-only PO history (no forward PO creation — that's a later follow-on, likely driven by 3c demand).
- FMP source (exported, verified): `/tmp/fmp-export/newapp/{po,po_items,po_receiving}.csv` (120/165/116). Fields — po: `id_pk,po_date,id_fk_supplier,vendor_po_no,notes,closed,locked,active,tax,shipping_amount,shipper,tracking_number,due_date,posted_date,qb_id`. po_items: `id_pk,id_fk_po,id_fk_raw,id_fk_material,id_fk_product,product_id,cost,qty,qty_unit,qty_left,fee_name,tax`. po_receiving: `id_pk,id_fk_po,id_fk_po_item,id_fk_material,id_fk_product,id_fk_raw,qty_received,received_size`. (`po_payments` is OUT — it links via id_fk_expense, not the PO.)

---

### Task 1: `dashboard/purchase_orders.py` — schema + reads + curated writes + wiring

**Files:** Create `dashboard/purchase_orders.py`; Modify `app.py` (`_init_purchase_orders_tables()` after `_init_materials_tables()`); Test `tests/test_purchase_orders.py`.

**Produces:** `init_purchase_orders_schema(cx)`; `search_purchase_orders(q,limit,offset,db_path)` (q against vendor_po_no or supplier company); `get_purchase_order(id,db_path)`; `list_po_items(po_id,db_path)` (resolve item label: ingredient/material name or product slug); `list_po_receiving(po_id,db_path)`; `update_po_curated(id,fields,db_path)` (only `notes`); `update_po_item_curated(id,fields,db_path)` (only `notes`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_purchase_orders.py
import sqlite3, pytest

@pytest.fixture
def db(tmp_path):
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx)
        init_purchase_orders_schema(cx); init_purchase_orders_schema(cx)
    return p

def test_schema_reads_curated(db):
    from dashboard.purchase_orders import (search_purchase_orders, get_purchase_order,
        list_po_items, list_po_receiving, update_po_curated)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')")
        sid = cx.execute("SELECT id FROM suppliers").fetchone()[0]
        cx.execute("INSERT INTO ingredients (fmp_id,name) VALUES ('r1','R-Lipoic Acid')")
        iid = cx.execute("SELECT id FROM ingredients").fetchone()[0]
        cx.execute("INSERT INTO purchase_orders (fmp_id,supplier_id,vendor_po_no,po_date) VALUES ('p1',?,'PO-100','2026-01-01')",(sid,))
        pid = cx.execute("SELECT id FROM purchase_orders").fetchone()[0]
        cx.execute("INSERT INTO po_items (fmp_id,po_id,ingredient_id,item_label,qty,cost) VALUES ('it1',?,?, 'R-Lipoic Acid', 2, 50)",(pid,iid))
        itid = cx.execute("SELECT id FROM po_items").fetchone()[0]
        cx.execute("INSERT INTO po_receiving (fmp_id,po_id,po_item_id,qty_received) VALUES ('rc1',?,?,2)",(pid,itid))
        cx.commit()
    r = search_purchase_orders("PO-100", db_path=db)
    assert r[0]["vendor_po_no"]=="PO-100" and r[0]["supplier_company"]=="Acme"
    items = list_po_items(pid, db_path=db)
    assert items[0]["qty"]==2 and items[0]["item_label"]=="R-Lipoic Acid" and items[0]["ingredient_canonical"]=="R-Lipoic Acid"
    assert list_po_receiving(pid, db_path=db)[0]["qty_received"]==2
    update_po_curated(pid, {"notes":"x","vendor_po_no":"HACK"}, db_path=db)
    g = get_purchase_order(pid, db_path=db)
    assert g["notes"]=="x" and g["vendor_po_no"]=="PO-100"
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_purchase_orders.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement `dashboard/purchase_orders.py`**

```python
"""Purchase orders (history) — FMP-migrated. PO header + line items + receiving.
Line items reference Phase-1 ingredients / Phase-3a materials / Phase-2 products.
Mirrors dashboard/ingredient_catalog.py."""
from __future__ import annotations
import sqlite3
from dashboard.ingredient_catalog import _connect


def init_purchase_orders_schema(cx: sqlite3.Connection) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS purchase_orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT, supplier_id INTEGER REFERENCES suppliers(id),
          supplier_name TEXT, vendor_po_no TEXT, po_date TEXT, status TEXT,
          tax REAL, shipping_amount REAL, shipper TEXT, tracking_number TEXT,
          due_date TEXT, posted_date TEXT, qb_id TEXT,
          extras TEXT, notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_po_fmp ON purchase_orders(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS po_items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT, po_id INTEGER REFERENCES purchase_orders(id),
          item_kind TEXT, item_label TEXT,
          ingredient_id INTEGER REFERENCES ingredients(id),
          material_id INTEGER REFERENCES materials(id),
          fmp_product_id TEXT, sku TEXT,
          qty REAL, qty_unit TEXT, qty_left REAL, cost REAL,
          extras TEXT, notes TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_poitems_fmp ON po_items(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_poitems_po ON po_items(po_id)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS po_receiving (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          fmp_id TEXT, po_id INTEGER REFERENCES purchase_orders(id),
          po_item_id INTEGER REFERENCES po_items(id),
          qty_received REAL, received_size TEXT, extras TEXT,
          created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_porec_fmp ON po_receiving(fmp_id) WHERE fmp_id IS NOT NULL")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_porec_po ON po_receiving(po_id)")
    cx.commit()


def search_purchase_orders(q="", limit=50, offset=0, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT po.*, sup.company AS supplier_company FROM purchase_orders po
            LEFT JOIN suppliers sup ON sup.id = po.supplier_id
            WHERE po.vendor_po_no LIKE ? OR sup.company LIKE ?
            ORDER BY po.po_date DESC, po.id DESC LIMIT ? OFFSET ?
        """, (f"%{q}%", f"%{q}%", int(limit), int(offset))).fetchall()
    return [dict(r) for r in rows]


def get_purchase_order(po_id, db_path=None):
    with _connect(db_path) as cx:
        r = cx.execute("""
            SELECT po.*, sup.company AS supplier_company FROM purchase_orders po
            LEFT JOIN suppliers sup ON sup.id = po.supplier_id WHERE po.id=?
        """, (po_id,)).fetchone()
    return dict(r) if r else None


def list_po_items(po_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("""
            SELECT pi.*, ing.name AS ingredient_canonical, mat.name AS material_name
            FROM po_items pi
            LEFT JOIN ingredients ing ON ing.id = pi.ingredient_id
            LEFT JOIN materials mat ON mat.id = pi.material_id
            WHERE pi.po_id = ? ORDER BY pi.id
        """, (po_id,)).fetchall()
    return [dict(r) for r in rows]


def list_po_receiving(po_id, db_path=None):
    with _connect(db_path) as cx:
        rows = cx.execute("SELECT * FROM po_receiving WHERE po_id=? ORDER BY id", (po_id,)).fetchall()
    return [dict(r) for r in rows]


_PO_CURATED = {"notes"}
_ITEM_CURATED = {"notes"}


def _update_allowed(table, row_id, fields, allowed, db_path):
    cols = {k: v for k, v in (fields or {}).items() if k in allowed}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols) + ", updated_at=datetime('now')"
    with _connect(db_path) as cx:
        cx.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*cols.values(), row_id))
        cx.commit()


def update_po_curated(po_id, fields, db_path=None):
    _update_allowed("purchase_orders", po_id, fields, _PO_CURATED, db_path)


def update_po_item_curated(item_id, fields, db_path=None):
    _update_allowed("po_items", item_id, fields, _ITEM_CURATED, db_path)
```

- [ ] **Step 4: Wire at app load** — in `app.py` after `_init_materials_tables()`:

```python
def _init_purchase_orders_tables():
    from dashboard.purchase_orders import init_purchase_orders_schema
    with sqlite3.connect(LOG_DB) as cx:
        init_purchase_orders_schema(cx)

_init_purchase_orders_tables()
```

- [ ] **Step 5: Run + commit** — `python3 -m pytest tests/test_purchase_orders.py -q` → PASS.
```bash
git add dashboard/purchase_orders.py app.py tests/test_purchase_orders.py
git commit -m "feat(po): purchase-orders schema + reads + curated writes + app wiring"
```

---

### Task 2: FMP purchase-orders importer

**Files:** Create `scripts/import_purchase_orders_from_fmp.py`; Test `tests/test_import_purchase_orders.py`.

**Produces:** `import_purchase_orders(cx, rows) -> int`; `import_po_items(cx, rows) -> dict` (resolve ingredient_id via id_fk_raw, material_id via id_fk_material, fmp_product_id raw from id_fk_product; item_kind + item_label derived); `import_po_receiving(cx, rows) -> int` (resolve po_id via id_fk_po, po_item_id via id_fk_po_item); CLI dry-run/`--write`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_purchase_orders.py
import sqlite3
from dashboard.ingredient_catalog import init_ingredients_schema
from dashboard.materials_catalog import init_materials_schema
from dashboard.purchase_orders import init_purchase_orders_schema
from scripts.import_purchase_orders_from_fmp import import_purchase_orders, import_po_items, import_po_receiving

def _db(tmp_path):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx); init_purchase_orders_schema(cx)
        cx.execute("INSERT INTO suppliers (fmp_id,company) VALUES ('s1','Acme')")
        cx.execute("INSERT INTO ingredients (fmp_id,name) VALUES ('r1','R-Lipoic Acid')")
        cx.execute("INSERT INTO materials (fmp_id,name) VALUES ('m1','Caps')"); cx.commit()
    return p

def test_import(tmp_path):
    p = _db(tmp_path)
    with sqlite3.connect(p) as cx:
        cx.row_factory = sqlite3.Row
        npo = import_purchase_orders(cx, [{"id_pk":"p1","id_fk_supplier":"s1","vendor_po_no":"PO-1","po_date":"2026-01-01","closed":"No"}])
        ri = import_po_items(cx, [
            {"id_pk":"it1","id_fk_po":"p1","id_fk_raw":"r1","product_id":"SKU","qty":"2","cost":"50","qty_unit":"kg"},
            {"id_pk":"it2","id_fk_po":"p1","id_fk_material":"m1","qty":"10"},
            {"id_pk":"it3","id_fk_po":"p1","id_fk_product":"5161","fee_name":"Freight"},
        ])
        nr = import_po_receiving(cx, [{"id_pk":"rc1","id_fk_po":"p1","id_fk_po_item":"it1","qty_received":"2"}])
        cx.commit()
        po = cx.execute("SELECT * FROM purchase_orders WHERE fmp_id='p1'").fetchone()
        items = {r["fmp_id"]: dict(r) for r in cx.execute("SELECT * FROM po_items")}
        rec = cx.execute("SELECT * FROM po_receiving WHERE fmp_id='rc1'").fetchone()
    assert npo==1 and po["supplier_id"] is not None and po["vendor_po_no"]=="PO-1" and po["status"]=="open"
    assert items["it1"]["ingredient_id"] is not None and items["it1"]["item_kind"]=="ingredient" and items["it1"]["item_label"]=="R-Lipoic Acid" and items["it1"]["qty"]==2.0
    assert items["it2"]["material_id"] is not None and items["it2"]["item_kind"]=="material" and items["it2"]["item_label"]=="Caps"
    assert items["it3"]["fmp_product_id"]=="5161" and items["it3"]["item_kind"]=="product"
    assert ri["items"]==3
    assert nr==1 and rec["po_id"]==po["id"] and rec["po_item_id"]==items["it1"]["id"] and rec["qty_received"]==2.0

def test_reimport_preserves_curated(tmp_path):
    p = _db(tmp_path)
    from dashboard.purchase_orders import update_po_curated
    with sqlite3.connect(p) as cx:
        import_purchase_orders(cx, [{"id_pk":"p1","id_fk_supplier":"s1","vendor_po_no":"PO-1","closed":"No"}]); cx.commit()
        pid = cx.execute("SELECT id FROM purchase_orders WHERE fmp_id='p1'").fetchone()[0]
    update_po_curated(pid, {"notes":"keep"}, db_path=p)
    with sqlite3.connect(p) as cx:
        import_purchase_orders(cx, [{"id_pk":"p1","id_fk_supplier":"s1","vendor_po_no":"PO-1-REV","closed":"Yes"}]); cx.commit()
        cx.row_factory=sqlite3.Row
        po = cx.execute("SELECT * FROM purchase_orders WHERE fmp_id='p1'").fetchone()
    assert po["vendor_po_no"]=="PO-1-REV" and po["status"]=="closed" and po["notes"]=="keep"
```

- [ ] **Step 2: Run to verify it fails** — ModuleNotFoundError.

- [ ] **Step 3: Implement `scripts/import_purchase_orders_from_fmp.py`**

```python
"""Import FileMaker purchase-order history (po + po_items + po_receiving) into
chat_log.db. Line items reference ingredients/materials/products. Idempotent by
fmp_id; curated-preserving. Dry-run default; --write."""
from __future__ import annotations
import argparse, csv, os, sqlite3, sys
csv.field_size_limit(sys.maxsize)
from scripts.import_ingredients_from_fmp import _active, _num, _clean, _extras, _upsert

EXPORT_DIR = os.environ.get("FMP_EXPORT_DIR", "/tmp/fmp-export/newapp")


def import_purchase_orders(cx, rows):
    sup = {r["fmp_id"]: (r["id"], r["company"]) for r in cx.execute("SELECT id, fmp_id, company FROM suppliers WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = ["supplier_id", "supplier_name", "vendor_po_no", "po_date", "status", "tax",
                "shipping_amount", "shipper", "tracking_number", "due_date", "posted_date", "qb_id", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_supplier", "closed", "notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        s = sup.get((r.get("id_fk_supplier") or "").strip())
        sid, sname = (s[0], s[1]) if s else (None, None)
        status = "closed" if _active(r.get("closed")) == 1 else "open"
        vals = [fid, sid, sname, _clean(r.get("vendor_po_no")) or None, _clean(r.get("po_date")) or None, status,
                _num(r.get("tax")), _num(r.get("shipping_amount")), _clean(r.get("shipper")) or None,
                _clean(r.get("tracking_number")) or None, _clean(r.get("due_date")) or None,
                _clean(r.get("posted_date")) or None, _clean(r.get("qb_id")) or None, _extras(r, mapped)]
        _upsert(cx, "purchase_orders", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def import_po_items(cx, rows):
    po = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM purchase_orders WHERE fmp_id IS NOT NULL")}
    ing = {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute("SELECT id, fmp_id, name FROM ingredients WHERE fmp_id IS NOT NULL")}
    mat = {r["fmp_id"]: (r["id"], r["name"]) for r in cx.execute("SELECT id, fmp_id, name FROM materials WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = ["po_id", "item_kind", "item_label", "ingredient_id", "material_id", "fmp_product_id",
                "sku", "qty", "qty_unit", "qty_left", "cost", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_po", "id_fk_raw", "id_fk_material", "id_fk_product",
                              "product_id", "fee_name", "notes"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        po_id = po.get((r.get("id_fk_po") or "").strip())
        raw = (r.get("id_fk_raw") or "").strip()
        matf = (r.get("id_fk_material") or "").strip()
        prod = (r.get("id_fk_product") or "").strip()
        ing_id = ing.get(raw)
        mat_id = mat.get(matf)
        if ing_id:
            kind, label, iid, mid, pid = "ingredient", ing_id[1], ing_id[0], None, None
        elif mat_id:
            kind, label, iid, mid, pid = "material", mat_id[1], None, mat_id[0], None
        elif prod:
            kind, label, iid, mid, pid = "product", _clean(r.get("fee_name")) or None, None, None, prod
        else:
            kind, label, iid, mid, pid = ("fee" if _clean(r.get("fee_name")) else "other"), _clean(r.get("fee_name")) or None, None, None, None
        vals = [fid, po_id, kind, label, iid, mid, pid, _clean(r.get("product_id")) or None,
                _num(r.get("qty")), _clean(r.get("qty_unit")) or None, _num(r.get("qty_left")), _num(r.get("cost")), _extras(r, mapped)]
        _upsert(cx, "po_items", fmp_cols, vals, fmp_cols)
        n += 1
    return {"items": n}


def import_po_receiving(cx, rows):
    po = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM purchase_orders WHERE fmp_id IS NOT NULL")}
    item = {r["fmp_id"]: r["id"] for r in cx.execute("SELECT id, fmp_id FROM po_items WHERE fmp_id IS NOT NULL")}
    n = 0
    fmp_cols = ["po_id", "po_item_id", "qty_received", "received_size", "extras"]
    mapped = set(fmp_cols) | {"id_pk", "id_fk_po", "id_fk_po_item", "id_fk_material", "id_fk_product", "id_fk_raw"}
    for r in rows:
        fid = (r.get("id_pk") or "").strip()
        if not fid:
            continue
        vals = [fid, po.get((r.get("id_fk_po") or "").strip()), item.get((r.get("id_fk_po_item") or "").strip()),
                _num(r.get("qty_received")), _clean(r.get("received_size")) or None, _extras(r, mapped)]
        _upsert(cx, "po_receiving", fmp_cols, vals, fmp_cols)
        n += 1
    return n


def _read(name):
    with open(os.path.join(EXPORT_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    po = _read("po.csv"); items = _read("po_items.csv"); rec = _read("po_receiving.csv")
    print(f"po={len(po)} po_items={len(items)} po_receiving={len(rec)}")
    if not args.write:
        print("(dry run — pass --write)")
        return 0
    from dashboard.ingredient_catalog import _default_db_path, init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    cx = sqlite3.connect(args.db or _default_db_path()); cx.row_factory = sqlite3.Row
    init_ingredients_schema(cx); init_materials_schema(cx); init_purchase_orders_schema(cx)
    npo = import_purchase_orders(cx, po); ri = import_po_items(cx, items); nr = import_po_receiving(cx, rec)
    cx.commit(); cx.close()
    print(f"wrote purchase_orders={npo} po_items={ri['items']} po_receiving={nr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests + real --write into a temp DB**

Run: `python3 -m pytest tests/test_import_purchase_orders.py -q` → PASS.
Real (seed ingredients+materials first into the same temp db):
```bash
python3 scripts/import_ingredients_from_fmp.py --write --db /tmp/p3b.db
python3 scripts/import_materials_from_fmp.py --write --db /tmp/p3b.db
python3 scripts/import_purchase_orders_from_fmp.py --write --db /tmp/p3b.db
```
Expected `purchase_orders=120 po_items=165 po_receiving=116`. Capture counts + how many po_items resolved to ingredient/material/product.

- [ ] **Step 5: Commit**
```bash
git add scripts/import_purchase_orders_from_fmp.py tests/test_import_purchase_orders.py
git commit -m "feat(po): FMP importer (POs + items + receiving, linked to catalog)"
```

---

### Task 3: PO console (endpoints + tab)

**Files:** Modify `app.py` (`/api/po/*`), `static/admin-ingredients.html` (Purchase Orders tab); Test `tests/test_admin_po_api.py`.

**Consumes Task 1 reads/writes.** Endpoints (all `@require_console_key`, `ok`/`fail`, `from dashboard import purchase_orders as _po`): GET `/api/po/search`; GET `/api/po/<int:pid>` → `{po, items, receiving}` (404 if None); PATCH `/api/po/<int:pid>` (update_po_curated); PATCH `/api/po/items/<int:item_id>` (update_po_item_curated).

- [ ] **Step 1: Write failing test** (route-level, Pinecone-skip, mirror `tests/test_admin_materials_api.py`)

```python
# tests/test_admin_po_api.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest

def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard.ingredient_catalog import init_ingredients_schema
    from dashboard.materials_catalog import init_materials_schema
    from dashboard.purchase_orders import init_purchase_orders_schema
    with sqlite3.connect(db) as cx:
        init_ingredients_schema(cx); init_materials_schema(cx); init_purchase_orders_schema(cx)
        cx.execute("INSERT INTO purchase_orders (fmp_id,vendor_po_no) VALUES ('p1','PO-1')"); cx.commit()
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
    r = c.get("/api/po/search?q=PO-1").get_json()
    pid = r["data"][0]["id"]
    d = c.get(f"/api/po/{pid}").get_json()["data"]
    assert d["po"]["vendor_po_no"]=="PO-1" and "items" in d and "receiving" in d
    assert c.patch(f"/api/po/{pid}", json={"notes":"x"}).status_code == 200
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404) or SKIP.

- [ ] **Step 3: Implement endpoints** (in `app.py`, beside `/api/materials/*`)

```python
from dashboard import purchase_orders as _po

@app.route("/api/po/search", methods=["GET"])
@require_console_key
def api_po_search():
    try:
        return ok(_po.search_purchase_orders(request.args.get("q",""), int(request.args.get("limit",50)), int(request.args.get("offset",0))))
    except Exception as e: return fail(e)

@app.route("/api/po/<int:pid>", methods=["GET"])
@require_console_key
def api_po_get(pid):
    try:
        p = _po.get_purchase_order(pid)
        if not p: return fail("not found", status=404)
        return ok({"po": p, "items": _po.list_po_items(pid), "receiving": _po.list_po_receiving(pid)})
    except Exception as e: return fail(e)

@app.route("/api/po/<int:pid>", methods=["PATCH"])
@require_console_key
def api_po_patch(pid):
    try:
        _po.update_po_curated(pid, request.get_json(silent=True) or {})
        return ok(_po.get_purchase_order(pid))
    except Exception as e: return fail(e)

@app.route("/api/po/items/<int:item_id>", methods=["PATCH"])
@require_console_key
def api_po_item_patch(item_id):
    try:
        _po.update_po_item_curated(item_id, request.get_json(silent=True) or {})
        return ok({"id": item_id})
    except Exception as e: return fail(e)
```

- [ ] **Step 4: Purchase Orders tab in `static/admin-ingredients.html`** — read the page first; add `"po"` to the `labels` array + a tab button + panel mirroring the Materials tab: debounced search `GET /api/po/search?q=` → list (vendor_po_no + supplier_company + po_date); on click `GET /api/po/<id>` → PO header read-only (vendor_po_no, supplier_company, po_date, status, tax, shipping) + editable curated `notes` (`PATCH /api/po/<id>`); a line-items table (item_label/kind, qty+qty_unit, cost) with editable item `notes` (`PATCH /api/po/items/<id>`); a receiving table (qty_received, received_size). Reuse `api()`/`escapeHtml`/`showTab`; FMP read-only; only notes editable; **add the `display:none` initial-state CSS for the PO detail panel + empty div (mirror the materials/formulations CSS rules)**. Verify HTML parses; ids exist; existing tabs untouched.

- [ ] **Step 5: Run tests + commit**
```bash
git add app.py static/admin-ingredients.html tests/test_admin_po_api.py
git commit -m "feat(po): console endpoints + purchase-orders tab"
```

---

### Task 4: Server-side PO import endpoint + UI section

**Files:** Modify `app.py` (`POST /api/po/import`), `static/admin-ingredients.html` (import section); Test `tests/test_admin_po_import.py`.

Mirror `/api/materials/import` exactly. Multipart files `po`, `po_items`, `po_receiving` + `write`. Dry-run → counts. Write → open LOG_DB, init ingredients+materials+purchase_orders schemas, run import_purchase_orders + import_po_items + import_po_receiving, commit, close (try/finally), return counts. field_size_limit; lazy imports; missing any file → 400. UI: a "Purchase Orders" section in the Import tab with 3 file pickers + Dry-run/Import (`poImport(write)` mirroring `matImport`); hint "import Suppliers + Ingredients + Materials first".

- [ ] **Step 1–5:** Mirror Task 4 of Phase 3a (templates: `/api/materials/import` handler, `tests/test_admin_materials_import.py`, the `matImport` JS + "Materials" import section). TDD; route test skips locally. Commit `feat(po): server-side PO import endpoint + UI section`.

---

## Self-Review
**Coverage:** po+po_items+po_receiving schema/reads/curated (T1); importer with catalog link resolution + receiving (T2); console (T3); server-side import (T4). PO history migrated (Glen's decision). ✓
**Placeholders:** complete code T1-T3; T4 mirrors the named Phase-3a templates. ✓
**Type consistency:** `_connect`/`db_path`; reused `_active/_num/_clean/_extras/_upsert`; `import_purchase_orders/import_po_items/import_po_receiving`; `purchase_orders as _po` alias unique; endpoint→module fn names. ✓
**Out of scope (note):** `po_payments` (expense-linked, no PO FK); forward PO creation (later, 3c-driven); FMP `notes` not imported (curated-owned).

## Operational note
Activation: console Import tab → import Suppliers/Ingredients/Materials first, then Purchase Orders (3 CSVs). Re-export the FMP po tables before each refresh.
