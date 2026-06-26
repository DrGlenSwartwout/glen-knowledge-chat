"""Purchase orders (history) — FMP-migrated. PO header + line items + receiving.
Line items reference Phase-1 ingredients / Phase-3a materials / Phase-2 products.
Mirrors dashboard/ingredient_catalog.py."""
from __future__ import annotations
import json
import sqlite3
from datetime import date
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


def create_draft_po(cx, supplier_id, supplier_name, lines):
    """Create a draft purchase order + its line items from reorder-report lines.
    `cx` is an open sqlite3 connection. Lines missing ingredient_id or suggested_qty are
    skipped; price_per_unit may be None (cost stored NULL). Returns {po_id, line_count}."""
    today = date.today().isoformat()
    vendor_po_no = "DRAFT-" + today.replace("-", "") + "-" + str(supplier_id)
    cur = cx.execute(
        "INSERT INTO purchase_orders (supplier_id, supplier_name, vendor_po_no, po_date, status) "
        "VALUES (?,?,?,?,'draft')",
        (supplier_id, supplier_name or "", vendor_po_no, today))
    po_id = cur.lastrowid
    n = 0
    for ln in (lines or []):
        ing_id = ln.get("ingredient_id")
        qty = ln.get("suggested_qty")
        if ing_id is None or qty is None:
            continue
        c = ln.get("price_per_unit")
        cost = float(c) if c not in (None, "") else None
        extras = json.dumps({k: ln.get(k) for k in ("unit_size", "packs", "est_cost")})
        cx.execute(
            "INSERT INTO po_items (po_id, item_kind, item_label, ingredient_id, qty, qty_unit, cost, extras) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (po_id, "ingredient", ln.get("ingredient") or "", int(ing_id),
             float(qty), ln.get("unit"), cost, extras))
        n += 1
    cx.commit()
    return {"po_id": po_id, "line_count": n}


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
