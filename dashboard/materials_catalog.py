"""Materials (production inputs + packaging) catalog — FMP-migrated.
Mirrors dashboard/ingredient_catalog.py; references Phase-1 suppliers."""
from __future__ import annotations
import sqlite3
from dashboard.ingredient_catalog import _connect


def init_materials_schema(cx: sqlite3.Connection) -> None:
    from dashboard import db
    if db.backend_of(cx) == "postgres":
        cx.execute("""
            CREATE TABLE IF NOT EXISTS materials (
              id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
              fmp_id TEXT, name TEXT NOT NULL, type TEXT, status TEXT,
              extras TEXT, notes TEXT,
              created_at TEXT DEFAULT (now()::text), updated_at TEXT DEFAULT (now()::text)
            )""")
    else:
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
        if db.backend_of(cx) == "postgres":
            cx.execute(f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                  fmp_id TEXT, {link},
                  supplier_id INTEGER REFERENCES suppliers(id),
                  supplier_name TEXT, sku TEXT, price REAL,
                  purchase_size REAL, purchase_size_unit TEXT, mfg TEXT, contact TEXT, product_link TEXT,
                  extras TEXT, preferred INTEGER DEFAULT 0, notes TEXT,
                  created_at TEXT DEFAULT (now()::text), updated_at TEXT DEFAULT (now()::text)
                )""")
        else:
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
