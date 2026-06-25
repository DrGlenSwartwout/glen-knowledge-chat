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


def _add_col(cx: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    """Idempotent ALTER TABLE ADD COLUMN."""
    have = {r[1] for r in cx.execute(f"PRAGMA table_info({table})")}
    if col not in have:
        cx.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


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
    # Override-protection columns (idempotent — safe on existing DBs)
    _add_col(cx, "ingredients", "overrides", "TEXT")
    _add_col(cx, "ingredients", "par_level", "REAL")
    _add_col(cx, "ingredients", "par_level_unit", "TEXT")
    _add_col(cx, "ingredient_sources", "overrides", "TEXT")
    # One-time backfill: promote par_level/par_level_unit out of extras JSON
    cx.execute("""UPDATE ingredients
                  SET par_level = CAST(json_extract(extras,'$.par_level') AS REAL),
                      par_level_unit = json_extract(extras,'$.par_level_unit')
                  WHERE par_level IS NULL AND json_extract(extras,'$.par_level') IS NOT NULL""")
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
