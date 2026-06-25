"""Formulations (recipes) — FMP-migrated, references Phase-1 ingredients.
Mirrors dashboard/ingredient_catalog.py conventions."""
from __future__ import annotations
import sqlite3
from dashboard.ingredient_catalog import _connect, _add_col  # reuse Phase-1 helpers


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
    # Override-protection column (idempotent — safe on existing DBs)
    _add_col(cx, "formulation_items", "overrides", "TEXT")
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
