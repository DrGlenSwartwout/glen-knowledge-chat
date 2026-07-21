"""Ingredients + sources catalog — FMP-migrated raw-material master in chat_log.db.
Mirrors dashboard/shipping.py conventions (idempotent schema, _connect, db_path kwarg)."""
from __future__ import annotations
import json, os, sqlite3
from pathlib import Path
from typing import Optional

from dashboard._core_edit import (
    _set_core as _set_core_field,
    _unlock_core as _unlock_core_field,
    _coerce_core,
)

_ING_CORE = {"name", "form", "common_names", "par_level", "par_level_unit"}
_SRC_CORE = {"price_per_unit", "unit_size", "unit_type"}


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
    from dashboard import db
    if db.backend_of(cx) == "postgres":
        have = {r[0] for r in cx.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name=? AND table_schema=current_schema()", (table,)).fetchall()}
    else:
        have = {r[1] for r in cx.execute(f"PRAGMA table_info({table})")}
    if col not in have:
        cx.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def init_ingredients_schema(cx: sqlite3.Connection) -> None:
    from dashboard import db
    pg = db.backend_of(cx) == "postgres"
    if pg:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS suppliers (
              id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
              fmp_id TEXT, company TEXT NOT NULL,
              address_street TEXT, address_city TEXT, address_province TEXT, address_postal_code TEXT,
              email TEXT, phone_business TEXT, phone_cell TEXT, phone_fax TEXT, url TEXT,
              qb_id TEXT, active INTEGER,
              notes TEXT, extras TEXT,
              created_at TEXT DEFAULT (now()::text), updated_at TEXT DEFAULT (now()::text)
            )""")
    else:
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
    if pg:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
              id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
              fmp_id TEXT, name TEXT NOT NULL, form TEXT, status TEXT,
              common_names TEXT, canonical_id INTEGER REFERENCES ingredients(id),
              extras TEXT,
              inci_name TEXT, cas_number TEXT, hygroscopic_rating TEXT, solubility TEXT,
              stability_notes TEXT, spec_notes TEXT, notes TEXT,
              created_at TEXT DEFAULT (now()::text), updated_at TEXT DEFAULT (now()::text)
            )""")
    else:
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
    if pg:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS ingredient_sources (
              id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
              fmp_id TEXT,
              ingredient_id INTEGER REFERENCES ingredients(id),
              supplier_id INTEGER REFERENCES suppliers(id),
              supplier_name TEXT, sku TEXT,
              price_per_unit REAL, unit_size REAL, unit_type TEXT, shipping_quote REAL,
              extras TEXT,
              preferred INTEGER DEFAULT 0, lead_time_days INTEGER,
              minimum_order REAL, minimum_order_unit TEXT, notes TEXT,
              created_at TEXT DEFAULT (now()::text), updated_at TEXT DEFAULT (now()::text)
            )""")
    else:
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
    if pg:
        cx.execute("""UPDATE ingredients
                      SET par_level = CAST(extras::jsonb ->> 'par_level' AS REAL),
                          par_level_unit = extras::jsonb ->> 'par_level_unit'
                      WHERE par_level IS NULL AND extras::jsonb ->> 'par_level' IS NOT NULL""")
    else:
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


# ---------------------------------------------------------------------------
# Console-create helpers (fmp_id=NULL — importer-invisible by construction)
# ---------------------------------------------------------------------------

_ING_CREATABLE = _ING_CORE | _ING_CURATED               # name/form/common_names/par_*/curated
_SUP_CREATABLE = {"company", "address_street", "address_city", "address_province",
                  "address_postal_code", "email", "phone_business", "phone_cell",
                  "phone_fax", "url", "notes"}
_SRC_CREATABLE = {"ingredient_id", "supplier_id", "supplier_name", "sku", "price_per_unit",
                  "unit_size", "unit_type", "shipping_quote", "preferred", "lead_time_days",
                  "minimum_order", "minimum_order_unit", "notes"}
_SRC_NUMERIC_EXTRA = {"minimum_order", "lead_time_days", "shipping_quote", "supplier_id", "ingredient_id"}


def _insert_allowed(table, fields, allowed, required, numeric_extra=None, db_path=None):
    """Insert a row using only columns in `allowed` (injection guard: cols are f-string interpolated).

    Raises ValueError for any field not in `allowed` BEFORE building SQL — the allowlist
    is the only injection guard, exactly mirroring E1's `_set_core` allowlist check.
    """
    fields = fields or {}
    for k in fields:
        if k not in allowed:
            raise ValueError(f"{k} is not a creatable field of {table}")
    for req in required:
        if str(fields.get(req) if fields.get(req) is not None else "").strip() == "":
            raise ValueError(f"{req} is required")
    cols, vals = [], []
    for k, v in fields.items():
        cols.append(k)
        vals.append(_coerce_core(k, v, numeric_extra=numeric_extra))
    if not cols:
        raise ValueError("no fields to insert")
    with _connect(db_path) as cx:
        cur = cx.execute(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", vals)
        cx.commit()
        return int(cur.lastrowid)


def create_ingredient(fields, db_path=None) -> int:
    """Create a new ingredient with fmp_id=NULL (console-created, importer-invisible)."""
    return _insert_allowed("ingredients", fields, _ING_CREATABLE, {"name"}, db_path=db_path)


def create_supplier(fields, db_path=None) -> int:
    """Create a new supplier with fmp_id=NULL (console-created, importer-invisible)."""
    return _insert_allowed("suppliers", fields, _SUP_CREATABLE, {"company"}, db_path=db_path)


def create_source(ingredient_id, fields, db_path=None) -> int:
    """Create a new ingredient_source row linking to an existing ingredient.

    Validates the ingredient exists; injects ingredient_id; if `preferred` is truthy,
    calls set_preferred_source after insert to clear other preferred flags.
    """
    with _connect(db_path) as cx:
        if not cx.execute("SELECT 1 FROM ingredients WHERE id=?", (ingredient_id,)).fetchone():
            raise ValueError(f"no ingredient {ingredient_id}")
    f = {**(fields or {}), "ingredient_id": ingredient_id}
    # Robust truthiness (callers incl. the email collector may send 1/"1"/true/2/etc.);
    # normalize the stored flag to 0/1 so WHERE preferred=1 always matches.
    _pref = f.get("preferred")
    want_pref = (_pref in (1, True)) or (str(_pref).strip().lower() in ("1", "true", "yes")) \
        or (isinstance(_pref, (int, float)) and not isinstance(_pref, bool) and _pref != 0)
    if "preferred" in f:
        f["preferred"] = 1 if want_pref else 0
    sid = _insert_allowed("ingredient_sources", f, _SRC_CREATABLE, set(),
                          numeric_extra=_SRC_NUMERIC_EXTRA, db_path=db_path)
    if want_pref:
        set_preferred_source(sid, db_path=db_path)   # unsets others for this ingredient
    return sid


# ---------------------------------------------------------------------------
# Core-field editing (FMP override tracking)
# ---------------------------------------------------------------------------

def set_ingredient_core(row_id, field, value, db_path=None):
    """Write a core ingredient field and record it in the overrides set."""
    _set_core_field(_connect, "ingredients", _ING_CORE, row_id, field, value, db_path=db_path)


def unlock_ingredient_core(row_id, field, db_path=None):
    """Remove a field from the ingredient overrides set (value unchanged)."""
    _unlock_core_field(_connect, "ingredients", _ING_CORE, row_id, field, db_path=db_path)


def set_source_core(row_id, field, value, db_path=None):
    """Write a core ingredient_sources field and record it in the overrides set."""
    _set_core_field(_connect, "ingredient_sources", _SRC_CORE, row_id, field, value, db_path=db_path)


def unlock_source_core(row_id, field, db_path=None):
    """Remove a field from the ingredient_sources overrides set (value unchanged)."""
    _unlock_core_field(_connect, "ingredient_sources", _SRC_CORE, row_id, field, db_path=db_path)
