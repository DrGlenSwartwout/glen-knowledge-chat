"""Shared helpers for core-field editing with override tracking.

Both ingredient_catalog.py and formulations.py import from here to keep
_coerce_core / _set_core / _unlock_core as a single source of truth.
"""
from __future__ import annotations
import json
from typing import Optional, Set


# Fields that must coerce to float — extended by callers via the numeric_extra arg.
_NUMERIC_BASE: Set[str] = {"par_level", "price_per_unit", "unit_size"}


def _coerce_core(field: str, value, numeric_extra: Optional[Set[str]] = None):
    """Coerce a core-field value for storage.

    - common_names: comma-separated string → JSON array
    - numeric fields (_NUMERIC_BASE ∪ numeric_extra): float, None on empty, ValueError on bad input
    - everything else: string or None on empty string
    """
    numeric = _NUMERIC_BASE | (numeric_extra or set())
    if field == "common_names":
        parts = [p.strip() for p in str(value or "").split(",") if p.strip()]
        return json.dumps(parts, ensure_ascii=False)
    if field in numeric:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be numeric, got {value!r}")
    return value if value not in ("",) else None


def _set_core(connect_fn, table: str, allowed: Set[str], row_id: int,
              field: str, value, db_path: Optional[str] = None,
              numeric_extra: Optional[Set[str]] = None) -> None:
    """Write a single core field, mark it in the overrides JSON set."""
    if field not in allowed:
        raise ValueError(f"{field!r} is not an editable core field of {table}")
    v = _coerce_core(field, value, numeric_extra=numeric_extra)
    with connect_fn(db_path) as cx:
        row = cx.execute(f"SELECT overrides FROM {table} WHERE id=?", (row_id,)).fetchone()
        if not row:
            raise ValueError(f"no {table} row with id={row_id}")
        ov: Set[str] = set(json.loads(row["overrides"] or "[]"))
        ov.add(field)
        cx.execute(
            f"UPDATE {table} SET {field}=?, overrides=?, updated_at=datetime('now') WHERE id=?",
            (v, json.dumps(sorted(ov)), row_id),
        )
        cx.commit()


def _unlock_core(connect_fn, table: str, row_id: int, field: str,
                 db_path: Optional[str] = None) -> None:
    """Remove a field from the overrides set (value is left unchanged)."""
    with connect_fn(db_path) as cx:
        row = cx.execute(f"SELECT overrides FROM {table} WHERE id=?", (row_id,)).fetchone()
        if not row:
            raise ValueError(f"no {table} row with id={row_id}")
        ov: Set[str] = set(json.loads(row["overrides"] or "[]"))
        ov.discard(field)
        cx.execute(
            f"UPDATE {table} SET overrides=?, updated_at=datetime('now') WHERE id=?",
            (json.dumps(sorted(ov)), row_id),
        )
        cx.commit()
