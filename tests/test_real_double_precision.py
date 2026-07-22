"""P06 C4: SQLite `REAL` column declarations must be `DOUBLE PRECISION` at the
source DDL, so Postgres gets true float8 instead of a truncating float4 (`real`).

Why source-edit and not a pgcompat.py regex translator: `REAL` is a common
English word that shows up in string-literal DATA (e.g. "the REAL catalog"),
and dashboard/pgcompat.py's `_translate_ddl_idioms` runs on raw SQL including
string literals — a `\\bREAL\\b` regex there would corrupt data. The fix is a
source-level column-type edit instead, which is safe on both backends:
SQLite gives `DOUBLE PRECISION` the same REAL storage-class affinity it gives
`REAL` (8-byte float either way — zero SQLite behavior change), while
Postgres gets genuine `double precision` (float8) instead of `real` (float4).
"""
import os
import sqlite3

import pytest

from dashboard import db, journal_store

pg = bool(os.environ.get("PG_DSN"))

# A 15-significant-digit float that a 32-bit float4 (Postgres `real`) cannot
# represent exactly (float4 has ~7 significant decimal digits) but a 64-bit
# float8 (`double precision`) preserves exactly via IEEE-754 round-trip.
HIGH_PRECISION_VALUE = 0.123456789012345


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_postgres_double_precision_round_trip_preserves_high_precision_float(monkeypatch):
    """RED before the fix: with `duration_seconds REAL`, Postgres stores it as
    `real` (float4) and the value read back would be rounded to ~7 significant
    digits (e.g. 0.12345679), NOT equal to the input. GREEN after the fix:
    `duration_seconds DOUBLE PRECISION` is float8 and the value survives the
    round trip exactly."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("/data/journal_entries.db")
    try:
        # Force re-creation so the table picks up the current (fixed) column
        # type even if a stale float4 version of it pre-exists in migtest.
        cx.execute("DROP TABLE IF EXISTS journal_entries")
        cx.commit()
        journal_store.init_table(cx)

        record = {c: None for c in journal_store._ALL_COLS}
        record["user_id"] = "c4-test-user"
        record["recorded_at"] = "2026-07-21T00:00:00Z"
        record["duration_seconds"] = HIGH_PRECISION_VALUE
        saved = journal_store.insert(cx, record)
        new_id = saved[0]["id"]

        row = cx.execute(
            "SELECT duration_seconds FROM journal_entries WHERE id = ?", (new_id,)
        ).fetchone()
        readback = row[0] if not hasattr(row, "keys") else row["duration_seconds"]

        assert readback == HIGH_PRECISION_VALUE, (
            f"expected exact float8 round-trip of {HIGH_PRECISION_VALUE!r}, "
            f"got {readback!r} — column is still float4 (REAL) truncating precision"
        )
    finally:
        cx.execute("DROP TABLE IF EXISTS journal_entries")
        cx.commit()
        cx.close()


# ---------------------------------------------------------------------------
# Source-declaration regression guard: the edited modules must declare
# DOUBLE PRECISION for the money/qty/dose/tax columns, not REAL, and must not
# have regressed back to a bare `<col> REAL` column-type token. This does NOT
# touch pgcompat.py — the fix is at the source DDL, never a translator regex.
# ---------------------------------------------------------------------------

_EXPECTED = {
    "dashboard/sourcing.py": ["price DOUBLE PRECISION", "moq DOUBLE PRECISION",
                              "confidence DOUBLE PRECISION"],
    "dashboard/purchase_orders.py": ["tax DOUBLE PRECISION", "shipping_amount DOUBLE PRECISION",
                                     "qty DOUBLE PRECISION", "qty_left DOUBLE PRECISION",
                                     "cost DOUBLE PRECISION", "qty_received DOUBLE PRECISION"],
    "dashboard/ingredient_catalog.py": ["price_per_unit DOUBLE PRECISION",
                                        "unit_size DOUBLE PRECISION",
                                        "shipping_quote DOUBLE PRECISION",
                                        "minimum_order DOUBLE PRECISION"],
    "dashboard/formulations.py": ["dose DOUBLE PRECISION"],
    "dashboard/materials_catalog.py": ["price DOUBLE PRECISION", "purchase_size DOUBLE PRECISION"],
    "dashboard/production.py": ["quantity_units DOUBLE PRECISION", "qty_used DOUBLE PRECISION"],
    "dashboard/journal_store.py": ["duration_seconds DOUBLE PRECISION"],
    "dashboard/formulation_map.py": ["score_min DOUBLE PRECISION"],
    "dashboard/testimonial_invites.py": ["confidence DOUBLE PRECISION"],
    "dashboard/inventory.py": ["qty DOUBLE PRECISION"],
}

# The bare `<col> REAL` column-type token that must NOT appear anymore for
# each changed column (a regression would put this back).
_OLD_TOKENS = {
    "dashboard/sourcing.py": ["price REAL", "moq REAL", "confidence REAL"],
    "dashboard/purchase_orders.py": ["tax REAL", "shipping_amount REAL", "qty REAL",
                                     "qty_left REAL", "cost REAL", "qty_received REAL"],
    "dashboard/ingredient_catalog.py": ["price_per_unit REAL", "unit_size REAL",
                                        "shipping_quote REAL", "minimum_order REAL"],
    "dashboard/formulations.py": ["dose REAL"],
    "dashboard/materials_catalog.py": ["price REAL", "purchase_size REAL"],
    "dashboard/production.py": ["quantity_units REAL", "qty_used REAL"],
    "dashboard/journal_store.py": ["duration_seconds REAL"],
    "dashboard/formulation_map.py": ["score_min REAL"],
    "dashboard/testimonial_invites.py": ["confidence REAL"],
    "dashboard/inventory.py": ["qty REAL"],
}


@pytest.mark.parametrize("relpath", sorted(_EXPECTED))
def test_source_declares_double_precision_not_real(relpath):
    text = open(relpath, encoding="utf-8").read()
    for needle in _EXPECTED[relpath]:
        assert needle in text, f"{relpath} missing expected {needle!r}"
    for stale in _OLD_TOKENS[relpath]:
        assert stale not in text, f"{relpath} still declares stale {stale!r}"


def test_pgcompat_not_touched_by_this_fix():
    """The fix must be a source-DDL edit, never a REAL-word regex added to
    pgcompat.py's DDL idiom translator (that would corrupt string-literal DATA
    containing the word REAL)."""
    text = open("dashboard/pgcompat.py", encoding="utf-8").read()
    assert "DOUBLE PRECISION" not in text, (
        "pgcompat.py should not need a DOUBLE PRECISION/REAL translation rule — "
        "the fix belongs at the source DDL, not the translator"
    )


# ---------------------------------------------------------------------------
# SQLite regression: the edited modules must still create their tables fine
# on SQLite, and a float still round-trips exactly (SQLite gives `DOUBLE
# PRECISION` the same REAL type affinity/8-byte storage it gave `REAL`).
# ---------------------------------------------------------------------------

def test_sqlite_journal_store_schema_and_round_trip_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    db_path = str(tmp_path / "journal.db")
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row
    try:
        journal_store.init_table(cx)
        record = {c: None for c in journal_store._ALL_COLS}
        record["user_id"] = "sqlite-test-user"
        record["recorded_at"] = "2026-07-21T00:00:00Z"
        record["duration_seconds"] = HIGH_PRECISION_VALUE
        saved = journal_store.insert(cx, record)
        new_id = saved[0]["id"]
        row = cx.execute(
            "SELECT duration_seconds FROM journal_entries WHERE id = ?", (new_id,)
        ).fetchone()
        assert row["duration_seconds"] == HIGH_PRECISION_VALUE
    finally:
        cx.close()


@pytest.mark.parametrize("modname,initfn", [
    ("dashboard.sourcing", "init_sourcing_schema"),
    ("dashboard.purchase_orders", "init_purchase_orders_schema"),
    ("dashboard.formulations", None),
    ("dashboard.materials_catalog", None),
    ("dashboard.production", "init_production_schema"),
    ("dashboard.formulation_map", "init_tables"),
    ("dashboard.testimonial_invites", "init_table"),
    ("dashboard.inventory", "init_inventory_schema"),
])
def test_sqlite_edited_modules_still_import_cleanly(modname, initfn):
    """Importing each edited module must still succeed on SQLite (unset
    DB_BACKEND) — DOUBLE PRECISION is valid SQLite column-type syntax
    (REAL affinity), so nothing here should raise at import time."""
    import importlib
    mod = importlib.import_module(modname)
    if initfn:
        assert hasattr(mod, initfn)
