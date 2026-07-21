"""Portable write helpers that keep call sites backend-agnostic."""
from typing import Sequence

def insert_or_ignore(cx, table: str, columns: Sequence[str], values: Sequence,
                     *, conflict_cols: Sequence[str]) -> None:
    """Insert a row, ignoring it if it collides on `conflict_cols` (idempotent).
    SQLite: INSERT OR IGNORE. Postgres: INSERT ... ON CONFLICT (...) DO NOTHING,
    which REQUIRES a unique index on conflict_cols — if it is missing, Postgres
    raises SQLSTATE 42P10; we convert that into a clear, actionable RuntimeError.
    `table`/`columns`/`conflict_cols` are code literals (not user input)."""
    from dashboard import db
    cols_sql = ",".join(columns)
    placeholders = ",".join(["?"] * len(columns))
    if db.backend_of(cx) == "postgres":
        conflict = ",".join(conflict_cols)
        sql = (f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
               f"ON CONFLICT ({conflict}) DO NOTHING")
        try:
            cx.execute(sql, tuple(values))
        except Exception as e:  # noqa: BLE001
            if getattr(e, "sqlstate", None) == "42P10":
                raise RuntimeError(
                    f"insert_or_ignore: table {table}({','.join(conflict_cols)}) has no unique "
                    f"index matching the ON CONFLICT target — create it (dedup rows first if "
                    f"needed) before this write path runs on Postgres") from e
            raise
    else:
        sql = f"INSERT OR IGNORE INTO {table} ({cols_sql}) VALUES ({placeholders})"
        cx.execute(sql, tuple(values))
