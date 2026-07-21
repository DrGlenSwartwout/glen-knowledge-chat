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


def insert_or_replace(cx, table: str, columns: Sequence[str], values: Sequence,
                      *, conflict_cols: Sequence[str]) -> None:
    """Upsert a row: insert it, or overwrite the existing row that collides on
    `conflict_cols`. SQLite: INSERT OR REPLACE. Postgres: INSERT ... ON CONFLICT
    (conflict_cols) DO UPDATE SET <every non-conflict column>=EXCLUDED.<col>.

    Semantics match INSERT OR REPLACE for the common case where the INSERT supplies
    the FULL row (all ported call sites do): both end with the new values in every
    supplied column. The only difference — a table column ABSENT from `columns`
    would be reset to its default/NULL by INSERT OR REPLACE but KEPT by ON CONFLICT
    DO UPDATE — does not arise here. INSERT OR REPLACE fires on ANY unique
    constraint; this targets the ONE key each caller passes explicitly (e.g.
    inquiry_reply_tokens dedups on its UNIQUE(inquiry_id, practitioner_id), not its
    random token_hash PK). Requires a unique index on conflict_cols on Postgres.
    `table`/`columns`/`conflict_cols` are code literals, not user input."""
    from dashboard import db
    cols_sql = ",".join(columns)
    placeholders = ",".join(["?"] * len(columns))
    if db.backend_of(cx) == "postgres":
        conflict = ",".join(conflict_cols)
        keyset = set(conflict_cols)
        setcols = [c for c in columns if c not in keyset]
        if setcols:
            set_sql = ",".join(f"{c}=EXCLUDED.{c}" for c in setcols)
            sql = (f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
                   f"ON CONFLICT ({conflict}) DO UPDATE SET {set_sql}")
        else:
            # every supplied column is part of the key -> nothing to overwrite
            sql = (f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
                   f"ON CONFLICT ({conflict}) DO NOTHING")
        cx.execute(sql, tuple(values))
    else:
        sql = f"INSERT OR REPLACE INTO {table} ({cols_sql}) VALUES ({placeholders})"
        cx.execute(sql, tuple(values))


def insert_returning_id(cx, sql: str, params=(), *, pk: str = "id"):
    """Run a single INSERT and return the new row's autoincrement/IDENTITY id.
    SQLite: execute + cur.lastrowid. Postgres has no lastrowid, so append
    `RETURNING <pk>` and read it back. `pk` (default 'id') is the autoincrement
    column — a code literal, not user input. `sql` must be ONE INSERT statement
    with no existing RETURNING clause. Returns None if no row was inserted (e.g.
    an ON CONFLICT that did nothing on Postgres — no call site here does that)."""
    from dashboard import db
    if db.backend_of(cx) == "postgres":
        stmt = sql.rstrip()
        if stmt.endswith(";"):
            stmt = stmt[:-1].rstrip()
        row = cx.execute(f"{stmt} RETURNING {pk}", params).fetchone()
        return row[0] if row else None
    return cx.execute(sql, params).lastrowid
