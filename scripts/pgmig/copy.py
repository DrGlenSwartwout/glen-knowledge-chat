"""Copy a live SQLite source into its Postgres target schema, parent-before-child.

Read-only on SQLite; writes Postgres via the shipped adapter (dashboard.db),
whose connection already has search_path set to the schema derived from the
db_path's basename (dashboard.dbschema.schema_for_path). Inserts use
`ON CONFLICT DO NOTHING` as a backstop for any residual duplicates the P05
dedup preflight (scripts/pgmig/dedup.py) surfaced but didn't remove.
"""
import sqlite3
from typing import Dict, List

from dashboard import db
from dashboard.dbschema import schema_for_path
from scripts.pgmig import introspect


def copy_table(sqlite_cx, pg_cx, table: str) -> Dict:
    """Copy every row of `table` from `sqlite_cx` into `pg_cx`'s current
    schema. Returns {"table","source_rows","inserted","conflicts"} and,
    if any individual row failed to insert (not just an ON CONFLICT skip),
    an "errors" list of {"row","error"} so the whole table isn't aborted
    over one bad row."""
    cols = [d[0] for d in sqlite_cx.execute(f'SELECT * FROM "{table}" LIMIT 0').description]
    rows = sqlite_cx.execute(f'SELECT * FROM "{table}"').fetchall()
    source_rows = len(rows)
    result = {"table": table, "source_rows": source_rows, "inserted": 0, "conflicts": 0}
    if source_rows == 0:
        return result

    collist = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    sql = f'INSERT INTO "{table}" ({collist}) VALUES ({placeholders}) ON CONFLICT DO NOTHING'

    inserted = 0
    errors: List[Dict] = []
    try:
        cur = pg_cx.executemany(sql, [tuple(r) for r in rows])
        inserted = cur.rowcount or 0
    except Exception:
        # One bad row aborts the whole batch on Postgres (the transaction goes
        # into an error state until rolled back) -- roll back and fall back to
        # per-row inserts so a single dialect/type mismatch doesn't sink the
        # rest of an otherwise-clean table.
        try:
            pg_cx.rollback()
        except Exception:
            pass
        inserted = 0
        # Per-row fallback shares ONE transaction with the caller (copy_all commits
        # once per table) -- without isolation, a later bad row's rollback() would
        # discard every good row this loop already inserted earlier in the SAME
        # pass, since they're all still uncommitted in that one transaction. A
        # SAVEPOINT per row isolates each row: a failure rolls back ONLY to that
        # row's savepoint, leaving previously-released (successful) rows intact
        # and still pending commit at the end of copy_table/copy_all.
        for row in rows:
            try:
                pg_cx.execute("SAVEPOINT pgmig_row")
                r = pg_cx.execute(sql, tuple(row))
                row_inserted = r.rowcount or 0
                pg_cx.execute("RELEASE SAVEPOINT pgmig_row")
                inserted += row_inserted
            except Exception as row_exc:
                try:
                    pg_cx.execute("ROLLBACK TO SAVEPOINT pgmig_row")
                    # RELEASE the savepoint once we've rolled back to it, same as
                    # the success path does -- otherwise savepoints of the same
                    # name (pgmig_row) accumulate, unreleased, across every
                    # consecutive failing row in this loop.
                    pg_cx.execute("RELEASE SAVEPOINT pgmig_row")
                except Exception:
                    # If even the ROLLBACK TO fails, the connection's transaction
                    # is in an unrecoverable state -- bail out of the whole
                    # fallback loop rather than risk silently losing further rows.
                    # M3: a full pg_cx.rollback() here undoes the WHOLE
                    # transaction, including every row this loop already
                    # inserted+released earlier in the SAME pass (none of them
                    # were committed yet -- copy_all commits once per table,
                    # after copy_table returns) -- so none of them actually
                    # persisted. Reset `inserted` to 0 to reflect that reality
                    # rather than report a stale non-zero count for rows that no
                    # longer exist in Postgres. `conflicts` may still overcount
                    # in this corner (rows that were rolled back, not genuine
                    # ON CONFLICT skips), but `errors` is guaranteed non-empty
                    # here, so any_errors()/the CLI's exit code still flag the
                    # table as failed either way. This only triggers on a
                    # broken/unrecoverable connection.
                    try:
                        pg_cx.rollback()
                    except Exception:
                        pass
                    inserted = 0
                    errors.append({"row": list(row), "error": str(row_exc)})
                    break
                errors.append({"row": list(row), "error": str(row_exc)})

    result["inserted"] = inserted
    result["conflicts"] = source_rows - inserted - len(errors)
    if errors:
        result["errors"] = errors
    return result


def any_errors(results: List[Dict]) -> bool:
    """True if any per-table result dict carries a non-empty 'errors' list --
    i.e. at least one row genuinely failed to persist (not merely an
    ON CONFLICT DO NOTHING skip, which lands in 'conflicts' instead). The CLI
    gates its exit status on this so a failed row can't pass unnoticed."""
    return any(r.get("errors") for r in results)


def _qident(name: str) -> str:
    """Double-quote a Postgres identifier, escaping any embedded double quote
    (`"` -> `""`, the standard Postgres quoted-identifier escape)."""
    return '"' + name.replace('"', '""') + '"'


def _sql_string_literal(s: str) -> str:
    """Single-quote a SQL string literal, escaping any embedded single quote
    (`'` -> `''`)."""
    return "'" + s.replace("'", "''") + "'"


def reset_identity_sequences(pg_cx, schema: str, table: str) -> None:
    """Reset every IDENTITY column's sequence on `schema.table` to
    max(id)+1, for each column `information_schema.columns` reports as
    `is_identity='YES'`.

    Why: Task 2's `GENERATED BY DEFAULT AS IDENTITY` lets `copy_table` write
    the migrated rows' ORIGINAL explicit ids (preserving FK relationships),
    but an explicit-id INSERT does not advance the identity sequence itself.
    Left unreset, the sequence would still be at its start after copying ids
    1..N, so the app's next auto-generated `INSERT ... RETURNING id` (no
    explicit id) would produce a colliding id. This must run after every
    table copy to keep the sequence ahead of the highest id actually loaded.

    `setval(seq, value, is_called=false)` means `value` itself (not
    value + 1) is the next value `nextval()` returns -- so passing
    `max(id) + 1` directly makes the next auto-generated id equal
    `max(id) + 1`, exactly what's wanted. An empty table has `max(id) IS
    NULL`, so `COALESCE(..., 0) + 1` falls back to `1` (first auto-gen id is
    1, no crash). A table with no identity column at all is a no-op (the
    `information_schema.columns` lookup returns no rows, so the loop below
    never runs).

    Identifier-safe: `schema`/`table`/each `col` are quoted (embedded `"`
    doubled) wherever used as identifiers, and the qualified name passed to
    `pg_get_serial_sequence` (itself a string-literal argument) is built from
    those quoted identifiers and then string-literal-escaped (embedded `'`
    doubled) as a whole -- so a schema/table/column name containing a quote
    character can't break out of either the identifier or the string-literal
    context.
    """
    cols = pg_cx.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = ? AND table_name = ? AND is_identity = 'YES'",
        (schema, table)).fetchall()
    qualified_table = f"{_qident(schema)}.{_qident(table)}"
    for row in cols:
        col = row[0]
        seq_arg = _sql_string_literal(qualified_table)
        col_arg = _sql_string_literal(col)
        pg_cx.execute(
            f"SELECT setval(pg_get_serial_sequence({seq_arg}, {col_arg}), "
            f"COALESCE((SELECT max({_qident(col)}) FROM {qualified_table}), 0) + 1, false)"
        )


def copy_all(sqlite_path: str, *, truncate: bool = False) -> List[Dict]:
    """Copy every table of `sqlite_path` into its Postgres target schema
    (schema_for_path(sqlite_path), via db.connect(sqlite_path)), in FK
    parent-before-child order. Tables present on only one side are reported
    (not silently skipped) with a "note" explaining why they weren't copied."""
    if db.backend() != "postgres":
        raise RuntimeError(
            "copy_all requires DB_BACKEND=postgres (an ops tool must not run "
            "information_schema introspection or DDL against a SQLite handle); "
            "got DB_BACKEND=%r" % db.backend())
    schema = schema_for_path(sqlite_path)
    # Money-grade run against a frozen snapshot: open the SOURCE truly
    # read-only so this tool cannot mutate it (or its WAL) even by accident.
    # The Postgres target (db.connect below) is unaffected -- it's written on
    # purpose.
    sqlite_cx = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        sqlite_tables = set(introspect.sqlite_tables(sqlite_cx))
        pg_cx = db.connect(sqlite_path)
        try:
            order = introspect.pg_fk_order(pg_cx, schema)

            if truncate:
                for t in reversed(order):
                    pg_cx.execute(f'TRUNCATE "{t}" CASCADE')
                pg_cx.commit()

            results = []
            for t in order:
                if t in sqlite_tables:
                    r = copy_table(sqlite_cx, pg_cx, t)
                    if not r.get("errors"):
                        # Reset AFTER a successful (or error-free) copy only --
                        # a table that hit a row error may be partially loaded,
                        # which is a fine base for max(id)+1 (still correct),
                        # but skip it anyway per spec: don't touch a table's
                        # sequence off the back of a copy that itself failed
                        # loudly and needs operator attention first.
                        reset_identity_sequences(pg_cx, schema, t)
                    pg_cx.commit()
                else:
                    r = {"table": t, "source_rows": 0, "inserted": 0, "conflicts": 0,
                         "note": "no matching sqlite source table"}
                results.append(r)

            for t in sorted(sqlite_tables - set(order)):
                n = sqlite_cx.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                results.append({"table": t, "source_rows": n, "inserted": 0, "conflicts": 0,
                                 "note": "no matching postgres target table"})

            return results
        finally:
            pg_cx.close()
    finally:
        sqlite_cx.close()
