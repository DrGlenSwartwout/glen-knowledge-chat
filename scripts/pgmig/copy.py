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
                except Exception:
                    # If even the ROLLBACK TO fails, the connection's transaction
                    # is in an unrecoverable state -- bail out of the whole
                    # fallback loop rather than risk silently losing further rows.
                    try:
                        pg_cx.rollback()
                    except Exception:
                        pass
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
    sqlite_cx = sqlite3.connect(sqlite_path)
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
