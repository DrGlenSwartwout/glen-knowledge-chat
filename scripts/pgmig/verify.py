"""Row-count parity check: SQLite source vs its Postgres target schema.

Kept simple for v1 (per the P05 task-2 brief): compare raw per-table counts
and flag mismatches for operator review. A table present on only one side
counts as a mismatch (its missing-side count is None).
"""
import sqlite3
from typing import Dict, List

from dashboard import db
from dashboard.dbschema import schema_for_path
from scripts.pgmig import introspect


def parity(sqlite_path: str) -> List[Dict]:
    schema = schema_for_path(sqlite_path)
    sqlite_cx = sqlite3.connect(sqlite_path)
    try:
        sqlite_tables = set(introspect.sqlite_tables(sqlite_cx))
        pg_cx = db.connect(sqlite_path)
        try:
            pg_tables = set(introspect.pg_fk_order(pg_cx, schema))
            results = []
            for t in sorted(sqlite_tables | pg_tables):
                s_count = (sqlite_cx.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                           if t in sqlite_tables else None)
                p_count = (pg_cx.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                           if t in pg_tables else None)
                ok = s_count is not None and p_count is not None and s_count == p_count
                results.append({"table": t, "sqlite": s_count, "postgres": p_count, "ok": ok})
            return results
        finally:
            pg_cx.close()
    finally:
        sqlite_cx.close()


def all_ok(results: List[Dict]) -> bool:
    return all(r["ok"] for r in results)
