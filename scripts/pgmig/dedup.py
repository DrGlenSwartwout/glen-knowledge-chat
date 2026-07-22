"""Scan a SQLite source for rows that would violate Postgres UNIQUE constraints."""
import sqlite3
from typing import List, Dict
from scripts.pgmig import introspect

def scan_collisions(cx, table: str, cols: List[str]) -> Dict:
    """Scan `table` for rows whose (cols) key already repeats -- a would-be
    UNIQUE-constraint collision.

    NULLs are excluded from the scan: SQL treats NULL as DISTINCT from any
    other NULL under a UNIQUE index/constraint (both SQLite and Postgres, by
    default), so a group of rows that all share a NULL key value does NOT
    actually collide. A row with ANY NULL key column would be excluded from
    the UNIQUE constraint entirely, so it must also be excluded here -- else
    the scan reports false-positive "collisions" the real cutover would never
    hit (this hit `subscriptions.order_ref` and `todos.dedup_key` in the
    2026-07-22 dry-run).
    """
    collist = ", ".join(f'"{c}"' for c in cols)
    not_null = " AND ".join(f'"{c}" IS NOT NULL' for c in cols)
    rows = cx.execute(
        f'SELECT {collist}, COUNT(*) c FROM "{table}" WHERE {not_null} '
        f'GROUP BY {collist} HAVING c > 1'
    ).fetchall()
    n_groups = len(rows)
    n_excess = sum(r[-1] - 1 for r in rows)
    examples = [{"key": list(r[:-1]), "count": r[-1]} for r in rows[:5]]
    return {"key_cols": cols, "n_groups": n_groups, "n_excess_rows": n_excess, "examples": examples}

def scan_db(sqlite_path: str) -> List[Dict]:
    """Scan every table of `sqlite_path` for collisions on ITS OWN unique
    indexes (introspect.unique_indexes). Findings are tagged source='sqlite'
    to distinguish them, when merged, from scan_against_targets' 'pg-target'
    findings -- see that function's docstring for why the two are both
    needed (this one alone can miss a real hazard)."""
    # M5: read-only -- a dedup scan of the source must not be able to mutate it.
    cx = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        findings = []
        for t in introspect.sqlite_tables(cx):
            for cols in introspect.unique_indexes(cx, t):
                r = scan_collisions(cx, t, cols)
                if r["n_groups"]:
                    findings.append({"table": t, "source": "sqlite", **r})
        return findings
    finally:
        cx.close()


def scan_against_targets(sqlite_path: str, target_map: Dict[str, List[List[str]]]) -> List[Dict]:
    """Cross-check the SQLite SOURCE against a POSTGRES TARGET's unique-index
    column sets (`target_map`: table -> list of column-tuples, typically
    `introspect.pg_unique_indexes(pg_cx, schema)`).

    Why this exists (a real cutover hazard): `scan_db` only scans columns
    that ALREADY have a UNIQUE index in the SQLite source. If a UNIQUE index
    silently failed to build on SQLite (CREATE UNIQUE INDEX fails outright
    when duplicate rows already exist, and app code may swallow that error),
    the index is simply ABSENT from SQLite -- `scan_db` has no way to know
    those columns were ever meant to be unique, so it never looks at them.
    But a fresh Postgres target schema (built from current DDL, not from
    SQLite's possibly-degraded history) DOES have that unique index, so the
    copy fails or silently loses rows on it. Cross-checking the SQLite data
    against the TARGET's actual unique indexes catches this before any write.

    Only scans tables the SQLite source actually has (a target-only table has
    no source rows to collide). Findings are tagged source='pg-target'.
    """
    # M5: read-only -- a dedup scan of the source must not be able to mutate it.
    cx = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        sqlite_tables = set(introspect.sqlite_tables(cx))
        findings = []
        for table, col_sets in target_map.items():
            if table not in sqlite_tables:
                continue
            for cols in col_sets:
                r = scan_collisions(cx, table, cols)
                if r["n_groups"]:
                    findings.append({"table": table, "source": "pg-target", **r})
        return findings
    finally:
        cx.close()
