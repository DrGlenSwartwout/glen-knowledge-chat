"""Scan a SQLite source for rows that would violate Postgres UNIQUE constraints."""
import sqlite3
from typing import List, Dict
from scripts.pgmig import introspect

def scan_collisions(cx, table: str, cols: List[str]) -> Dict:
    collist = ", ".join(f'"{c}"' for c in cols)
    rows = cx.execute(
        f'SELECT {collist}, COUNT(*) c FROM "{table}" GROUP BY {collist} HAVING c > 1'
    ).fetchall()
    n_groups = len(rows)
    n_excess = sum(r[-1] - 1 for r in rows)
    examples = [{"key": list(r[:-1]), "count": r[-1]} for r in rows[:5]]
    return {"key_cols": cols, "n_groups": n_groups, "n_excess_rows": n_excess, "examples": examples}

def scan_db(sqlite_path: str) -> List[Dict]:
    cx = sqlite3.connect(sqlite_path)
    try:
        findings = []
        for t in introspect.sqlite_tables(cx):
            for cols in introspect.unique_indexes(cx, t):
                r = scan_collisions(cx, t, cols)
                if r["n_groups"]:
                    findings.append({"table": t, **r})
        return findings
    finally:
        cx.close()
