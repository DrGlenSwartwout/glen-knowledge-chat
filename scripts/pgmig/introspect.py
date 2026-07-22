"""Introspect a SQLite source: user tables and their UNIQUE index column-tuples."""
from typing import List

def sqlite_tables(cx) -> List[str]:
    rows = cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]

def unique_indexes(cx, table: str) -> List[List[str]]:
    out = []
    for _seq, name, unique, _origin, _partial in cx.execute(
            f"PRAGMA index_list('{table}')").fetchall():
        if not unique:
            continue
        cols = [r[2] for r in cx.execute(f"PRAGMA index_info('{name}')").fetchall()]
        if cols:
            out.append(cols)
    return out
