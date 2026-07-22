"""Introspect a SQLite source: user tables and their UNIQUE index column-tuples.
Also introspects a Postgres target schema's FK graph for topological table order,
and (for the P05 preflight cross-check) the target's own UNIQUE indexes."""
from collections import deque
from typing import Dict, List

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


def pg_fk_order(pg_cx, schema: str) -> List[str]:
    """Tables of `schema` topologically sorted parent -> child (Kahn's algorithm
    over information_schema FK edges). A table with no FKs at all sorts first
    (alongside any other FK-free table). Self-referential FKs (a table
    referencing itself) don't constrain table-level order and are ignored --
    all of a table's own rows are copied together in one pass regardless.
    Raises RuntimeError naming the offending tables if a genuine cross-table
    cycle prevents a total order."""
    tables = [r[0] for r in pg_cx.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = ? AND table_type = 'BASE TABLE' ORDER BY table_name",
        (schema,)).fetchall()]

    edge_rows = pg_cx.execute(
        "SELECT DISTINCT tc.table_name AS child_table, ccu.table_name AS parent_table "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "  ON tc.constraint_name = kcu.constraint_name "
        " AND tc.constraint_schema = kcu.constraint_schema "
        "JOIN information_schema.constraint_column_usage ccu "
        "  ON tc.constraint_name = ccu.constraint_name "
        " AND tc.constraint_schema = ccu.constraint_schema "
        "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = ?",
        (schema,)).fetchall()

    indegree = {t: 0 for t in tables}
    children = {t: set() for t in tables}
    seen = set()
    for row in edge_rows:
        child, parent = row[0], row[1]
        if child == parent:
            continue  # self-referential FK -- not a table-level ordering constraint
        if child not in indegree or parent not in indegree:
            continue  # defensive: constraint referencing a table outside this schema listing
        key = (parent, child)
        if key in seen:
            continue
        seen.add(key)
        children[parent].add(child)
        indegree[child] += 1

    queue = deque(sorted(t for t in tables if indegree[t] == 0))
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(order) != len(tables):
        remaining = sorted(set(tables) - set(order))
        raise RuntimeError(f"FK cycle among {remaining}")
    return order


def pg_unique_indexes(pg_cx, schema: str) -> Dict[str, List[List[str]]]:
    """Map table -> list of UNIQUE-index column-tuples for every unique index
    (including ones backing a UNIQUE/PK constraint AND bare `CREATE UNIQUE
    INDEX`) in `schema`, ordered by each index's column position.

    Why pg_index/pg_class/pg_attribute rather than information_schema: a bare
    `CREATE UNIQUE INDEX ux_foo ON t (col)` (no ADD CONSTRAINT) shows up in
    pg_index but NOT in information_schema.table_constraints -- and this is
    exactly the case the P05 preflight cross-check exists to catch (see
    dedup.scan_against_targets). information_schema alone would silently miss
    it, defeating the point of the enhancement.

    `idx.indkey::int2[]` casts the raw int2vector column list so it can be
    unnest()'d WITH ORDINALITY to preserve column order; expression indexes
    (an indkey entry of 0, i.e. no backing column) are excluded -- they're
    out of scope for a byte-level SQLite dedup scan, which only knows about
    literal columns.
    """
    rows = pg_cx.execute(
        "SELECT tbl.relname AS table_name, idx.indexrelid AS index_oid, "
        "att.attname AS col_name "
        "FROM pg_index idx "
        "JOIN pg_class tbl ON tbl.oid = idx.indrelid "
        "JOIN pg_namespace ns ON ns.oid = tbl.relnamespace "
        "JOIN LATERAL unnest(idx.indkey::int2[]) WITH ORDINALITY AS ord(attnum, ordinality) ON true "
        "JOIN pg_attribute att ON att.attrelid = tbl.oid AND att.attnum = ord.attnum "
        "WHERE idx.indisunique AND ns.nspname = ? AND tbl.relkind = 'r' "
        "AND NOT (0 = ANY(idx.indkey::int2[])) "
        "ORDER BY tbl.relname, idx.indexrelid, ord.ordinality",
        (schema,)).fetchall()

    out: Dict[str, List[List[str]]] = {}
    current_key = None
    current_cols: List[str] = []
    for r in rows:
        table, index_oid, col = r[0], r[1], r[2]
        key = (table, index_oid)
        if key != current_key:
            if current_key is not None:
                out.setdefault(current_key[0], []).append(current_cols)
            current_key = key
            current_cols = []
        current_cols.append(col)
    if current_key is not None:
        out.setdefault(current_key[0], []).append(current_cols)
    return out
