"""Build the Postgres schema directly from a SOURCE SQLite's own `CREATE TABLE`/
`CREATE INDEX` statements (`sqlite_master`), instead of via `import app` (which
creates only the ~97 tables the app touches at import time -- 129 of prod's 226
tables are created lazily at runtime and would otherwise be missing entirely).

Pipeline for each table's DDL (`translate_ddl`):
  1. `dashboard.pgcompat.translate_sql` -- the mechanical idioms already used on
     every live query/DDL (AUTOINCREMENT -> IDENTITY, datetime('now'), etc).
  2. DDL-only `REAL` -> `DOUBLE PRECISION` -- safe HERE (CREATE TABLE text has no
     data literals), unlike the conservative runtime query-translation path.
  3. Force every column named in `text_cols` to `TEXT` -- data-driven widening
     for SQLite's loose typing (an INTEGER-declared column that actually holds
     non-integer data, e.g. `testimonial_tokens` holding UUID strings).
  4. Strip FK constraints entirely (column-level `REFERENCES ...` and
     table-level `FOREIGN KEY (...) REFERENCES ...`) -- see the design-decision
     note on `create_schema` for why.

Read-only on SQLite throughout (source opened `mode=ro`); all Postgres access
goes through `dashboard.db` (whose connection's search_path is already the
schema derived from the source path's basename -- see `dashboard.dbschema`).
"""
import re
import sqlite3
from typing import Dict, List, Optional, Set, Tuple

from dashboard import db, pgcompat

# ---------------------------------------------------------------------------
# translate_ddl and its helpers
# ---------------------------------------------------------------------------

# DDL-only: safe because this runs solely on CREATE TABLE/INDEX text pulled from
# sqlite_master, never on a data literal (unlike the runtime query-translation
# path in pgcompat.translate_sql, which must stay conservative around values).
_RE_REAL = re.compile(r"(?i)\bREAL\b")

# A leading identifier token -- double-quoted, backtick-quoted, bracket-quoted,
# or bare -- capturing any leading whitespace along with it, plus everything
# that follows (the type + column-constraints). DOTALL so a multi-line CREATE
# TABLE (common in this codebase's triple-quoted DDL strings) still matches.
_RE_LEADING_NAME = re.compile(
    r'^(\s*(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))(.*)$', re.DOTALL)

# A table-level constraint item starts with (optionally) a CONSTRAINT name,
# then one of PRIMARY KEY / FOREIGN KEY / UNIQUE / CHECK followed immediately
# by '('. This is what distinguishes a table-constraint item from a column
# definition -- a column definition never has one of these keywords as its
# OWN name followed directly by '(' in this exact shape, so a column literally
# named e.g. "check" (declared "check TEXT") does not match (no '(' follows).
_RE_TABLE_CONSTRAINT_START = re.compile(
    r'(?i)^\s*(?:CONSTRAINT\s+(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|\w+)\s+)?'
    r'(PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK)\s*\(')

# Inline column-level FK: `REFERENCES <table>[(<cols>)] [ON DELETE ...]
# [ON UPDATE ...] [[NOT] DEFERRABLE [INITIALLY ...]]`. Applied only to the
# `rest` of a column item (the text AFTER the column name has already been
# peeled off by _RE_LEADING_NAME), so a column literally NAMED "references"
# is never touched -- the regex never sees the name token at all.
_RE_INLINE_REFERENCES = re.compile(
    r'(?i)\bREFERENCES\s+(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|\w+)\s*(?:\([^)]*\))?'
    r'(?:\s*ON\s+(?:DELETE|UPDATE)\s+'
    r'(?:CASCADE|RESTRICT|SET\s+NULL|SET\s+DEFAULT|NO\s+ACTION))*'
    r'(?:\s*(?:NOT\s+)?DEFERRABLE(?:\s+INITIALLY\s+(?:DEFERRED|IMMEDIATE))?)?'
)

# The leading type token of a column's `rest` (e.g. "INTEGER", "VARCHAR(50)") --
# used to force-override a loose-int column's declared type to TEXT.
_RE_LEADING_TYPE = re.compile(r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*\([^)]*\))?')


def _logical_name(token: str) -> str:
    """Strip quoting (double/backtick/bracket) from an identifier token,
    for case-insensitive comparison against `text_cols`."""
    t = token.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        return t[1:-1]
    if len(t) >= 2 and t[0] == "`" and t[-1] == "`":
        return t[1:-1]
    if len(t) >= 2 and t[0] == "[" and t[-1] == "]":
        return t[1:-1]
    return t


def _split_top_level(s: str) -> List[str]:
    """Split `s` on commas at parenthesis-depth 0, treating single-quoted
    string literals and double-quoted identifiers as opaque (their contents,
    including any comma/paren inside, are never inspected)."""
    parts = []
    depth = 0
    buf = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "'":
            j = i + 1
            while j < n and s[j] != "'":
                j += 1
            j = min(j + 1, n)
            buf.append(s[i:j])
            i = j
            continue
        if ch == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 1
            j = min(j + 1, n)
            buf.append(s[i:j])
            i = j
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _find_column_list_span(sql: str) -> Tuple[int, int]:
    """Locate the outer `(...)` column/constraint list of a CREATE TABLE
    statement: returns (index of the opening '(', index of the matching
    closing ')'), skipping over any single- or double-quoted spans (so a
    quoted table/column name can't fool the scan) and honoring paren nesting."""
    i, n = 0, len(sql)
    open_idx = None
    while i < n:
        ch = sql[i]
        if ch == "'":
            j = i + 1
            while j < n and sql[j] != "'":
                j += 1
            i = j + 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and sql[j] != '"':
                j += 1
            i = j + 1
            continue
        if ch == "(":
            open_idx = i
            break
        i += 1
    if open_idx is None:
        raise ValueError("no column list found in CREATE TABLE statement")

    depth = 0
    i = open_idx
    close_idx = None
    while i < n:
        ch = sql[i]
        if ch == "'":
            j = i + 1
            while j < n and sql[j] != "'":
                j += 1
            i = j + 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and sql[j] != '"':
                j += 1
            i = j + 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            if depth == 0:
                close_idx = i - 1
                break
            continue
        i += 1
    if close_idx is None:
        raise ValueError("unbalanced parentheses in CREATE TABLE statement")
    return open_idx, close_idx


def _table_constraint_kind(item: str) -> Optional[str]:
    """None if `item` is a column definition; otherwise the table-level
    constraint keyword ('PRIMARY KEY' | 'FOREIGN KEY' | 'UNIQUE' | 'CHECK')."""
    m = _RE_TABLE_CONSTRAINT_START.match(item)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1).upper())


def _force_text_type(rest: str) -> str:
    """Replace the leading type token (+ optional `(...)` precision) of a
    column's post-name text with `TEXT`, preserving everything else
    (NOT NULL, DEFAULT, etc. -- and any leading whitespace)."""
    m = _RE_LEADING_TYPE.match(rest)
    if not m:
        return rest
    return rest[: m.start(2)] + "TEXT" + rest[m.end():]


def translate_ddl(table: str, sqlite_sql: str, *, text_cols: Optional[Set[str]] = None) -> str:
    """Turn one SQLite `CREATE TABLE` statement into Postgres DDL. Pure
    (no I/O). See module docstring for the 4-step pipeline."""
    text_cols_lower = {c.lower() for c in (text_cols or ())}

    sql = pgcompat.translate_sql(sqlite_sql)
    sql = _RE_REAL.sub("DOUBLE PRECISION", sql)

    open_idx, close_idx = _find_column_list_span(sql)
    prefix = sql[: open_idx + 1]
    content = sql[open_idx + 1 : close_idx]
    suffix = sql[close_idx:]

    new_items = []
    for item in _split_top_level(content):
        if not item.strip():
            continue
        kind = _table_constraint_kind(item)
        if kind is not None:
            if kind == "FOREIGN KEY":
                continue  # strip the whole table-level FK constraint
            new_items.append(item)  # PRIMARY KEY / UNIQUE / CHECK: keep as-is
            continue

        m = _RE_LEADING_NAME.match(item)
        if not m:
            new_items.append(item)  # defensive: unparseable item, pass through
            continue
        name_token, rest = m.group(1), m.group(2)
        rest = _RE_INLINE_REFERENCES.sub("", rest)  # strip column-level FK
        if _logical_name(name_token).lower() in text_cols_lower:
            rest = _force_text_type(rest)
        new_items.append(name_token + rest)

    return prefix + ", ".join(x.strip() for x in new_items) + suffix


# ---------------------------------------------------------------------------
# loose_int_text_cols
# ---------------------------------------------------------------------------

def loose_int_text_cols(sqlite_cx, table: str) -> Set[str]:
    """Columns declared INTEGER (per `PRAGMA table_info`) whose stored data
    holds at least one non-NULL, non-integer value (SQLite's own `typeof()`
    check -- the exact, data-driven test, not a heuristic). Read-only."""
    cols = sqlite_cx.execute(f'PRAGMA table_info("{table}")').fetchall()
    out: Set[str] = set()
    for row in cols:
        name = row[1]
        decl_type = (row[2] or "").strip().upper()
        if decl_type != "INTEGER":
            continue
        cnt = sqlite_cx.execute(
            f'SELECT count(*) FROM "{table}" '
            f'WHERE "{name}" IS NOT NULL AND typeof("{name}") <> \'integer\''
        ).fetchone()[0]
        if cnt > 0:
            out.add(name)
    return out


# ---------------------------------------------------------------------------
# create_schema
# ---------------------------------------------------------------------------

def create_schema(sqlite_path: str, *, drop_first: bool = False) -> Dict:
    """Build the full Postgres schema for `sqlite_path` from the source
    SQLite's own `sqlite_master` DDL -- every user table (`type='table'`,
    name NOT LIKE 'sqlite_%'), FK-free (design decision: avoids CREATE-order
    problems and faithfully copies possibly-orphaned SQLite data -- see the
    P05.5 plan's "Design decisions (locked)"), then the non-auto indexes.

    Returns {"tables_created", "indexes_created", "widened_cols": [(table,col)],
    "skipped": [...]}.
    """
    if db.backend() != "postgres":
        raise RuntimeError(
            "create_schema requires DB_BACKEND=postgres (an ops tool must not "
            "run DDL against a SQLite handle); got DB_BACKEND=%r" % db.backend())

    report: Dict = {
        "tables_created": 0,
        "indexes_created": 0,
        "widened_cols": [],
        "skipped": [],
    }

    sqlite_cx = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        if drop_first:
            from dashboard.dbschema import schema_for_path
            schema = schema_for_path(sqlite_path)
            with db.connect(sqlite_path) as pg_cx:
                pg_cx.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                pg_cx.execute(f'CREATE SCHEMA "{schema}"')
                pg_cx.commit()

        tables = [r[0] for r in sqlite_cx.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()]

        pg_cx = db.connect(sqlite_path)
        try:
            for table in tables:
                row = sqlite_cx.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,)).fetchone()
                if not row or not row[0]:
                    report["skipped"].append(
                        {"table": table, "reason": "no CREATE sql on source (virtual/internal table?)"})
                    continue
                sqlite_sql = row[0]
                text_cols = loose_int_text_cols(sqlite_cx, table)
                pg_sql = translate_ddl(table, sqlite_sql, text_cols=text_cols)
                try:
                    pg_cx.execute(pg_sql)
                except Exception as exc:
                    report["skipped"].append({"table": table, "reason": str(exc)})
                    continue
                report["tables_created"] += 1
                for col in sorted(text_cols):
                    report["widened_cols"].append((table, col))
            pg_cx.commit()

            index_rows = sqlite_cx.execute(
                "SELECT name, tbl_name, sql FROM sqlite_master "
                "WHERE type='index' AND sql IS NOT NULL ORDER BY name"
            ).fetchall()
            for idx_name, tbl_name, idx_sql in index_rows:
                pg_idx_sql = pgcompat.translate_sql(idx_sql)
                try:
                    pg_cx.execute(pg_idx_sql)
                    report["indexes_created"] += 1
                except Exception as exc:
                    report["skipped"].append(
                        {"index": idx_name, "table": tbl_name, "reason": str(exc)})
            pg_cx.commit()
        finally:
            pg_cx.close()
    finally:
        sqlite_cx.close()

    return report
