"""Pure helpers to run SQLite-dialect SQL through psycopg with minimal call-site churn."""
import re
from typing import Optional, Sequence

# Two mechanical, high-volume SQLite->Postgres DDL idioms auto-translated on every
# Postgres query so the ~117 unported modules' CREATE TABLEs work with ZERO source
# changes. Case-insensitive. Compiled once at module level.
#
# 1) `INTEGER PRIMARY KEY AUTOINCREMENT` -> `BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY`
_RE_AUTOINCREMENT = re.compile(r"(?i)\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b")
# 2) `datetime('now')` -> `now()::text`. Bare form only — `datetime('now','localtime')`
#    and other multi-arg variants are intentionally left alone (out of scope for v1).
_RE_DATETIME_NOW = re.compile(r"(?i)datetime\(\s*'now'\s*\)")
# 3) `INSERT OR IGNORE INTO` -> `INSERT INTO` (the "ON CONFLICT DO NOTHING" suffix
#    is appended separately by _translate_insert_or_ignore, since it must land at
#    the end of the statement or just before a trailing RETURNING clause).
_RE_INSERT_OR_IGNORE = re.compile(r"(?i)\bINSERT\s+OR\s+IGNORE\s+INTO\b")
# Trailing RETURNING clause (to the end of the statement) — used to find where to
# splice in "ON CONFLICT DO NOTHING" ahead of it, case-insensitive.
_RE_RETURNING_CLAUSE = re.compile(r"(?i)\bRETURNING\b.*\Z", re.DOTALL)

def _translate_ddl_idioms(sql: str) -> str:
    """Auto-translate the mechanical SQLite DDL/DML idioms to their Postgres
    equivalents. Runs on the raw SQL, before any quote/comment-aware parsing —
    so a string-literal DATA value that happens to contain one of these exact
    phrases would also be rewritten. This is a known, accepted low-blast-radius
    risk (see tests + report): the idioms are DDL/DML-only in practice, and none
    of the replacements re-introduce the pattern they replace, so repeated
    translation is idempotent."""
    sql = _RE_AUTOINCREMENT.sub("BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY", sql)
    sql = _RE_DATETIME_NOW.sub("now()::text", sql)
    sql = _translate_insert_or_ignore(sql)
    return sql

def _translate_insert_or_ignore(sql: str) -> str:
    """`INSERT OR IGNORE INTO <t> (...) VALUES (...)` ->
    `INSERT INTO <t> (...) VALUES (...) ON CONFLICT DO NOTHING`.

    Uses `ON CONFLICT DO NOTHING` with NO explicit conflict target, so it
    matches an ignore-on-any-constraint-violation (matching SQLite's INSERT OR
    IGNORE dedup behavior) without needing to know which unique index/constraint
    is involved (avoids Postgres error 42P10, "no unique or exclusion constraint
    matching the ON CONFLICT specification", which a wrong/missing explicit
    target would trigger).

    Only acts if "INSERT OR IGNORE" was actually present, so a plain INSERT (or
    an already-translated statement carrying ON CONFLICT DO NOTHING with no
    "INSERT OR IGNORE" left in it) is returned unchanged -- keeping this
    idempotent.

    Semantic caveat (not fixed here, see report): SQLite's INSERT OR IGNORE also
    silently skips rows that would violate NOT NULL/CHECK constraints, not just
    UNIQUE/PK conflicts. Untargeted ON CONFLICT DO NOTHING only suppresses
    UNIQUE/PK/exclusion conflicts -- Postgres will still raise on a NOT NULL or
    CHECK violation. This matches the dominant, intended use (dedup on a unique
    key); the rare case of relying on IGNORE to swallow a NOT NULL/CHECK
    violation would need a source-level review, not a mechanical translation.
    """
    if not _RE_INSERT_OR_IGNORE.search(sql):
        return sql
    sql = _RE_INSERT_OR_IGNORE.sub("INSERT INTO", sql)
    m = _RE_RETURNING_CLAUSE.search(sql)
    if m:
        start = m.start()
        sql = sql[:start] + "ON CONFLICT DO NOTHING " + sql[start:]
    else:
        sql = sql.rstrip()
        if sql.endswith(";"):
            sql = sql[:-1].rstrip()
        sql = sql + " ON CONFLICT DO NOTHING"
    return sql

def translate_sql(sql: str) -> str:
    """SQLite '?' params -> psycopg '%s'. Leaves '?' inside single-quoted string
    literals and inside SQL comments alone, and escapes every literal '%' as '%%'
    (psycopg treats '%' as a placeholder marker). Also auto-translates the two
    mechanical AUTOINCREMENT / datetime('now') DDL idioms first (see
    `_translate_ddl_idioms`) — neither introduces a '%' or '?', so running them
    before the escape/placeholder pass below is safe."""
    sql = _translate_ddl_idioms(sql)
    sql = sql.replace("%", "%%")
    out = []
    i, n = 0, len(sql)
    in_str = False
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if in_str:
            out.append(ch)
            if ch == "'":
                in_str = False
            i += 1
        elif ch == "'":
            in_str = True
            out.append(ch)
            i += 1
        elif ch == "-" and nxt == "-":
            j = sql.find("\n", i)
            j = n if j == -1 else j
            out.append(sql[i:j])
            i = j
        elif ch == "/" and nxt == "*":
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(sql[i:j])
            i = j
        elif ch == "?":
            out.append("%s")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)

class HybridRow:
    """Row that supports both int-index and column-name access, like sqlite3.Row."""
    __slots__ = ("_cols", "_vals", "_idx")
    def __init__(self, columns: Sequence[str], values: Sequence):
        self._cols = list(columns)
        self._vals = tuple(values)
        self._idx = {}
        for i, c in enumerate(self._cols):
            k = c.lower() if isinstance(c, str) else c
            if k not in self._idx:
                self._idx[k] = i
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._vals[self._idx[key.lower()]]
        return self._vals[key]
    def keys(self):
        return list(self._cols)
    def __len__(self):
        return len(self._vals)
    def __iter__(self):
        return iter(self._vals)
