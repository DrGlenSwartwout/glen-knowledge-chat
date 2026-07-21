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

def _translate_ddl_idioms(sql: str) -> str:
    """Auto-translate the two mechanical SQLite DDL idioms to their Postgres
    equivalents. Runs on the raw SQL, before any quote/comment-aware parsing —
    so a string-literal DATA value that happens to contain the exact phrase
    "INTEGER PRIMARY KEY AUTOINCREMENT" would also be rewritten. This is a
    known, accepted low-blast-radius risk for v1 (see tests + report): the
    idiom is DDL-only in practice, and neither replacement re-introduces the
    pattern it replaces, so repeated translation is idempotent."""
    sql = _RE_AUTOINCREMENT.sub("BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY", sql)
    sql = _RE_DATETIME_NOW.sub("now()::text", sql)
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
