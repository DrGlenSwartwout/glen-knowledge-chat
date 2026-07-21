"""Pure helpers to run SQLite-dialect SQL through psycopg with minimal call-site churn."""
from typing import Optional, Sequence

def translate_sql(sql: str) -> str:
    """SQLite '?' params -> psycopg '%s'. Leaves '?' inside single-quoted string
    literals and inside SQL comments alone, and escapes every literal '%' as '%%'
    (psycopg treats '%' as a placeholder marker)."""
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
            if c not in self._idx:
                self._idx[c] = i
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._vals[self._idx[key]]
        return self._vals[key]
    def keys(self):
        return list(self._cols)
    def __len__(self):
        return len(self._vals)
    def __iter__(self):
        return iter(self._vals)
