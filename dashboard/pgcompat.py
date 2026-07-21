"""Pure helpers to run SQLite-dialect SQL through psycopg with minimal call-site churn."""
from typing import Optional, Sequence

def translate_sql(sql: str) -> str:
    """SQLite '?' params -> psycopg '%s', leaving '?' inside single-quoted string
    literals alone, and escaping every literal '%' as '%%' (psycopg treats '%' as
    a placeholder marker). Escape percents first, then convert '?' outside strings."""
    sql = sql.replace("%", "%%")
    out = []
    in_str = False
    for ch in sql:
        if ch == "'":
            in_str = not in_str
            out.append(ch)
        elif ch == "?" and not in_str:
            out.append("%s")
        else:
            out.append(ch)
    return "".join(out)

class HybridRow:
    """Row that supports both int-index and column-name access, like sqlite3.Row."""
    __slots__ = ("_cols", "_vals", "_idx")
    def __init__(self, columns: Sequence[str], values: Sequence):
        self._cols = list(columns)
        self._vals = tuple(values)
        self._idx = {c: i for i, c in enumerate(self._cols)}
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
