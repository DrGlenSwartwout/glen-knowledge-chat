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
# A bare RETURNING keyword — only meaningful when matched against a quote/
# comment-aware "code span" (see _scan_sql_spans), never against the raw SQL,
# so it can't false-match inside a string literal or a comment.
_RE_RETURNING_WORD = re.compile(r"(?i)\bRETURNING\b")


def _scan_sql_spans(sql: str):
    """Split `sql` into contiguous (kind, start, end) spans classified as
    'code', 'string' (single-quoted literal, including its quotes), or
    'comment' (`--` line or `/* */` block). Mirrors the scanner already used
    in `translate_sql` below for the `?` pass, so the two stay consistent
    (same simplistic single-quote model: a doubled '' escape is not treated
    specially, matching the existing placeholder-scanner behavior)."""
    spans = []
    i, n = 0, len(sql)
    code_start = 0
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "'":
            if i > code_start:
                spans.append(("code", code_start, i))
            j = i + 1
            while j < n and sql[j] != "'":
                j += 1
            j = min(j + 1, n)
            spans.append(("string", i, j))
            i = j
            code_start = i
        elif ch == "-" and nxt == "-":
            if i > code_start:
                spans.append(("code", code_start, i))
            j = sql.find("\n", i)
            j = n if j == -1 else j
            spans.append(("comment", i, j))
            i = j
            code_start = i
        elif ch == "/" and nxt == "*":
            if i > code_start:
                spans.append(("code", code_start, i))
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            spans.append(("comment", i, j))
            i = j
            code_start = i
        else:
            i += 1
    if n > code_start:
        spans.append(("code", code_start, n))
    return spans


def _find_trailing_returning(sql: str, spans) -> Optional[int]:
    """Return the start offset of a genuine RETURNING keyword -- one that
    appears in a 'code' span, i.e. outside any string literal or comment --
    or None if there isn't one. Takes the last such match (there should only
    ever be one real trailing RETURNING clause)."""
    last = None
    for kind, start, end in spans:
        if kind != "code":
            continue
        for m in _RE_RETURNING_WORD.finditer(sql, start, end):
            last = m.start()
    return last


def _end_of_code(sql: str, spans) -> int:
    """Return the offset of the true end of SQL *code* -- i.e. right after
    the last 'code' or 'string' span, before any trailing comment(s), with
    trailing whitespace stripped."""
    end = 0
    for kind, _start, s_end in spans:
        if kind in ("code", "string"):
            end = s_end
    while end > 0 and sql[end - 1].isspace():
        end -= 1
    return end

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
    spans = _scan_sql_spans(sql)
    ret_pos = _find_trailing_returning(sql, spans)
    if ret_pos is not None:
        # Genuine trailing RETURNING (outside strings/comments): splice the
        # clause in immediately before it.
        sql = sql[:ret_pos] + "ON CONFLICT DO NOTHING " + sql[ret_pos:]
    else:
        # No real RETURNING: insert at the true end of the SQL *code*, before
        # any trailing comment (and any trailing ';'), not after it.
        end = _end_of_code(sql, spans)
        code = sql[:end]
        if code.endswith(";"):
            code = code[:-1].rstrip()
        trailing = sql[end:].lstrip()
        sql = code + " ON CONFLICT DO NOTHING"
        if trailing:
            sql = sql + " " + trailing
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
