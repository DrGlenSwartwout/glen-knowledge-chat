"""Row-count parity check: SQLite source vs its Postgres target schema.

Kept simple for v1 (per the P05 task-2 brief): compare raw per-table counts
and flag mismatches for operator review. A table present on only one side
counts as a mismatch (its missing-side count is None).

P06 Code Task C1 adds CONTENT-checksum parity (`checksum_table` /
`checksum_parity` / `checksum_all_ok`) below -- row-count parity above
cannot detect content corruption (truncation, type coercion, encoding)
that preserves cardinality, which is unacceptable before an irreversible
cutover. See those functions' docstrings for the normalization rules that
make the digest identical across backends.
"""
import hashlib
import math
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from dashboard import db
from dashboard.dbschema import schema_for_path
from scripts.pgmig import introspect


def parity(sqlite_path: str) -> List[Dict]:
    schema = schema_for_path(sqlite_path)
    # M5: open the SOURCE truly read-only -- a verify run must not be able to
    # mutate the frozen snapshot (or its WAL) it's comparing against.
    sqlite_cx = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        sqlite_tables = set(introspect.sqlite_tables(sqlite_cx))
        pg_cx = db.connect(sqlite_path)
        try:
            pg_tables = set(introspect.pg_fk_order(pg_cx, schema))
            results = []
            for t in sorted(sqlite_tables | pg_tables):
                s_count = (sqlite_cx.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                           if t in sqlite_tables else None)
                p_count = (pg_cx.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                           if t in pg_tables else None)
                ok = s_count is not None and p_count is not None and s_count == p_count
                results.append({"table": t, "sqlite": s_count, "postgres": p_count, "ok": ok})
            return results
        finally:
            pg_cx.close()
    finally:
        sqlite_cx.close()


def all_ok(results: List[Dict]) -> bool:
    return all(r["ok"] for r in results)


# --------------------------------------------------------------------------
# P06 Code Task C1: content-checksum parity.
#
# Row-count parity (above) cannot catch content corruption that preserves
# cardinality -- a truncated string, a coerced number, a mangled encoding all
# keep COUNT(*) identical. This computes a per-table digest of the actual
# row CONTENT, the same way on both backends, so an identical copy always
# matches and a content-corrupted copy (same count) never does.
#
# PARITY-BY-CONSTRUCTION: both sides are computed in PYTHON from rows read
# back through the normal driver (sqlite3 / the dashboard.db Postgres
# adapter) -- never a SQL-side digest function -- so the exact same
# normalization code runs for both backends. That is what makes "identical
# data -> identical digest" true by construction rather than by luck.
#
# Normalization rules (`_normalize_value`) -- chosen so the SAME logical
# value renders to the SAME string regardless of which backend/driver
# produced the Python object:
#   - NULL -> the fixed sentinel "\x00NULL". A control byte prefix keeps it
#     from ever colliding with a real text value (including the literal
#     string "None" or an empty string) -- see
#     test_checksum_table_null_vs_value_not_confused.
#   - bool -> "1"/"0". Checked BEFORE the int branch (bool is an int
#     subclass in Python). Postgres BOOLEAN columns come back as real
#     Python bool via psycopg; SQLite has no bool type, so an app storing
#     0/1 in an INTEGER column already comes back as a plain int -- both
#     paths converge on the same "1"/"0" string.
#   - int -> str(v). Identical on both backends for integer-affinity
#     columns.
#   - float -> non-finite values (NaN/+Inf/-Inf) are checked FIRST via
#     `math.isfinite` and rendered as a fixed, distinct sentinel
#     ("\x00NAN" / "\x00INF+" / "\x00INF-") -- `v == int(v)` below would
#     otherwise call `int(v)`, which raises OverflowError/ValueError for
#     those values and crashed the whole checksum run. Otherwise: if the
#     value is integral (5.0), render as "5" so it matches an
#     INTEGER-column int 5 that a loosely-typed SQLite column might hold
#     for the same logical value; otherwise repr(v) (Python's shortest
#     round-trip text form, so the same float value strified the same way
#     regardless of driver).
#   - decimal.Decimal (Postgres NUMERIC/DECIMAL columns come back as
#     Decimal via psycopg) -> normalize() to strip insignificant trailing
#     zeros, then render with format(..., 'f') to avoid scientific
#     notation, so Decimal('5.00') and a SQLite-side float 5.0 both land on
#     "5".
#   - bytes/bytearray/memoryview (BLOB / bytea) -> a "\x00BYTES:" prefix
#     (so it can never collide with a text value) + lowercase hex.
#   - datetime.datetime -> a single CANONICAL rendering, regardless of
#     which backend produced the object: if tz-aware, convert to UTC first
#     (`astimezone(timezone.utc)`); naive values are left as-is (assumed
#     already the intended wall-clock/UTC instant); then render with a
#     fixed `strftime("%Y-%m-%dT%H:%M:%S.%f")` (always 6 fractional digits,
#     so precision/format never depends on whether the driver happened to
#     omit a zero microsecond). This is what makes a Postgres TIMESTAMP
#     column's `datetime` object converge with a SQLite TEXT column's ISO
#     string for the SAME logical instant (see the str-branch below) --
#     fixing the false-positive where a faithful copy was reported
#     ok=False just because str(datetime) uses a space separator while the
#     app's ISO strings use "T"/"Z".
#   - str -> first TRY to parse it as an ISO-8601 *datetime* (must contain
#     an explicit time component -- a "T" or a space -- so plain DATE-only
#     strings like "2026-07-10" are left alone, since those correspond to
#     `datetime.date` objects on the Postgres side, not `datetime`, and are
#     out of scope for this fix). A trailing "Z" is treated as "+00:00"
#     before calling `datetime.fromisoformat`; both "T" and space
#     separators are accepted (fromisoformat handles both). If it parses,
#     render it through the SAME canonical datetime rendering above. If it
#     does NOT parse (any ValueError), the string is returned UNCHANGED.
#     Because the exact same normalize() function runs on BOTH backends'
#     readback, this is safe even for a column that merely looks
#     date-like: a plain text value that coincidentally parses as a
#     datetime is transformed IDENTICALLY on both sides (still matches,
#     no new false positive), while two genuinely different instants still
#     render to different canonical strings (no over-normalization masking
#     real corruption).
#   - datetime.date/datetime.time (not datetime.datetime) -> str(v),
#     unchanged from before -- out of scope for this fix.
#   - anything else -> str(v) (e.g. an already-str value that didn't parse
#     as a datetime, used as-is).
#
# NUL-byte normalization (P05.5 hardening, live dry-run): `scripts.pgmig
# .copy.copy_table` strips embedded NUL (0x00) bytes from `str` values
# before inserting into Postgres (TEXT/VARCHAR cannot hold 0x00; SQLite
# can). Left unaccounted for here, a faithfully-sanitized row would read
# back WITH the NUL on the SQLite side and WITHOUT it on the Postgres side
# -- two different strings -> a false-positive checksum mismatch on an
# otherwise-correct copy. So `\x00` is stripped from `str` values in THIS
# normalization too (both sides, same code, same as the copy step) --
# applied to the datetime-candidate string BEFORE the ISO-parse attempt,
# so a NUL embedded in a date-like string can't affect whether it parses.
# This does not touch `bytes`/BYTEA (a legitimate 0x00 there is preserved
# by both the copy step and this normalization).
#
# Per-row hashing: the row's normalized values are joined with "\x1f" (ASCII
# Unit Separator -- vanishingly unlikely to appear in real column data) and
# md5'd; the row order of VALUES is fixed by an explicit column list (same
# list used against both backends), so within a row this is positional, not
# order-independent -- only ACROSS rows does order not matter.
#
# Order-independent aggregation across rows: SQLite and Postgres may return
# rows in different physical orders, so the final digest must not depend on
# row order. Per-row md5 ints are combined with a commutative SUM modulo
# 2**128 (a fixed-width, order-independent combiner) rather than
# concatenating in read order.
# --------------------------------------------------------------------------

_NULL_SENTINEL = "\x00NULL"
_ROW_SEP = "\x1f"
_AGG_MOD = 1 << 128
_NAN_SENTINEL = "\x00NAN"
_INF_POS_SENTINEL = "\x00INF+"
_INF_NEG_SENTINEL = "\x00INF-"
_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def _canonical_datetime(dt: datetime) -> str:
    """Single canonical rendering for a `datetime` used on BOTH backends
    (see the module docstring's normalization-rules comment). tz-aware
    values are converted to UTC first; naive values are left as-is."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(_DATETIME_FMT)


def _try_parse_iso_datetime(s: str) -> Optional[datetime]:
    """Try to parse `s` as an ISO-8601 *datetime* (must carry an explicit
    time component). Returns None (never raises) if it isn't one -- see
    the module docstring's normalization-rules comment for the reasoning."""
    text = s.strip()
    if "T" not in text and " " not in text:
        # No time separator at all -- this is not a datetime string (could
        # be a plain DATE-only string, or unrelated text). Leave it alone.
        return None
    candidate = text
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _normalize_value(v) -> str:
    if v is None:
        return _NULL_SENTINEL
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            if math.isnan(v):
                return _NAN_SENTINEL
            return _INF_POS_SENTINEL if v > 0 else _INF_NEG_SENTINEL
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
    if isinstance(v, Decimal):
        return format(v.normalize(), "f")
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "\x00BYTES:" + bytes(v).hex()
    if isinstance(v, datetime):
        return _canonical_datetime(v)
    if isinstance(v, str):
        if "\x00" in v:
            v = v.replace("\x00", "")
        parsed = _try_parse_iso_datetime(v)
        if parsed is not None:
            return _canonical_datetime(parsed)
        return v
    return str(v)


def _row_hash(row) -> int:
    rendered = _ROW_SEP.join(_normalize_value(v) for v in row)
    digest = hashlib.md5(rendered.encode("utf-8", errors="surrogatepass")).hexdigest()
    return int(digest, 16)


def _digest_rows(cursor) -> str:
    """Order-independent digest of every row `cursor` yields: sum the
    per-row md5 ints modulo 2**128 (so row order never affects the result),
    rendered as a fixed-width 32-hex-digit string."""
    total = 0
    for row in cursor:
        total = (total + _row_hash(row)) % _AGG_MOD
    return format(total, "032x")


def _sqlite_columns(cx, table: str) -> List[str]:
    return [d[0] for d in cx.execute(f'SELECT * FROM "{table}" LIMIT 0').description]


def _pg_columns(cx, table: str) -> List[str]:
    rows = cx.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND table_schema = current_schema() "
        "ORDER BY ordinal_position",
        (table,)).fetchall()
    return [r[0] for r in rows]


def checksum_table(sqlite_cx, pg_cx, table: str, cols: Optional[List[str]] = None) -> Dict:
    """Order-independent content digest of `table`'s rows, computed the SAME
    way (in Python, from rows read back through each backend's normal
    driver) on both `sqlite_cx` and `pg_cx`. Returns
    {"table","sqlite_digest","pg_digest","ok"}.

    `cols` defaults to ALL of the SQLite table's columns (in that order),
    and the SAME list is then used to query the Postgres side too -- this
    is the "column-order-normalized so both backends use the same column
    order" requirement: order is decided once, from the SQLite source, and
    never independently re-derived per backend.

    If the column SET differs between backends (e.g. a column present on
    only one side), this does NOT crash -- it returns ok=False with a
    human-readable "reason", since that itself is a parity failure the
    operator needs to see before cutover.
    """
    if cols is None:
        cols = _sqlite_columns(sqlite_cx, table)

    pg_cols = _pg_columns(pg_cx, table)
    if {c.lower() for c in cols} != {c.lower() for c in pg_cols}:
        return {
            "table": table,
            "sqlite_digest": None,
            "pg_digest": None,
            "ok": False,
            "reason": (f"column set differs: sqlite={sorted(cols)} "
                       f"postgres={sorted(pg_cols)}"),
        }

    collist = ", ".join(f'"{c}"' for c in cols)
    sqlite_digest = _digest_rows(sqlite_cx.execute(f'SELECT {collist} FROM "{table}"'))
    pg_digest = _digest_rows(pg_cx.execute(f'SELECT {collist} FROM "{table}"'))
    return {
        "table": table,
        "sqlite_digest": sqlite_digest,
        "pg_digest": pg_digest,
        "ok": sqlite_digest == pg_digest,
    }


def checksum_parity(sqlite_path: str) -> List[Dict]:
    """Content-checksum parity for every table of `sqlite_path` vs its
    Postgres target schema (mirrors `parity`'s table enumeration/open
    pattern). A table missing on one side is reported ok=False with a
    reason rather than silently skipped or crashed on."""
    schema = schema_for_path(sqlite_path)
    sqlite_cx = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        sqlite_tables = set(introspect.sqlite_tables(sqlite_cx))
        pg_cx = db.connect(sqlite_path)
        try:
            pg_tables = set(introspect.pg_fk_order(pg_cx, schema))
            results = []
            for t in sorted(sqlite_tables | pg_tables):
                if t not in sqlite_tables:
                    results.append({"table": t, "sqlite_digest": None, "pg_digest": None,
                                     "ok": False, "reason": "table missing on sqlite side"})
                elif t not in pg_tables:
                    results.append({"table": t, "sqlite_digest": None, "pg_digest": None,
                                     "ok": False, "reason": "table missing on postgres side"})
                else:
                    results.append(checksum_table(sqlite_cx, pg_cx, t))
            return results
        finally:
            pg_cx.close()
    finally:
        sqlite_cx.close()


def checksum_all_ok(results: List[Dict]) -> bool:
    return all(r["ok"] for r in results)
