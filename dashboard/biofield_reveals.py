"""Begin #4a store: per-scan funnel Biofield reveal (interpretation auto-shown +
ranked remedies, blurred until the top is approved). Distinct from
portal_biofield_reports."""
import json
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _rows_cursor(cx):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    return cur


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS biofield_reveals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            interpretation_json TEXT NOT NULL DEFAULT '{}',
            remedies_json TEXT NOT NULL DEFAULT '[]',
            first_approved INTEGER NOT NULL DEFAULT 0,
            token_hash TEXT,
            approved_at TEXT, approved_by TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(email, scan_date)
        )
    """)
    cx.commit()


def _row(r):
    if r is None:
        return None
    d = dict(r)
    d["interpretation"] = json.loads(d.pop("interpretation_json") or "{}")
    d["remedies"] = json.loads(d.pop("remedies_json") or "[]")
    d["first_approved"] = bool(d.get("first_approved"))
    return d


def upsert(cx, email, scan_date, interpretation, remedies, source):
    """Insert or update a reveal. Content updates only while not yet approved.
    Returns (id, is_new) - is_new True only on first insert (caller mints token +
    emails exactly once)."""
    email = (email or "").strip().lower()
    now = _now()
    existing = cx.execute(
        "SELECT id, first_approved FROM biofield_reveals WHERE email=? AND scan_date=?",
        (email, scan_date)).fetchone()
    if existing is not None:
        rid, approved = existing[0], existing[1]
        if not approved:
            cx.execute(
                "UPDATE biofield_reveals SET interpretation_json=?, remedies_json=?, updated_at=? WHERE id=?",
                (json.dumps(interpretation or {}), json.dumps(remedies or []), now, rid))
            cx.commit()
        return rid, False
    cur = cx.execute(
        "INSERT INTO biofield_reveals (email, scan_date, interpretation_json, remedies_json, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        (email, scan_date, json.dumps(interpretation or {}), json.dumps(remedies or []), now, now))
    cx.commit()
    return cur.lastrowid, True


def set_token(cx, rid, token_hash):
    cx.execute("UPDATE biofield_reveals SET token_hash=?, updated_at=? WHERE id=?",
               (token_hash, _now(), rid))
    cx.commit()


def set_interpretation(cx, rid, interpretation):
    cx.execute("UPDATE biofield_reveals SET interpretation_json=?, updated_at=? WHERE id=? AND first_approved=0",
               (json.dumps(interpretation or {}), _now(), rid))
    cx.commit()


def set_remedies(cx, rid, remedies):
    cx.execute("UPDATE biofield_reveals SET remedies_json=?, updated_at=? WHERE id=? AND first_approved=0",
               (json.dumps(remedies or []), _now(), rid))
    cx.commit()


def approve_first(cx, rid, by):
    now = _now()
    cur = cx.execute(
        "UPDATE biofield_reveals SET first_approved=1, approved_at=?, approved_by=?, updated_at=? WHERE id=?",
        (now, by, now, rid))
    cx.commit()
    return cur.rowcount == 1


def list_pending(cx):
    rows = _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE first_approved=0 ORDER BY id DESC").fetchall()
    return [_row(r) for r in rows]


def get(cx, rid):
    return _row(_rows_cursor(cx).execute("SELECT * FROM biofield_reveals WHERE id=?", (rid,)).fetchone())


def get_by_token_hash(cx, th):
    return _row(_rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE token_hash=?", (th,)).fetchone())
