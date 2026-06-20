"""Begin #4a store: per-scan funnel Biofield reveal drafts. ai_draft -> confirmed.
Distinct from portal_biofield_reports (the $300-service portal report)."""
import json
import sqlite3
from datetime import datetime, timezone


def _rows_cursor(cx):
    """A cursor with its own Row factory, so callers' connection state is never
    mutated (a leaked row_factory would silently turn later bare queries into Rows)."""
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    return cur


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS biofield_reveals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            top_json TEXT NOT NULL,
            blurred_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'ai_draft',
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
    d["top"] = json.loads(d.pop("top_json") or "{}")
    d["blurred"] = json.loads(d.pop("blurred_json") or "[]")
    return d


def upsert_draft(cx, email, scan_date, top, blurred, source):
    cx.row_factory = None
    email = (email or "").strip().lower()
    now = _now()
    existing = cx.execute(
        "SELECT id, status FROM biofield_reveals WHERE email=? AND scan_date=?",
        (email, scan_date)).fetchone()
    if existing is not None:
        rid, status = existing[0], existing[1]
        if status == "confirmed":
            return rid  # never overwrite a confirmed reveal
        cx.execute(
            "UPDATE biofield_reveals SET top_json=?, blurred_json=?, updated_at=? WHERE id=?",
            (json.dumps(top or {}), json.dumps(blurred or []), now, rid))
        cx.commit()
        return rid
    cur = cx.execute(
        "INSERT INTO biofield_reveals (email, scan_date, top_json, blurred_json, status, created_at, updated_at) "
        "VALUES (?,?,?,?, 'ai_draft', ?, ?)",
        (email, scan_date, json.dumps(top or {}), json.dumps(blurred or []), now, now))
    cx.commit()
    return cur.lastrowid


def list_drafts(cx):
    rows = _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE status='ai_draft' ORDER BY id DESC").fetchall()
    return [_row(r) for r in rows]


def get(cx, rid):
    return _row(_rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE id=?", (rid,)).fetchone())


def get_by_token_hash(cx, th):
    return _row(_rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE token_hash=? AND status='confirmed'", (th,)).fetchone())


def set_top(cx, rid, top):
    cx.execute("UPDATE biofield_reveals SET top_json=?, updated_at=? WHERE id=? AND status='ai_draft'",
               (json.dumps(top or {}), _now(), rid))
    cx.commit()


def approve(cx, rid, by, token_hash):
    now = _now()
    cur = cx.execute(
        "UPDATE biofield_reveals SET status='confirmed', approved_at=?, approved_by=?, token_hash=?, updated_at=? "
        "WHERE id=? AND status='ai_draft'",
        (now, by, token_hash, now, rid))
    cx.commit()
    return cur.rowcount == 1
