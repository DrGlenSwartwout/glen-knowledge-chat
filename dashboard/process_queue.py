"""Cold-client synthesis request queue (Phase 2). One pending row per email; the
local watcher claims pending requests, runs the importer, and marks done."""
import datetime
import sqlite3


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS portal_process_requests (
        email TEXT PRIMARY KEY, scan_date TEXT, status TEXT DEFAULT 'pending',
        requested_at TEXT, updated_at TEXT)""")
    cx.commit()


def _norm(email):
    return (email or "").strip().lower()


def enqueue(cx, email, scan_date=""):
    init_table(cx)
    email = _norm(email)
    now = _now()
    cx.execute("""INSERT INTO portal_process_requests (email, scan_date, status, requested_at, updated_at)
                  VALUES (?,?, 'pending', ?, ?)
                  ON CONFLICT(email) DO UPDATE SET scan_date=excluded.scan_date,
                      status='pending', requested_at=?, updated_at=?""",
               (email, scan_date or "", now, now, now, now))
    cx.commit()


def list_pending(cx):
    init_table(cx)
    rows = cx.execute("SELECT email, scan_date, requested_at FROM portal_process_requests "
                      "WHERE status='pending' ORDER BY requested_at ASC").fetchall()
    return [{"email": r[0], "scan_date": r[1], "requested_at": r[2]} for r in rows]


def mark_done(cx, email):
    init_table(cx)
    cur = cx.execute("UPDATE portal_process_requests SET status='done', updated_at=? "
                     "WHERE email=? AND status='pending'", (_now(), _norm(email)))
    cx.commit()
    return cur.rowcount > 0
