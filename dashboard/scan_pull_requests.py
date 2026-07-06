"""Queue of console-initiated 'pull this client's latest scan from E4L' requests.

A local worker (02 Skills/scan-pull-fulfill.py) resolves each query to exactly one
E4L client, scrapes the scan, ingests it, and pushes a SILENT reveal draft for
Glen's console review, then marks the row done/failed. Unlike analysis_requests
(keyed on a KNOWN email+scan_date), a pull targets a client whose scan may not be
in e4l.db yet — so the dedup key is the normalized query while pending/working."""
import datetime
import sqlite3

STALE_WORKING_SECS = 1800  # a 'working' row older than this (crash/reboot mid-fulfill) no longer blocks a new request


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(q):
    return (q or "").strip().lower()


def init_scan_pull_requests_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS scan_pull_requests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query        TEXT NOT NULL,
            query_norm   TEXT NOT NULL,
            status       TEXT NOT NULL,       -- pending | working | done | failed
            requested_by TEXT,
            scan_id      TEXT,
            draft_id     INTEGER,
            message      TEXT,
            created_at   TEXT,
            updated_at   TEXT
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_spr_status ON scan_pull_requests(status)")
    cx.commit()


def create_request(cx, query, requested_by=None):
    qn = _norm(query)
    if not qn:
        return {"created": False, "id": None, "status": None}
    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(seconds=STALE_WORKING_SECS)).strftime("%Y-%m-%d %H:%M:%S")
    row = cx.execute(
        "SELECT id, status FROM scan_pull_requests "
        "WHERE query_norm=? AND (status='pending' OR (status='working' AND updated_at >= ?)) "
        "ORDER BY id DESC LIMIT 1",
        (qn, cutoff)).fetchone()
    if row:
        return {"created": False, "id": row[0], "status": row[1]}
    now = _now()
    cur = cx.execute(
        "INSERT INTO scan_pull_requests (query, query_norm, status, requested_by, created_at, updated_at) "
        "VALUES (?,?, 'pending', ?, ?, ?)",
        (query.strip(), qn, (requested_by or None), now, now))
    cx.commit()
    return {"created": True, "id": cur.lastrowid, "status": "pending"}


def pending(cx, limit=50):
    rows = cx.execute(
        "SELECT id, query FROM scan_pull_requests WHERE status='pending' ORDER BY id LIMIT ?",
        (int(limit),)).fetchall()
    return [{"id": r[0], "query": r[1]} for r in rows]


def mark(cx, req_id, status, scan_id=None, draft_id=None, message=None):
    cx.execute(
        "UPDATE scan_pull_requests SET status=?, "
        "scan_id=COALESCE(?,scan_id), draft_id=COALESCE(?,draft_id), "
        "message=COALESCE(?,message), updated_at=? WHERE id=?",
        (status, (str(scan_id) if scan_id is not None else None),
         draft_id, message, _now(), req_id))
    cx.commit()


def get(cx, req_id):
    r = cx.execute(
        "SELECT id, query, status, requested_by, scan_id, draft_id, message, created_at, updated_at "
        "FROM scan_pull_requests WHERE id=?", (req_id,)).fetchone()
    if not r:
        return None
    return {"id": r[0], "query": r[1], "status": r[2], "requested_by": r[3],
            "scan_id": r[4], "draft_id": r[5], "message": r[6],
            "created_at": r[7], "updated_at": r[8]}
