"""Queue of client requests to analyze an as-yet-unprocessed E4L scan. A local worker
fulfills pending rows (synthesize + publish) and marks them done. One row per
(email, scan_date). LOG_DB (SQLite). Separate from the published-report 'requested' flow."""
import datetime


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(e):
    return (e or "").strip().lower()


def init_analysis_requests_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS analysis_requests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email        TEXT NOT NULL,
            scan_id      TEXT,
            scan_date    TEXT NOT NULL,
            requested_at TEXT,
            status       TEXT NOT NULL,
            fulfilled_at TEXT,
            UNIQUE(email, scan_date)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_ar_status ON analysis_requests(status)")
    cx.commit()


def create_request(cx, email, scan_id, scan_date):
    e, d = _norm(email), (scan_date or "").strip()
    if not e or not d:
        return {"created": False, "status": None}
    row = cx.execute("SELECT status FROM analysis_requests WHERE email=? AND scan_date=?",
                     (e, d)).fetchone()
    if row:
        return {"created": False, "status": row[0]}
    cx.execute("INSERT INTO analysis_requests (email, scan_id, scan_date, requested_at, status) "
               "VALUES (?,?,?,?, 'pending')", (e, str(scan_id or ""), d, _now()))
    cx.commit()
    return {"created": True, "status": "pending"}


def has_pending(cx, email, scan_date):
    return cx.execute("SELECT 1 FROM analysis_requests WHERE email=? AND scan_date=? "
                      "AND status='pending' LIMIT 1", (_norm(email), (scan_date or "").strip())
                      ).fetchone() is not None


def pending(cx, limit=50):
    rows = cx.execute("SELECT id, email, scan_id, scan_date FROM analysis_requests "
                      "WHERE status='pending' ORDER BY id LIMIT ?", (int(limit),)).fetchall()
    return [{"id": r[0], "email": r[1], "scan_id": r[2], "scan_date": r[3]} for r in rows]


def statuses_for(cx, email):
    rows = cx.execute("SELECT scan_date, status FROM analysis_requests WHERE email=?",
                      (_norm(email),)).fetchall()
    return {r[0]: r[1] for r in rows}


def mark(cx, req_id, status):
    cx.execute("UPDATE analysis_requests SET status=?, fulfilled_at=? WHERE id=?",
               (status, _now() if status in ("done", "failed") else None, req_id))
    cx.commit()
