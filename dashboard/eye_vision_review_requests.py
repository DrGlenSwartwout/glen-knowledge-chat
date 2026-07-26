"""Paid-member requests for review of generated Eye & Vision E4L reports."""
import json
from datetime import datetime, timezone
from dashboard import db


VALID_ACTIONS = {"approve", "decline", "save_draft"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS eye_vision_review_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'requested',
            report_json TEXT NOT NULL,
            reviewer_notes TEXT NOT NULL DEFAULT '',
            requested_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reviewed_at TEXT,
            UNIQUE(email, scan_date)
        )""")
    if not db.column_exists(cx, "eye_vision_review_requests", "reviewer_id"):
        cx.execute("ALTER TABLE eye_vision_review_requests "
                   "ADD COLUMN reviewer_id TEXT NOT NULL DEFAULT ''")
    if not db.column_exists(cx, "eye_vision_review_requests", "revision_log_json"):
        cx.execute("ALTER TABLE eye_vision_review_requests "
                   "ADD COLUMN revision_log_json TEXT NOT NULL DEFAULT '[]'")
    cx.execute("""
        CREATE INDEX IF NOT EXISTS idx_eye_vision_review_status
        ON eye_vision_review_requests(status, updated_at)
        """)
    cx.commit()


def _row(row):
    if row is None:
        return None
    return {
        "id": row["id"], "email": row["email"], "scan_date": row["scan_date"],
        "status": row["status"], "report": json.loads(row["report_json"] or "{}"),
        "reviewer_notes": row["reviewer_notes"] or "",
        "requested_at": row["requested_at"], "updated_at": row["updated_at"],
        "reviewed_at": row["reviewed_at"],
        "reviewer_id": (
            row["reviewer_id"] if "reviewer_id" in row.keys() else "") or "",
        "revision_log": json.loads(
            (row["revision_log_json"]
             if "revision_log_json" in row.keys() else None) or "[]"),
    }


def request_review(cx, email, scan_date, report):
    init_table(cx)
    email = (email or "").strip().lower()
    scan_date = (scan_date or "").strip()
    if not email or not scan_date or not isinstance(report, dict):
        raise ValueError("email, scan_date, and report are required")
    now = _now()
    cx.execute("""
        INSERT INTO eye_vision_review_requests (
            email, scan_date, status, report_json, requested_at, updated_at
        ) VALUES (?,?,?,?,?,?)
        ON CONFLICT(email, scan_date) DO UPDATE SET
            status=CASE
                WHEN eye_vision_review_requests.status='approved' THEN 'approved'
                ELSE 'requested'
            END,
            report_json=CASE
                WHEN eye_vision_review_requests.status='approved'
                THEN eye_vision_review_requests.report_json
                ELSE excluded.report_json
            END,
            updated_at=excluded.updated_at
        """, (email, scan_date, "requested", json.dumps(report), now, now))
    cx.commit()
    return get_for_scan(cx, email, scan_date)


def get_for_scan(cx, email, scan_date):
    init_table(cx)
    row = cx.execute("""
        SELECT * FROM eye_vision_review_requests
        WHERE lower(email)=lower(?) AND scan_date=?
        """, ((email or "").strip(), (scan_date or "").strip())).fetchone()
    return _row(row)


def pending(cx):
    init_table(cx)
    rows = cx.execute("""
        SELECT * FROM eye_vision_review_requests
        WHERE status IN ('requested','draft')
        ORDER BY updated_at ASC
        """).fetchall()
    return [_row(row) for row in rows]


def review(cx, request_id, action, report, reviewer_notes="", reviewer_id=""):
    init_table(cx)
    action = (action or "").strip()
    if action not in VALID_ACTIONS:
        raise ValueError("action must be approve, decline, or save_draft")
    row = cx.execute(
        "SELECT * FROM eye_vision_review_requests WHERE id=?",
        (int(request_id),)).fetchone()
    if row is None:
        return None
    reviewer_id = (reviewer_id or "").strip()
    if not reviewer_id:
        raise ValueError("reviewer identity is required")
    if action != "decline" and not isinstance(report, dict):
        raise ValueError("an edited report object is required")
    status = {
        "approve": "approved", "decline": "declined", "save_draft": "draft",
    }[action]
    now = _now()
    report_json = row["report_json"] if action == "decline" else json.dumps(report)
    reviewed_at = now if action in {"approve", "decline"} else None
    revision_log = json.loads(
        (row["revision_log_json"]
         if "revision_log_json" in row.keys() else None) or "[]")
    revision_log.append({
        "action": action, "from_status": row["status"],
        "to_status": status, "reviewer_id": reviewer_id, "at": now,
    })
    cx.execute("""
        UPDATE eye_vision_review_requests
        SET status=?, report_json=?, reviewer_notes=?, updated_at=?, reviewed_at=?,
            reviewer_id=?, revision_log_json=?
        WHERE id=?
        """, (status, report_json, (reviewer_notes or "").strip(),
              now, reviewed_at, reviewer_id, json.dumps(revision_log),
              int(request_id)))
    cx.commit()
    return _row(cx.execute(
        "SELECT * FROM eye_vision_review_requests WHERE id=?",
        (int(request_id),)).fetchone())
