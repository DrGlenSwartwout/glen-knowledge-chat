# dashboard/coach_connect.py
"""Coaching pairing (arc slice 2): member applications, waitlist, paid-tier
interest, and the opaque coach ref. Pure sqlite; no app-layer imports. Double
opt-in: the member applies (pending request + note), the coach accepts.
Privacy: coaches are referenced by an opaque ref (sha256 of email), and a
member is shown to a coach as first name + note only."""

import hashlib

_DDL = """
CREATE TABLE IF NOT EXISTS coach_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coach_email TEXT NOT NULL,
    member_email TEXT NOT NULL,
    member_name TEXT,
    note TEXT,
    member_video_url TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT,
    decided_at TEXT,
    UNIQUE(coach_email, member_email)
);
CREATE INDEX IF NOT EXISTS ix_creq_coach ON coach_requests(coach_email, status);
CREATE INDEX IF NOT EXISTS ix_creq_member ON coach_requests(member_email, status);
CREATE TABLE IF NOT EXISTS coach_waitlist (
    member_email TEXT PRIMARY KEY,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS coaching_interest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_email TEXT NOT NULL,
    tier TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(member_email, tier)
);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_connect_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def coach_ref(email):
    return hashlib.sha256(_lc(email).encode("utf-8")).hexdigest()[:16]


def email_for_ref(cx, ref):
    """Resolve an opaque ref to an ACTIVE+cert_ok coach email, or None."""
    from dashboard import coach_directory as _cd
    for c in _cd.list_active_full(cx):
        if coach_ref(c["email"]) == ref:
            return c["email"]
    return None


def member_has_accepted(cx, member_email):
    """True if the member is already matched (has an accepted coach)."""
    return cx.execute("SELECT 1 FROM coach_requests WHERE member_email=? AND status='accepted' "
                      "LIMIT 1", (_lc(member_email),)).fetchone() is not None


def member_applications(cx, member_email):
    """The member's pending + accepted applications: [{coach_email, status}]."""
    rows = cx.execute("SELECT coach_email, status FROM coach_requests WHERE member_email=? "
                      "AND status IN ('pending','accepted') ORDER BY id", (_lc(member_email),)).fetchall()
    return [{"coach_email": r["coach_email"], "status": r["status"]} for r in rows]


def request_member(cx, request_id):
    """The member_email that owns a request id, or None."""
    row = cx.execute("SELECT member_email FROM coach_requests WHERE id=?", (request_id,)).fetchone()
    return row["member_email"] if row else None


def withdraw_other_pendings(cx, member_email, keep_request_id):
    """When a member is claimed (a coach accepts), withdraw their OTHER pending
    applications so first-accept-wins."""
    cx.execute("UPDATE coach_requests SET status='withdrawn', decided_at=? "
               "WHERE member_email=? AND status='pending' AND id!=?",
               (_now(), _lc(member_email), keep_request_id))
    cx.commit()


def create_request(cx, coach_email, member_email, member_name, note, member_video_url=""):
    """Create a pending application. A member may hold MULTIPLE pending
    applications, but not once already matched: returns None if the member
    already has an accepted coach, or if they already applied to THIS coach."""
    member_email = _lc(member_email)
    if member_has_accepted(cx, member_email):
        return None
    if cx.execute("SELECT 1 FROM coach_requests WHERE coach_email=? AND member_email=? "
                  "AND status IN ('pending','accepted') LIMIT 1",
                  (_lc(coach_email), member_email)).fetchone():
        return None  # already applied to this coach
    cur = cx.execute(
        "INSERT INTO coach_requests (coach_email,member_email,member_name,note,"
        "member_video_url,status,created_at) VALUES (?,?,?,?,?, 'pending', ?) "
        "ON CONFLICT(coach_email,member_email) DO UPDATE SET status='pending', "
        "member_name=excluded.member_name, note=excluded.note, "
        "member_video_url=excluded.member_video_url, created_at=excluded.created_at",
        (_lc(coach_email), member_email, member_name, (note or "")[:500],
         member_video_url or "", _now()))
    cx.commit()
    return cur.lastrowid


def requests_for_coach(cx, coach_email, status="pending"):
    rows = cx.execute("SELECT id, member_name, note, member_video_url, status "
                      "FROM coach_requests WHERE coach_email=? AND status=? ORDER BY created_at",
                      (_lc(coach_email), status)).fetchall()
    return [{"id": r["id"], "member_name": r["member_name"], "note": r["note"],
             "member_video_url": r["member_video_url"], "status": r["status"]} for r in rows]


def accepted_count(cx, coach_email):
    return cx.execute("SELECT COUNT(*) FROM coach_requests WHERE coach_email=? "
                      "AND status='accepted'", (_lc(coach_email),)).fetchone()[0]


def set_request_status(cx, request_id, status):
    cx.execute("UPDATE coach_requests SET status=?, decided_at=? WHERE id=?",
               (status, _now(), request_id))
    cx.commit()


def request_owner(cx, request_id):
    """The coach_email that owns a request id (for authorization), or None."""
    row = cx.execute("SELECT coach_email FROM coach_requests WHERE id=?",
                     (request_id,)).fetchone()
    return row["coach_email"] if row else None


def join_waitlist(cx, member_email):
    cx.execute("INSERT OR IGNORE INTO coach_waitlist (member_email,created_at) VALUES (?,?)",
               (_lc(member_email), _now()))
    cx.commit()


def on_waitlist(cx, member_email):
    return cx.execute("SELECT 1 FROM coach_waitlist WHERE member_email=?",
                      (_lc(member_email),)).fetchone() is not None


def record_interest(cx, member_email, tier):
    cx.execute("INSERT OR IGNORE INTO coaching_interest (member_email,tier,created_at) "
               "VALUES (?,?,?)", (_lc(member_email), tier, _now()))
    cx.commit()
