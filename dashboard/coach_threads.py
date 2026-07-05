"""1:1 coaching thread store (arc slice 3). Pure sqlite. A thread hangs off a
matched pair (coach_email, member_email); two roles (coach/member) + a source tag
('coaching' now, 'peer' later) so the peer-matching arc reuses these tables. Text
only, async. Privacy + block/report policy live in the routes; this module is state."""

_DDL = """
CREATE TABLE IF NOT EXISTS coach_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL DEFAULT 'coaching',
    coach_email TEXT NOT NULL,
    member_email TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    blocked_by TEXT,
    reported INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    coach_last_read_at TEXT,
    member_last_read_at TEXT,
    UNIQUE(coach_email, member_email)
);
CREATE TABLE IF NOT EXISTS coach_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL,
    sender_role TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_cmsg_thread ON coach_messages(thread_id, id);
CREATE TABLE IF NOT EXISTS coach_thread_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL,
    reporter_role TEXT NOT NULL,
    reason TEXT,
    created_at TEXT,
    resolved INTEGER NOT NULL DEFAULT 0
);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_thread_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def get_or_create_thread(cx, *, coach_email, member_email, source="coaching"):
    ce, me = _lc(coach_email), _lc(member_email)
    cx.execute("INSERT OR IGNORE INTO coach_threads (source, coach_email, member_email, "
               "status, created_at) VALUES (?,?,?, 'active', ?)", (source, ce, me, _now()))
    cx.commit()
    return dict(cx.execute("SELECT * FROM coach_threads WHERE coach_email=? AND member_email=?",
                           (ce, me)).fetchone())


def get_thread(cx, thread_id):
    row = cx.execute("SELECT * FROM coach_threads WHERE id=?", (thread_id,)).fetchone()
    return dict(row) if row else None


def thread_for_pair(cx, coach_email, member_email):
    row = cx.execute("SELECT * FROM coach_threads WHERE coach_email=? AND member_email=?",
                     (_lc(coach_email), _lc(member_email))).fetchone()
    return dict(row) if row else None


def post_message(cx, *, thread_id, sender_role, body):
    cur = cx.execute("INSERT INTO coach_messages (thread_id, sender_role, body, created_at) "
                     "VALUES (?,?,?,?)", (thread_id, sender_role, body, _now()))
    cx.commit()
    return cur.lastrowid


def messages(cx, thread_id):
    rows = cx.execute("SELECT id, sender_role, body, created_at FROM coach_messages "
                      "WHERE thread_id=? ORDER BY id", (thread_id,)).fetchall()
    return [dict(r) for r in rows]


def mark_read(cx, thread_id, role):
    col = "coach_last_read_at" if role == "coach" else "member_last_read_at"
    cx.execute(f"UPDATE coach_threads SET {col}=? WHERE id=?", (_now(), thread_id))
    cx.commit()


def unread_count(cx, thread_id, role):
    col = "coach_last_read_at" if role == "coach" else "member_last_read_at"
    other = "member" if role == "coach" else "coach"
    row = cx.execute(f"SELECT {col} AS lr FROM coach_threads WHERE id=?", (thread_id,)).fetchone()
    last = (row["lr"] if row else None) or ""
    return cx.execute("SELECT COUNT(*) FROM coach_messages WHERE thread_id=? AND sender_role=? "
                      "AND created_at > ?", (thread_id, other, last)).fetchone()[0]


def block_thread(cx, thread_id, blocked_by_role):
    cx.execute("UPDATE coach_threads SET status='blocked', blocked_by=? WHERE id=?",
               (blocked_by_role, thread_id))
    cx.commit()


def report_thread(cx, *, thread_id, reporter_role, reason):
    cx.execute("INSERT INTO coach_thread_reports (thread_id, reporter_role, reason, created_at) "
               "VALUES (?,?,?,?)", (thread_id, reporter_role, (reason or "")[:500], _now()))
    cx.execute("UPDATE coach_threads SET reported=1 WHERE id=?", (thread_id,))
    cx.commit()


def list_all_threads(cx):
    rows = cx.execute(
        "SELECT t.*, "
        "(SELECT COUNT(*) FROM coach_messages m WHERE m.thread_id=t.id) AS message_count, "
        "(SELECT MAX(created_at) FROM coach_messages m WHERE m.thread_id=t.id) AS last_message_at "
        "FROM coach_threads t "
        "ORDER BY (t.reported=1 OR t.status='blocked') DESC, last_message_at DESC").fetchall()
    return [dict(r) for r in rows]
