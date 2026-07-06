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
    active_epoch INTEGER NOT NULL DEFAULT 1,
    UNIQUE(coach_email, member_email)
);
CREATE TABLE IF NOT EXISTS coach_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL,
    sender_role TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT,
    epoch INTEGER NOT NULL DEFAULT 1
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
    # Lazy migration for tables created before the epoch columns existed (prod already
    # has coach_threads/coach_messages from slice 3). ALTER raises if the column exists.
    for tbl, col in (("coach_threads", "active_epoch"), ("coach_messages", "epoch")):
        try:
            cx.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} INTEGER NOT NULL DEFAULT 1")
        except Exception:
            pass
    cx.commit()


def reactivate_thread(cx, coach_email, member_email):
    """On a fresh re-accept of a previously-matched pair, give the two participants a
    clean slate: bump active_epoch so prior messages are hidden from BOTH sides (but
    kept for the owner transcript), clear the blocked/reported flags, resolve old
    reports, and reset read marks. No-op when no thread exists yet (a first-time
    accept). Returns True when a thread was reactivated."""
    ce, me = _lc(coach_email), _lc(member_email)
    row = cx.execute("SELECT id FROM coach_threads WHERE coach_email=? AND member_email=?",
                     (ce, me)).fetchone()
    if not row:
        return False
    tid = row["id"]
    cx.execute("UPDATE coach_threads SET status='active', blocked_by=NULL, reported=0, "
               "active_epoch=active_epoch+1, coach_last_read_at=NULL, member_last_read_at=NULL "
               "WHERE id=?", (tid,))
    cx.execute("UPDATE coach_thread_reports SET resolved=1 WHERE thread_id=?", (tid,))
    cx.commit()
    return True


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
    ep = cx.execute("SELECT active_epoch FROM coach_threads WHERE id=?", (thread_id,)).fetchone()
    epoch = (ep["active_epoch"] if ep else 1) or 1
    cur = cx.execute("INSERT INTO coach_messages (thread_id, sender_role, body, created_at, epoch) "
                     "VALUES (?,?,?,?,?)", (thread_id, sender_role, body, _now(), epoch))
    cx.commit()
    return cur.lastrowid


def messages(cx, thread_id, epoch=None):
    """Messages for a thread. With `epoch` (the thread's active_epoch), returns only
    that session's messages — the participant view, which hides pre-reactivation
    history. Without `epoch` (the owner transcript), returns the full history."""
    if epoch is None:
        rows = cx.execute("SELECT id, sender_role, body, created_at, epoch FROM coach_messages "
                          "WHERE thread_id=? ORDER BY id", (thread_id,)).fetchall()
    else:
        rows = cx.execute("SELECT id, sender_role, body, created_at, epoch FROM coach_messages "
                          "WHERE thread_id=? AND epoch=? ORDER BY id", (thread_id, epoch)).fetchall()
    return [dict(r) for r in rows]


def mark_read(cx, thread_id, role):
    col = "coach_last_read_at" if role == "coach" else "member_last_read_at"
    cx.execute(f"UPDATE coach_threads SET {col}=? WHERE id=?", (_now(), thread_id))
    cx.commit()


def unread_count(cx, thread_id, role):
    col = "coach_last_read_at" if role == "coach" else "member_last_read_at"
    other = "member" if role == "coach" else "coach"
    row = cx.execute(f"SELECT {col} AS lr, active_epoch FROM coach_threads WHERE id=?",
                     (thread_id,)).fetchone()
    if not row:
        return 0
    last = row["lr"] or ""
    ep = row["active_epoch"] or 1
    return cx.execute("SELECT COUNT(*) FROM coach_messages WHERE thread_id=? AND sender_role=? "
                      "AND epoch=? AND created_at > ?", (thread_id, other, ep, last)).fetchone()[0]


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
