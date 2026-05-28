"""Journey-state engine for the /begin progressive-disclosure funnel.

Pure functions over a sqlite3 connection. Routes in app.py manage the
connection + _db_lock; tests pass their own connection. See
docs/superpowers/specs/2026-05-28-progressive-disclosure-funnel-design.md
"""

from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_journey_tables(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS journey_state (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            email           TEXT,
            first_name      TEXT,
            ref_slug        TEXT,
            current_rung    TEXT    DEFAULT 'arrival',
            unlocked_gates  TEXT    DEFAULT '[]',
            awareness_stage TEXT    DEFAULT 'unknown',
            path            TEXT    DEFAULT 'none',
            tos_agreed_at   TEXT,
            tos_version     TEXT,
            last_signal     TEXT,
            created_at      TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_journey_session ON journey_state(session_id)")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_journey_email   ON journey_state(email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS journey_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            session_id  TEXT,
            email       TEXT,
            trigger     TEXT NOT NULL,
            detail      TEXT DEFAULT '',
            rung_before TEXT,
            rung_after  TEXT
        )
    """)
    cx.commit()


RUNGS = ["arrival", "listening", "inquire", "personalize", "free_tier",
         "explore_voice", "assess", "choose_path", "ascend", "advocate"]
RUNG_INDEX = {r: i for i, r in enumerate(RUNGS)}

# All accepted unlock triggers. The page (slice 1) only fires the first six;
# the rest are accepted so the engine spine is forward-compatible with later
# slices (rooms built in slices 4-6).
VALID_TRIGGERS = {
    "load", "video", "scroll", "question", "name", "email", "tos",
    "voice", "scan", "quiz", "paid_fork", "purchase", "share_video",
}

# Gate keys stored in unlocked_gates (email/tos drive their own columns, but
# are still recorded as gates for completeness).
GATE_TRIGGERS = VALID_TRIGGERS - {"load"}


def compute_rung(gates, email, tos_agreed):
    """Derive the highest rung reached. Monotonic in ladder order. The
    free_tier rung specifically requires BOTH an email and ToS agreement."""
    gates = set(gates or ())
    rung = "arrival"
    if "video" in gates or "scroll" in gates:
        rung = "listening"
    if "question" in gates:
        rung = "inquire"
    if "name" in gates:
        rung = "personalize"
    if email and tos_agreed:
        rung = "free_tier"
    if "voice" in gates:
        rung = "explore_voice"
    if "scan" in gates or "quiz" in gates:
        rung = "assess"
    if "paid_fork" in gates:
        rung = "choose_path"
    if "purchase" in gates:
        rung = "ascend"
    if "share_video" in gates:
        rung = "advocate"
    return rung
