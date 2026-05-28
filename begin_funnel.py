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
