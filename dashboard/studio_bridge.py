"""Pure sqlite store for Studio.com bridge Flow B free-month claims.

No Flask dependency. The caller passes in a sqlite3 connection (``cx``); every
write commits. Tracks one claim row per email so the free-month grant for a
studio.com/drglen joiner is idempotent.
"""

from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_table(cx):
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS studio_bridge_claims (
          email TEXT PRIMARY KEY,
          signup_via TEXT,
          status TEXT NOT NULL DEFAULT 'pending',
          sub_id INTEGER,
          created_at TEXT,
          granted_at TEXT
        )
        """
    )
    cx.commit()


def record_pending(cx, email, *, signup_via):
    cx.execute(
        "INSERT OR IGNORE INTO studio_bridge_claims "
        "(email, signup_via, status, created_at) VALUES (?,?, 'pending', ?)",
        (email, signup_via, _now()),
    )
    cx.execute(
        "UPDATE studio_bridge_claims SET signup_via=? WHERE email=?",
        (signup_via, email),
    )
    cx.commit()


def mark_granted(cx, email, sub_id):
    cx.execute(
        "UPDATE studio_bridge_claims "
        "SET status='granted', sub_id=?, granted_at=? WHERE email=?",
        (sub_id, _now(), email),
    )
    cx.commit()


def already_granted(cx, email):
    row = cx.execute(
        "SELECT status FROM studio_bridge_claims WHERE email=?", (email,)
    ).fetchone()
    return bool(row) and row["status"] == "granted"


def get(cx, email):
    row = cx.execute(
        "SELECT * FROM studio_bridge_claims WHERE email=?", (email,)
    ).fetchone()
    return dict(row) if row else None
