"""Slice 3 of Condition Support Programs: operator override store for a
client's eye-condition support-program key. Pure sqlite, no Flask.

A client's condition is normally auto-detected from their health tags (see
`_condition_key_from_tags`/`_client_condition_for` in app.py). This store lets
Glen/Rae set (or clear) an explicit override when the tags are missing,
ambiguous (a bare "glaucoma"/"cataract" with no qualifier), or simply wrong.
The override ALWAYS wins over auto-detection.
"""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS client_conditions (
            email TEXT PRIMARY KEY,
            condition_key TEXT,
            set_by TEXT,
            updated_at TEXT
        )""")


def get(cx, email):
    """The override condition_key for email (lowercased), or None if unset."""
    email = (email or "").strip().lower()
    if not email:
        return None
    r = cx.execute("SELECT condition_key FROM client_conditions WHERE email=?",
                   (email,)).fetchone()
    if r is None:
        return None
    return r[0] or None


def set(cx, email, condition_key, set_by):
    """Set (or replace) the override for email (lowercased)."""
    email = (email or "").strip().lower()
    now = _now()
    cx.execute("""
        INSERT INTO client_conditions (email, condition_key, set_by, updated_at)
        VALUES (?,?,?,?)
        ON CONFLICT(email) DO UPDATE SET
            condition_key=excluded.condition_key,
            set_by=excluded.set_by,
            updated_at=excluded.updated_at
        """, (email, condition_key, set_by, now))
    cx.commit()


def clear(cx, email):
    """Remove the override for email (lowercased), if any."""
    email = (email or "").strip().lower()
    cx.execute("DELETE FROM client_conditions WHERE email=?", (email,))
    cx.commit()
