"""Persistent per-client fulfillment preferences. Today there is exactly one:
whether this client (by email) collects in person, so the order builder can
pre-check Pickup for them. Mirrors client_prices.py — pure functions over a
sqlite connection (testable).

Nothing writes this except an explicit operator toggle on the order builder.
Creating or saving an order NEVER writes it: ticking Pickup on one order is an
override for that order alone. That is why a Biofield hand-off invoice, which
sends pickup=True as a deliberate shipping courtesy, cannot silently teach the
system that every biofield client picks up.
"""
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS client_prefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            pickup_default INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            UNIQUE(email)
        )
    """)
    cx.commit()


def set_pickup_default(cx, email, value):
    """Upsert this client's pickup default. Explicit operator action only."""
    email = _norm(email)
    if not email:
        raise ValueError("email required")
    cx.execute(
        "INSERT INTO client_prefs (email, pickup_default, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(email) DO UPDATE SET pickup_default=excluded.pickup_default, "
        "updated_at=excluded.updated_at",
        (email, 1 if value else 0, _now()))
    cx.commit()


def get_pickup_default(cx, email):
    """True when this client collects in person by default. Unknown -> False.

    "Unknown" includes a `client_prefs` table that does not exist yet: the table is
    created lazily by the console panel, so an operator who has never opened it must
    still be able to take an order. Reading is on the order path — it must never raise
    and must never CREATE the table. False is the safe direction: it charges shipping,
    whereas a wrong True ships physical goods for free."""
    email = _norm(email)
    if not email:
        return False
    try:
        row = cx.execute("SELECT pickup_default FROM client_prefs WHERE email=?",
                         (email,)).fetchone()
    except sqlite3.OperationalError:
        return False          # table absent — nobody has set a preference yet
    return bool(row[0]) if row else False
