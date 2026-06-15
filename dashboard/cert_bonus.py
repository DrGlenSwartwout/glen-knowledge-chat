"""Certification bonus Biofield module (pure: cx + args, no Flask).

A committed certification enrollee earns a bonus Biofield each elapsed month
(12 total) plus one per module completed (1-12). This module holds:

- a commitment store (cert_commitments)
- an idempotent grant ledger (cert_bonus_grants)
- a pure due_bonuses() calculator (no cx)
"""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_tables(cx):
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS cert_commitments (
          email TEXT PRIMARY KEY, kind TEXT, started_at TEXT,
          active INTEGER NOT NULL DEFAULT 1, created_at TEXT, updated_at TEXT
        )
        """
    )
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS cert_bonus_grants (
          email TEXT, kind TEXT, idx INTEGER, todo_id INTEGER, granted_at TEXT,
          PRIMARY KEY (email, kind, idx)
        )
        """
    )
    cx.commit()


def set_commitment(cx, email, *, kind, started_at):
    now = _now()
    cx.execute(
        """
        INSERT INTO cert_commitments (email, kind, started_at, active, created_at, updated_at)
        VALUES (?, ?, ?, 1, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
          kind = excluded.kind,
          started_at = excluded.started_at,
          active = 1,
          updated_at = excluded.updated_at
        """,
        (email, kind, started_at, now, now),
    )
    cx.commit()


def get_commitment(cx, email):
    row = cx.execute(
        "SELECT * FROM cert_commitments WHERE email = ?", (email,)
    ).fetchone()
    return dict(row) if row else None


def clear_commitment(cx, email):
    cx.execute(
        "UPDATE cert_commitments SET active = 0, updated_at = ? WHERE email = ?",
        (_now(), email),
    )
    cx.commit()


def list_active(cx):
    rows = cx.execute(
        "SELECT * FROM cert_commitments WHERE active = 1 ORDER BY email"
    ).fetchall()
    return [dict(r) for r in rows]


def record_grant(cx, email, *, kind, idx, todo_id=None):
    cx.execute(
        """
        INSERT OR IGNORE INTO cert_bonus_grants
          (email, kind, idx, todo_id, granted_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (email, kind, idx, todo_id, _now()),
    )
    cx.commit()


def granted_pairs(cx, email):
    rows = cx.execute(
        "SELECT kind, idx FROM cert_bonus_grants WHERE email = ?", (email,)
    ).fetchall()
    return {(r["kind"], r["idx"]) for r in rows}


def due_bonuses(*, started_at, modules_completed, granted, today):
    """Pure calculator. No cx.

    Returns the sorted list of (kind, idx) bonus pairs that are owed but not
    yet granted.
    """
    sy, sm, sd = (int(x) for x in started_at.split("-"))
    ty, tm, td = (int(x) for x in today.split("-"))

    me = (ty - sy) * 12 + (tm - sm)
    if td < sd:
        me -= 1
    me = max(me, 0)

    monthly = {("monthly", m) for m in range(1, min(me, 12) + 1)}
    level = {
        ("level", n)
        for n in range(1, max(0, min(int(modules_completed or 0), 12)) + 1)
    }
    return sorted((monthly | level) - set(granted))
