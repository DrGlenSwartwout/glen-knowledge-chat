"""Pure sqlite store for per-email Biofield checkout readiness state.

No Flask dependency. The caller passes in a sqlite3 connection (``cx``); every
write commits. Used to track payment + readiness gate flags (photo on file,
intake confirmed, scan confirmed, booking) for the $300 Biofield offer.
"""

from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_table(cx):
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS biofield_readiness (
          email TEXT PRIMARY KEY,
          paid_at TEXT, paid_via TEXT, order_ref TEXT,
          photo_on_file INTEGER NOT NULL DEFAULT 0, photo_path TEXT,
          intake_confirmed INTEGER NOT NULL DEFAULT 0,
          scan_confirmed INTEGER NOT NULL DEFAULT 0,
          booked_at TEXT, created_at TEXT, updated_at TEXT
        )
        """
    )
    cx.commit()


def _ensure_row(cx, email):
    now = _now()
    cx.execute(
        "INSERT OR IGNORE INTO biofield_readiness (email, created_at, updated_at) "
        "VALUES (?,?,?)",
        (email, now, now),
    )
    cx.commit()


def seed_paid(cx, email, *, via, order_ref):
    _ensure_row(cx, email)
    now = _now()
    # COALESCE keeps the first payment — a second call must not overwrite it.
    cx.execute(
        "UPDATE biofield_readiness SET "
        "paid_at=COALESCE(paid_at, ?), "
        "paid_via=COALESCE(paid_via, ?), "
        "order_ref=COALESCE(order_ref, ?), "
        "updated_at=? WHERE email=?",
        (now, via, order_ref, now, email),
    )
    cx.commit()


def set_photo_on_file(cx, email, path):
    _ensure_row(cx, email)
    now = _now()
    cx.execute(
        "UPDATE biofield_readiness SET photo_on_file=1, photo_path=?, updated_at=? "
        "WHERE email=?",
        (path, now, email),
    )
    cx.commit()


def set_intake_confirmed(cx, email, value):
    _ensure_row(cx, email)
    now = _now()
    cx.execute(
        "UPDATE biofield_readiness SET intake_confirmed=?, updated_at=? WHERE email=?",
        (1 if value else 0, now, email),
    )
    cx.commit()


def set_scan_confirmed(cx, email, value):
    _ensure_row(cx, email)
    now = _now()
    cx.execute(
        "UPDATE biofield_readiness SET scan_confirmed=?, updated_at=? WHERE email=?",
        (1 if value else 0, now, email),
    )
    cx.commit()


def set_booked(cx, email):
    _ensure_row(cx, email)
    now = _now()
    cx.execute(
        "UPDATE biofield_readiness SET booked_at=?, updated_at=? WHERE email=?",
        (now, now, email),
    )
    cx.commit()


def get(cx, email):
    row = cx.execute(
        "SELECT * FROM biofield_readiness WHERE email=?", (email,)
    ).fetchone()
    if row is None:
        return None
    return dict(row)
