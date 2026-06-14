"""Subscriptions table + CRUD + escalating loyalty tier math.

Pure module — no Flask, no Stripe, no QBO imports.  All I/O goes through a
sqlite3 connection passed in by the caller so tests can use :memory:.
"""

import calendar
import json
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Tier math
# ---------------------------------------------------------------------------

SUBSCRIBE_TIERS = [5, 10, 15]   # percent discount at order_count 0, 1, 2+


def tier_for(n: int) -> int:
    """Return the subscriber discount % for a subscriber with *n* completed
    orders (order_count).  Capped at the highest tier."""
    idx = min(int(n), len(SUBSCRIBE_TIERS) - 1)
    return SUBSCRIBE_TIERS[idx]


# ---------------------------------------------------------------------------
# Date helper (no external deps)
# ---------------------------------------------------------------------------

def add_months(yyyy_mm_dd: str, n: int) -> str:
    """Add *n* calendar months to a YYYY-MM-DD date string.

    Month overflow is handled by clamping the day to the last day of the
    resulting month (e.g. add_months('2026-01-31', 1) → '2026-02-28').
    """
    dt = datetime.strptime(yyyy_mm_dd, "%Y-%m-%d")
    year = dt.year
    month = dt.month + n
    # normalise month > 12
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    # clamp day to month-end
    max_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, max_day)
    return f"{year:04d}-{month:02d}-{day:02d}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS subscriptions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    email                   TEXT NOT NULL,
    stripe_customer_id      TEXT NOT NULL,
    stripe_payment_method_id TEXT NOT NULL,
    items_json              TEXT NOT NULL DEFAULT '[]',
    cadence_months          INTEGER NOT NULL DEFAULT 1,
    status                  TEXT NOT NULL DEFAULT 'active',
    order_count             INTEGER NOT NULL DEFAULT 0,
    next_charge_date        TEXT NOT NULL,
    ship_address_json       TEXT NOT NULL DEFAULT '{}',
    skip_next               INTEGER NOT NULL DEFAULT 0,
    last_notified_date      TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    cancelled_at            TEXT
);
CREATE INDEX IF NOT EXISTS idx_subs_status_date
    ON subscriptions (status, next_charge_date);
CREATE INDEX IF NOT EXISTS idx_subs_email
    ON subscriptions (email);
"""


def init_subscriptions_table(cx) -> None:
    """Create the subscriptions table + indexes if they don't exist."""
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            cx.execute(stmt)
    cx.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_dict(row) -> dict:
    """Convert a sqlite3.Row to a plain dict with JSON fields unpacked."""
    d = dict(row)
    for field in ("items_json", "ship_address_json"):
        raw = d.pop(field, None)
        key = field.replace("_json", "")
        try:
            d[key] = json.loads(raw) if raw else ([] if "items" in field else {})
        except (TypeError, json.JSONDecodeError):
            d[key] = [] if "items" in field else {}
    return d


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create(cx, *, email: str, stripe_customer_id: str,
           stripe_payment_method_id: str, items: list,
           cadence_months: int, ship_address: dict,
           next_charge_date: str) -> int:
    """Insert a new active subscription and return its id."""
    now = _now_iso()
    cur = cx.execute(
        """INSERT INTO subscriptions
               (email, stripe_customer_id, stripe_payment_method_id,
                items_json, cadence_months, status, order_count,
                next_charge_date, ship_address_json, skip_next,
                created_at, updated_at)
           VALUES (?,?,?,?,?,'active',0,?,?,0,?,?)""",
        (email, stripe_customer_id, stripe_payment_method_id,
         json.dumps(items or []), cadence_months,
         next_charge_date, json.dumps(ship_address or {}),
         now, now),
    )
    cx.commit()
    return cur.lastrowid


def get(cx, sub_id: int) -> dict | None:
    """Return a subscription row as a dict, or None if not found."""
    row = cx.execute(
        "SELECT * FROM subscriptions WHERE id = ?", (sub_id,)
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_active_by_email(cx, email: str) -> list[dict]:
    """Return all active subscriptions for an email address."""
    rows = cx.execute(
        "SELECT * FROM subscriptions WHERE email = ? AND status = 'active'"
        " ORDER BY id", (email,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_manageable_by_email(cx, email: str) -> list[dict]:
    """Return a member's manageable subscriptions (active + paused, NOT cancelled),
    so the portal can show — and let them resume — a paused plan."""
    rows = cx.execute(
        "SELECT * FROM subscriptions WHERE email = ? AND status != 'cancelled'"
        " ORDER BY id", (email,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_due(cx, *, as_of: str) -> list[dict]:
    """Return active subscriptions (skip_next=0) whose next_charge_date <= as_of,
    ordered by next_charge_date ascending."""
    rows = cx.execute(
        """SELECT * FROM subscriptions
           WHERE status = 'active'
             AND skip_next = 0
             AND next_charge_date <= ?
           ORDER BY next_charge_date ASC""",
        (as_of,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def advance_after_charge(cx, sub_id: int) -> None:
    """Increment order_count and advance next_charge_date by cadence_months."""
    row = cx.execute(
        "SELECT order_count, cadence_months, next_charge_date FROM subscriptions WHERE id = ?",
        (sub_id,),
    ).fetchone()
    if row is None:
        return
    new_count = row["order_count"] + 1
    new_date = add_months(row["next_charge_date"], row["cadence_months"])
    cx.execute(
        "UPDATE subscriptions SET order_count=?, next_charge_date=?, updated_at=? WHERE id=?",
        (new_count, new_date, _now_iso(), sub_id),
    )
    cx.commit()


def consume_skip(cx, sub_id: int) -> None:
    """Advance next_charge_date by cadence_months and clear skip_next flag.
    Does NOT increment order_count (the cycle is skipped, not fulfilled)."""
    row = cx.execute(
        "SELECT cadence_months, next_charge_date FROM subscriptions WHERE id = ?",
        (sub_id,),
    ).fetchone()
    if row is None:
        return
    new_date = add_months(row["next_charge_date"], row["cadence_months"])
    cx.execute(
        "UPDATE subscriptions SET next_charge_date=?, skip_next=0, updated_at=? WHERE id=?",
        (new_date, _now_iso(), sub_id),
    )
    cx.commit()


def set_skip_next(cx, sub_id: int, value: bool) -> None:
    cx.execute(
        "UPDATE subscriptions SET skip_next=?, updated_at=? WHERE id=?",
        (1 if value else 0, _now_iso(), sub_id),
    )
    cx.commit()


def set_status(cx, sub_id: int, status: str) -> None:
    """Set subscription status.  When cancelling, reset order_count to 0 and
    record cancelled_at."""
    now = _now_iso()
    if status == "cancelled":
        cx.execute(
            "UPDATE subscriptions SET status=?, order_count=0, cancelled_at=?, updated_at=?"
            " WHERE id=?",
            (status, now, now, sub_id),
        )
    else:
        cx.execute(
            "UPDATE subscriptions SET status=?, updated_at=? WHERE id=?",
            (status, now, sub_id),
        )
    cx.commit()


def set_cadence(cx, sub_id: int, months: int) -> None:
    """Store a new cadence_months value (caller decides whether to recompute
    next_charge_date)."""
    cx.execute(
        "UPDATE subscriptions SET cadence_months=?, updated_at=? WHERE id=?",
        (months, _now_iso(), sub_id),
    )
    cx.commit()


def set_next_charge_date(cx, sub_id: int, date: str) -> None:
    cx.execute(
        "UPDATE subscriptions SET next_charge_date=?, updated_at=? WHERE id=?",
        (date, _now_iso(), sub_id),
    )
    cx.commit()


# ---------------------------------------------------------------------------
# failed_count column (added in Task 4 — idempotent migration)
# ---------------------------------------------------------------------------

def migrate_add_failed_count(cx) -> None:
    """Add failed_count column to subscriptions table if it doesn't exist yet.
    Safe to call on every startup — the ALTER is inside a try/except."""
    try:
        cx.execute(
            "ALTER TABLE subscriptions ADD COLUMN failed_count INTEGER NOT NULL DEFAULT 0"
        )
        cx.commit()
    except Exception:
        pass  # column already exists — ignore


def bump_failed_count(cx, sub_id: int) -> None:
    """Increment failed_count by 1. Used by the charge scheduler on payment failure."""
    cx.execute(
        "UPDATE subscriptions SET failed_count = COALESCE(failed_count, 0) + 1, updated_at=?"
        " WHERE id=?",
        (_now_iso(), sub_id),
    )
    cx.commit()


def reset_failed_count(cx, sub_id: int) -> None:
    """Reset failed_count to 0 after a successful charge."""
    cx.execute(
        "UPDATE subscriptions SET failed_count = 0, updated_at=? WHERE id=?",
        (_now_iso(), sub_id),
    )
    cx.commit()


def list_skip_due(cx, *, as_of: str) -> list[dict]:
    """Return active subscriptions with skip_next=1 whose next_charge_date <= as_of.
    These are excluded from list_due but still need to be advanced (skipped cycle)."""
    rows = cx.execute(
        """SELECT * FROM subscriptions
           WHERE status = 'active'
             AND skip_next = 1
             AND next_charge_date <= ?
           ORDER BY next_charge_date ASC""",
        (as_of,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_heads_up_due(cx, *, as_of: str, lead_days: int = 3) -> list[dict]:
    """Return active subscriptions whose next_charge_date is within lead_days days
    and whose last_notified_date differs from next_charge_date (i.e. haven't been
    notified for this upcoming charge yet)."""
    rows = cx.execute(
        """SELECT * FROM subscriptions
           WHERE status = 'active'
             AND next_charge_date > ?
             AND next_charge_date <= date(?, '+' || ? || ' days')
             AND (last_notified_date IS NULL OR last_notified_date != next_charge_date)
           ORDER BY next_charge_date ASC""",
        (as_of, as_of, str(lead_days)),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def set_last_notified_date(cx, sub_id: int, date: str) -> None:
    """Record that a heads-up email was sent for the upcoming next_charge_date."""
    cx.execute(
        "UPDATE subscriptions SET last_notified_date=?, updated_at=? WHERE id=?",
        (date, _now_iso(), sub_id),
    )
    cx.commit()
