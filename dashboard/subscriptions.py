"""Subscriptions table + CRUD + escalating loyalty tier math.

Pure module — no Flask, no Stripe, no QBO imports.  All I/O goes through a
sqlite3 connection passed in by the caller so tests can use :memory:.
"""

import calendar
import json
import sqlite3
from datetime import datetime, timezone

from dashboard import customers as _customers

# ---------------------------------------------------------------------------
# Tier math
# ---------------------------------------------------------------------------

# Loyalty discount % keyed on active months (order_count): a +2%/month climb from
# 3% (first active month) to 25% (month 12+), clamped at the top. Paused months
# don't advance order_count (climb holds); only cancel resets it to 0.
SUBSCRIBE_TIERS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


def tier_for(n: int) -> int:
    """Subscriber discount % for a member with *n* completed active months
    (order_count). Clamped at the top step (25%)."""
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
           next_charge_date: str, order_count: int = 0) -> int:
    """Insert a new active subscription and return its id.

    order_count is the number of orders ALREADY placed on this subscription. At
    sign-up the setup checkout charges the 1st order (at tier_for(0)=3%), so the
    subscription is created with order_count=1 — that way the first SCHEDULED charge
    reads tier_for(1)=5% (the 2nd active month), climbing the 3→25% loyalty curve.
    """
    now = _now_iso()
    cur = cx.execute(
        """INSERT INTO subscriptions
               (email, stripe_customer_id, stripe_payment_method_id,
                items_json, cadence_months, status, order_count,
                next_charge_date, ship_address_json, skip_next,
                created_at, updated_at)
           VALUES (?,?,?,?,?,'active',?,?,?,0,?,?)""",
        (email, stripe_customer_id, stripe_payment_method_id,
         json.dumps(items or []), cadence_months, int(order_count),
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


def migrate_add_membership_columns(cx) -> None:
    """Add kind + amount_cents columns if missing. Safe on every startup."""
    for ddl in (
        "ALTER TABLE subscriptions ADD COLUMN kind TEXT NOT NULL DEFAULT 'product'",
        "ALTER TABLE subscriptions ADD COLUMN amount_cents INTEGER NOT NULL DEFAULT 0",
    ):
        try:
            cx.execute(ddl)
            cx.commit()
        except Exception:
            pass


def migrate_add_term_cap_column(cx):
    """Idempotent: add term_charges_total (NULL = uncapped) for fixed-term memberships."""
    cols = {r[1] for r in cx.execute("PRAGMA table_info(subscriptions)")}
    if "term_charges_total" not in cols:
        cx.execute("ALTER TABLE subscriptions ADD COLUMN term_charges_total INTEGER")
        cx.commit()


def migrate_add_attribution_column(cx):
    """Idempotent: add attributed_practitioner_id (TEXT, NULL default) — the
    Supabase practitioner id string that owns this Continuous Care membership,
    for fee-share credit in later tasks."""
    cols = {r[1] for r in cx.execute("PRAGMA table_info(subscriptions)")}
    if "attributed_practitioner_id" not in cols:
        cx.execute("ALTER TABLE subscriptions ADD COLUMN attributed_practitioner_id TEXT")
        cx.commit()


def create_membership(cx, *, email, stripe_customer_id, stripe_payment_method_id,
                      amount_cents, next_charge_date, cadence_months=1,
                      term_charges_total=None, initial_order_count=0,
                      attributed_practitioner_id=None) -> int:
    """Insert an active flat-amount membership subscription (no product items).
    The first charge lands on next_charge_date. term_charges_total caps total charges
    (NULL = uncapped, legacy behavior); initial_order_count records charges already taken
    at checkout (e.g. 1 when month 1 was charged in the checkout session).
    attributed_practitioner_id is the Supabase practitioner id (string) that owns this
    membership for fee-share credit purposes (NULL = unattributed)."""
    now = _now_iso()
    cur = cx.execute(
        """INSERT INTO subscriptions
               (email, stripe_customer_id, stripe_payment_method_id, items_json,
                cadence_months, status, order_count, next_charge_date, ship_address_json,
                skip_next, created_at, updated_at, kind, amount_cents, term_charges_total,
                attributed_practitioner_id)
           VALUES (?,?,?,?,?,'active',?,?,?,0,?,?, 'membership', ?, ?, ?)""",
        (email, stripe_customer_id, stripe_payment_method_id, "[]",
         int(cadence_months), int(initial_order_count), next_charge_date, "{}", now, now,
         int(amount_cents), (int(term_charges_total) if term_charges_total is not None else None),
         (str(attributed_practitioner_id) if attributed_practitioner_id else None)),
    )
    cx.commit()
    try:
        _customers.find_or_create_by_email(cx, email=email)
    except Exception:
        pass
    return cur.lastrowid


def active_memberships_by_email(cx, email) -> list:
    rows = cx.execute(
        "SELECT * FROM subscriptions WHERE email=? AND status='active' AND kind='membership'"
        " ORDER BY id", (email,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def category_for(cx, email) -> str:
    """Classify a member into 'none' | 'trial' | 'full' | 'paused'.

    Source of truth = the member's active kind=membership subscription (mirrors
    pause_membership_by_email / set_membership_cadence_by_email, which treat the
    member's membership as rows[0]):
      paused  -> active membership sub with skip_next set
      full    -> active membership sub, not paused, order_count >= 1 (>=1 real $99 charge cleared)
      trial   -> active membership sub, not paused, order_count == 0 (free first month)
      none    -> no active membership sub

    Product volume/quantity pricing is now open to everyone regardless of category
    (member-gating and the trial upgrade-credit accrual were both retired); this
    classification still drives the membership board and pause/cadence logic.
    """
    rows = active_memberships_by_email(cx, email)
    if not rows:
        # NOTE: spec §1 defensive fallback (a biofield_trial *grant* with no
        # membership sub → 'trial') is deferred to the credit-accrual PR, which
        # owns the memberships-grant table. For PR1 pricing this 'none' is safe
        # (errs toward no-discount); the _is_paid_member gate is grant-aware.
        return "none"
    return classify_sub(rows[0])


def classify_sub(sub) -> str:
    """Classify a SINGLE active kind=membership subscription row into
    'trial' | 'full' | 'paused' (see category_for for the rules). Pure — takes a
    row dict, no DB. Used per-row by the /console/members board so each sub is
    classified on its own merits."""
    if sub.get("skip_next"):
        return "paused"
    return "full" if int(sub.get("order_count") or 0) >= 1 else "trial"


def list_active_memberships(cx) -> list:
    """ONE row per member for the /console/members board: every active
    kind=membership subscription, deduped to a single sub per email. Cancelled subs
    (status!='active') and product subs are excluded.

    Dedup keeps the OLDEST sub (lowest id) per email so the board's classification
    agrees with category_for (which keys off active_memberships_by_email's rows[0],
    ORDER BY id) — and therefore with the _is_paid_member pricing gate. Without this,
    a buyer holding two active membership subs (e.g. a $1-biofield-trial sub plus a
    later separate join — join idempotency is not unified across funnel paths) would
    appear in two columns and inflate the counts. Returned newest-member-first."""
    rows = cx.execute(
        "SELECT * FROM subscriptions WHERE status='active' AND kind='membership' "
        "ORDER BY id ASC"
    ).fetchall()
    seen, out = set(), []
    for r in rows:
        d = _row_to_dict(r)
        em = (d.get("email") or "").strip().lower()
        if em in seen:
            continue
        seen.add(em)
        out.append(d)
    out.sort(key=lambda d: d.get("created_at") or "", reverse=True)
    return out


def member_board_row(sub, *, name="", credit_cents=0) -> dict:
    """Build one /console/members row from a membership sub. Trial rows carry the
    accrued upgrade `credit_cents` (the call-list signal); paused rows carry a
    `resume_date` (next_charge_date advanced by the cadence); full rows carry
    neither. `started` = the sub's created_at."""
    cat = classify_sub(sub)
    next_charge = sub.get("next_charge_date") or ""
    cadence = int(sub.get("cadence_months") or 1)
    order_count = int(sub.get("order_count") or 0)
    row = {
        "email": sub.get("email") or "",
        "name": name or "",
        "category": cat,
        "plan_cents": int(sub.get("amount_cents") or 0),
        "started": sub.get("created_at") or "",
        "next_charge_date": next_charge,
        "tier": tier_for(order_count),
        "order_count": order_count,
    }
    if cat == "paused":
        row["resume_date"] = add_months(next_charge, cadence) if next_charge else ""
    elif cat == "trial":
        row["credit_cents"] = int(credit_cents or 0)
    return row


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


def pause_membership_by_email(cx, email: str) -> dict | None:
    """Soft-pause the member's active membership: set skip_next so the NEXT charge
    is skipped (one cycle), then auto-resumes the cycle after. Preserves order_count
    (loyalty tier) -- a pause loses nothing, unlike cancel. Idempotent. Returns
    {sub_id, paused_charge_date, resume_date} or None when no active membership."""
    rows = active_memberships_by_email(cx, email)
    if not rows:
        return None
    sub = rows[0]
    if not sub.get("skip_next"):
        set_skip_next(cx, sub["id"], True)
    nc = sub["next_charge_date"]
    return {"sub_id": sub["id"], "paused_charge_date": nc,
            "resume_date": add_months(nc, int(sub.get("cadence_months") or 1))}


def set_membership_cadence_by_email(cx, email: str, months: int) -> dict | None:
    """Settle the member's active membership into a slower recurring rhythm
    (e.g. every 2 or 3 months). Clears any one-time skip_next (cadence supersedes
    a single skip). Idempotent. Returns {sub_id, cadence_months} or None."""
    rows = active_memberships_by_email(cx, email)
    if not rows:
        return None
    sub = rows[0]
    set_cadence(cx, sub["id"], int(months))
    if sub.get("skip_next"):
        set_skip_next(cx, sub["id"], False)
    return {"sub_id": sub["id"], "cadence_months": int(months)}


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


# ---------------------------------------------------------------------------
# founding columns (Task 1 — idempotent migration)
# ---------------------------------------------------------------------------

def migrate_add_founding_columns(cx) -> None:
    """Add founding launch columns if missing. Safe on every startup."""
    for ddl in (
        "ALTER TABLE subscriptions ADD COLUMN founding INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN founding_state TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE subscriptions ADD COLUMN founding_slug TEXT NOT NULL DEFAULT ''",
    ):
        try:
            cx.execute(ddl)
            cx.commit()
        except Exception:
            pass


# Far-future sentinel: a pending reservation must never be picked up by list_due
# until mark_founding_active sets a real next_charge_date (the charge-on-ship event).
_FOUNDING_PENDING_DATE = "2999-01-01"


def create_founding_reservation(cx, *, email, stripe_customer_id,
                                stripe_payment_method_id, items, ship_address,
                                founding_slug) -> int:
    """Insert a pending founding product subscription (card vaulted, $0 today).
    order_count=0 and a far-future next_charge_date keep it out of list_due until
    the founding batch ships (mark_founding_active)."""
    now = _now_iso()
    cur = cx.execute(
        """INSERT INTO subscriptions
               (email, stripe_customer_id, stripe_payment_method_id, items_json,
                cadence_months, status, order_count, next_charge_date, ship_address_json,
                skip_next, created_at, updated_at, founding, founding_state, founding_slug)
           VALUES (?,?,?,?,1,'active',0,?,?,0,?,?,1,'pending',?)""",
        (email, stripe_customer_id, stripe_payment_method_id, json.dumps(items or []),
         _FOUNDING_PENDING_DATE, json.dumps(ship_address or {}), now, now, founding_slug),
    )
    cx.commit()
    return cur.lastrowid


def mark_founding_active(cx, sub_id: int, *, next_charge_date: str) -> None:
    """Flip a pending founding reservation to active after its first (on-ship) charge:
    record the first order and schedule the next autoship charge."""
    cx.execute(
        "UPDATE subscriptions SET founding_state='active', order_count=1,"
        " next_charge_date=?, updated_at=? WHERE id=?",
        (next_charge_date, _now_iso(), sub_id),
    )
    cx.commit()


def list_founding_pending(cx, founding_slug: str) -> list[dict]:
    """Reserved-but-not-yet-shipped founding subscriptions for a launch slug."""
    rows = cx.execute(
        "SELECT * FROM subscriptions WHERE founding=1 AND founding_state='pending'"
        " AND founding_slug=? AND status!='cancelled' ORDER BY id", (founding_slug,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_founding(cx, founding_slug: str) -> int:
    """Count of founding slots consumed for a launch (pending + active, not cancelled).

    Returns 0 if the subscriptions table does not exist yet. On a fresh database
    (no subscription ever written) the table is created lazily on first write, so
    a read-only founding path (e.g. the public /begin/founding/status counter or
    the reserve open-check) would otherwise raise 'no such table: subscriptions'
    and 500. No table means no reservations, which is exactly 0."""
    try:
        row = cx.execute(
            "SELECT COUNT(*) FROM subscriptions WHERE founding=1 AND founding_slug=?"
            " AND status!='cancelled'", (founding_slug,)
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0


def backfill_member_people(cx):
    """Ensure every current member (active membership subscription OR unexpired access
    grant) has a people row, so their personal portal is reachable via self-login.
    Reuses customers.find_or_create_by_email. Idempotent; returns count created."""
    now = _now_iso()
    rows = cx.execute(
        "SELECT DISTINCT email FROM subscriptions WHERE kind='membership' AND status='active' "
        "UNION SELECT DISTINCT email FROM memberships WHERE expires_at > ?", (now,)).fetchall()
    created = 0
    for (email,) in rows:
        em = (email or "").strip().lower()
        if not em:
            continue
        try:
            if cx.execute("SELECT 1 FROM people WHERE lower(email)=?", (em,)).fetchone():
                continue
            _customers.find_or_create_by_email(cx, email=em)
            created += 1
        except Exception:
            continue
    return created
