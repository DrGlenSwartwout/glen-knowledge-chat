"""Family Plan store + entitlement. Pure sqlite, no Flask, no Stripe.

One plan per CAREGIVER. It is the single entitlement source for the whole
consented household — full scan results, member product pricing, the monthly
family shipment, and group coaching access all ask the same `covers()`.

Mirrors dashboard/coach_subscriptions.py: this module holds state only;
money-path correctness (idempotent first charge, no cron double-charge) lives in
the routes/cron, guarded by next_charge_at advancing only on success.

`past_due` still covers: a failed renewal must not blur a client's report
mid-month. The cron cancels after its retry budget, and cancelling stops cover.
"""

PLAN = {
    "amount_cents": 14700,   # the special offer, charged monthly
    "value_cents": 19700,    # the anchor shown alongside it
    "label": "Family Plan",
}

# Statuses that still entitle. `past_due` = grace while the cron retries.
ACTIVE_STATUSES = ("active", "past_due")

_DDL = """
CREATE TABLE IF NOT EXISTS family_subscriptions (
    caregiver_email TEXT PRIMARY KEY,
    amount_cents INTEGER,
    stripe_customer_id TEXT,
    payment_method_id TEXT,
    status TEXT,
    source TEXT,
    started_at TEXT,
    next_charge_at TEXT,
    last_charged_at TEXT,
    fail_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_familysub_due ON family_subscriptions(status, next_charge_at);
CREATE TABLE IF NOT EXISTS family_sub_charges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caregiver_email TEXT,
    amount_cents INTEGER,
    pi_id TEXT,
    status TEXT,
    charged_at TEXT
);
"""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _lc(email):
    return (email or "").strip().lower()


def init_family_plan_table(cx):
    cx.executescript(_DDL)
    cx.commit()


def activate(cx, caregiver_email, *, next_charge_at, customer_id=None,
             payment_method_id=None, source="stripe"):
    """Start (or restart) a caregiver's plan. A comped plan passes source='comp'
    with no card and no next_charge_at — same entitlement, never billed."""
    cx.execute(
        "INSERT INTO family_subscriptions (caregiver_email,amount_cents,stripe_customer_id,"
        "payment_method_id,status,source,started_at,next_charge_at,fail_count) "
        "VALUES (?,?,?,?,'active',?,?,?,0) "
        "ON CONFLICT(caregiver_email) DO UPDATE SET "
        "amount_cents=excluded.amount_cents, stripe_customer_id=excluded.stripe_customer_id, "
        "payment_method_id=excluded.payment_method_id, status='active', "
        "source=excluded.source, next_charge_at=excluded.next_charge_at, fail_count=0",
        (_lc(caregiver_email), PLAN["amount_cents"], customer_id, payment_method_id,
         source, _now(), next_charge_at))
    cx.commit()


def get(cx, caregiver_email):
    row = cx.execute("SELECT * FROM family_subscriptions WHERE caregiver_email=?",
                     (_lc(caregiver_email),)).fetchone()
    return dict(row) if row else None


def set_status(cx, caregiver_email, status):
    cx.execute("UPDATE family_subscriptions SET status=? WHERE caregiver_email=?",
               (status, _lc(caregiver_email)))
    cx.commit()


def is_active(cx, caregiver_email):
    """True when this email holds an entitling plan of their own."""
    email = _lc(caregiver_email)
    if not email:
        return False
    row = cx.execute(
        "SELECT 1 FROM family_subscriptions WHERE caregiver_email=? AND status IN (?,?)",
        (email, *ACTIVE_STATUSES)).fetchone()
    return row is not None


def covers(cx, email):
    """True when `email` is entitled — either they hold the plan, or a caregiver
    who holds it is linked to them with share_consent=1.

    Consent gates the caregiver's reach in both directions: a member who has
    revoked it is not viewable (household.can_view) and is not covered here.
    """
    email = _lc(email)
    if not email:
        return False
    if is_active(cx, email):
        return True
    from dashboard import household as _hh
    for cg in _hh.caregivers_for(cx, email):
        if cg["share_consent"] and is_active(cx, cg["primary_email"]):
            return True
    return False


def due(cx, today):
    """Billable subs whose next_charge_at has arrived. Includes both 'active' and
    'past_due': a past_due sub (a prior failed charge) still entitles the household
    (grace) and MUST be retried on each cron run so its fail_count can climb to the
    cancel threshold — otherwise a single failed payment would cover forever.
    Comped plans (source='comp', next_charge_at NULL) are never billable, excluded."""
    rows = cx.execute(
        "SELECT * FROM family_subscriptions WHERE status IN ('active','past_due') "
        "AND next_charge_at IS NOT NULL AND next_charge_at <= ? "
        "AND (source IS NULL OR source != 'comp') ORDER BY next_charge_at",
        (today,)).fetchall()
    return [dict(r) for r in rows]


def mark_charged(cx, caregiver_email, next_charge_at):
    cx.execute("UPDATE family_subscriptions SET next_charge_at=?, last_charged_at=?, "
               "fail_count=0, status='active' WHERE caregiver_email=?",
               (next_charge_at, _now(), _lc(caregiver_email)))
    cx.commit()


def mark_failed(cx, caregiver_email, retry_at):
    """Record a failed charge: bump fail_count, set past_due, and schedule the next
    retry by moving next_charge_at to retry_at (so the daily cron re-attempts on that
    date, not every day — spacing the dunning retries). A later success resets
    fail_count via mark_charged; enough failed attempts get the plan cancelled by the
    cron. Advancing next_charge_at here is a retry schedule, not a paid-cycle advance
    (the charge captured no money), so it does not violate the paid-once invariant."""
    cx.execute("UPDATE family_subscriptions SET fail_count=fail_count+1, status='past_due', "
               "next_charge_at=? WHERE caregiver_email=?", (retry_at, _lc(caregiver_email)))
    cx.commit()


def record_charge(cx, *, caregiver_email, amount_cents, pi_id, status):
    cx.execute("INSERT INTO family_sub_charges (caregiver_email,amount_cents,pi_id,status,charged_at) "
               "VALUES (?,?,?,?,?)", (_lc(caregiver_email), amount_cents, pi_id, status, _now()))
    cx.commit()
