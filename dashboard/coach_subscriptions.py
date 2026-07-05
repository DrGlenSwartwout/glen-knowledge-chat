"""Paid coaching subscription store (arc slice 2b). Pure sqlite. One subscription
per member (Rae $100/mo or Glen $200/mo). Recurring billing = card on file +
a monthly cron; this module holds state only. Money-path correctness (idempotent
first charge, no cron double-charge) lives in the routes/cron, guarded by
next_charge_at advancing only on success."""

TIERS = {
    "rae":  {"amount_cents": 10000, "service": "evox",     "label": "Rae"},
    "glen": {"amount_cents": 20000, "service": "biofield", "label": "Dr. Glen"},
}

_DDL = """
CREATE TABLE IF NOT EXISTS coach_subscriptions (
    member_email TEXT PRIMARY KEY,
    tier TEXT,
    amount_cents INTEGER,
    stripe_customer_id TEXT,
    payment_method_id TEXT,
    status TEXT,
    started_at TEXT,
    next_charge_at TEXT,
    last_charged_at TEXT,
    fail_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_coachsub_due ON coach_subscriptions(status, next_charge_at);
CREATE TABLE IF NOT EXISTS coach_sub_charges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_email TEXT,
    tier TEXT,
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


def init_sub_tables(cx):
    cx.executescript(_DDL)
    cx.commit()


def get_sub(cx, email):
    row = cx.execute("SELECT * FROM coach_subscriptions WHERE member_email=?",
                     (_lc(email),)).fetchone()
    return dict(row) if row else None


def create_sub(cx, *, email, tier, customer_id, payment_method_id, next_charge_at):
    amount = TIERS.get(tier, {}).get("amount_cents", 0)
    cx.execute(
        "INSERT INTO coach_subscriptions (member_email,tier,amount_cents,stripe_customer_id,"
        "payment_method_id,status,started_at,next_charge_at,fail_count) "
        "VALUES (?,?,?,?,?, 'active', ?, ?, 0) "
        "ON CONFLICT(member_email) DO UPDATE SET tier=excluded.tier, amount_cents=excluded.amount_cents, "
        "stripe_customer_id=excluded.stripe_customer_id, payment_method_id=excluded.payment_method_id, "
        "status='active', next_charge_at=excluded.next_charge_at, fail_count=0",
        (_lc(email), tier, amount, customer_id, payment_method_id, _now(), next_charge_at))
    cx.commit()


def set_status(cx, email, status):
    cx.execute("UPDATE coach_subscriptions SET status=? WHERE member_email=?",
               (status, _lc(email)))
    cx.commit()


def mark_charged(cx, email, next_charge_at):
    cx.execute("UPDATE coach_subscriptions SET next_charge_at=?, last_charged_at=?, "
               "fail_count=0, status='active' WHERE member_email=?",
               (next_charge_at, _now(), _lc(email)))
    cx.commit()


def mark_failed(cx, email):
    cx.execute("UPDATE coach_subscriptions SET fail_count=fail_count+1, status='past_due' "
               "WHERE member_email=?", (_lc(email),))
    cx.commit()


def record_charge(cx, *, email, tier, amount_cents, pi_id, status):
    cx.execute("INSERT INTO coach_sub_charges (member_email,tier,amount_cents,pi_id,status,charged_at) "
               "VALUES (?,?,?,?,?,?)", (_lc(email), tier, amount_cents, pi_id, status, _now()))
    cx.commit()


def due(cx, today):
    rows = cx.execute("SELECT * FROM coach_subscriptions WHERE status='active' "
                      "AND next_charge_at <= ? ORDER BY next_charge_at", (today,)).fetchall()
    return [dict(r) for r in rows]
