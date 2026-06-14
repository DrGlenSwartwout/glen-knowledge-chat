# dashboard/points.py
"""Loyalty points ledger. Values stored in redemption-value CENTS (1 point = 5c).
Earn is on full-price spend only (caller decides eligibility); redeem is bounded
by balance here and by the price floor in dashboard.pricing.compute()."""


def init_points_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS points_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            delta_cents INTEGER NOT NULL,
            reason TEXT,
            order_ref TEXT,
            balance_after INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_points_email ON points_ledger(email)")
    cx.commit()


def balance(cx, email):
    row = cx.execute("SELECT COALESCE(SUM(delta_cents),0) FROM points_ledger WHERE email=?",
                     (email,)).fetchone()
    return int(row[0] or 0)


def _add(cx, email, delta_cents, reason, order_ref):
    # Read-then-write balance. SAFE only because the app uses ONE sqlite connection,
    # which serializes all operations. If this is ever called from pooled/parallel
    # connections, switch balance_after to an atomic INSERT subquery (or BEGIN IMMEDIATE)
    # or two concurrent redeems could both pass the guard and overdraw the balance.
    bal = balance(cx, email) + int(delta_cents)
    cx.execute("""INSERT INTO points_ledger(email,delta_cents,reason,order_ref,balance_after)
                  VALUES (?,?,?,?,?)""", (email, int(delta_cents), reason, order_ref, bal))
    cx.commit()
    return bal


def earn(cx, email, *, full_price_cents, earn_pct, order_ref):
    # earn_pct is a fraction in 0.0–1.0 (e.g. 0.05 = 5%), NOT a whole percent.
    delta = int(round(int(full_price_cents) * float(earn_pct)))
    return _add(cx, email, delta, "earn", order_ref)


def redeem(cx, email, *, value_cents, order_ref):
    value_cents = int(value_cents)
    if value_cents > balance(cx, email):
        raise ValueError("redeem exceeds balance")
    return _add(cx, email, -value_cents, "redeem", order_ref)


def has_entry(cx, *, order_ref, reason):
    """True if a ledger row already exists for this order_ref + reason (idempotency guard)."""
    row = cx.execute("SELECT 1 FROM points_ledger WHERE order_ref=? AND reason=? LIMIT 1",
                     (order_ref, reason)).fetchone()
    return row is not None


def credit(cx, email: str, *, value_cents: int, reason: str, order_ref: str) -> None:
    """Idempotent direct credit — bypasses earn_pct calculation. Safe to call repeatedly."""
    if not has_entry(cx, order_ref=order_ref, reason=reason):
        _add(cx, email, value_cents, reason, order_ref)
