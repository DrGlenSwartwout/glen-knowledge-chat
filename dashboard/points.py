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
            created_at TEXT DEFAULT (datetime('now')),
            scope TEXT NOT NULL DEFAULT 'rm'
        )""")
    cols = [r[1] for r in cx.execute("PRAGMA table_info(points_ledger)").fetchall()]
    if "scope" not in cols:
        cx.execute("ALTER TABLE points_ledger ADD COLUMN scope TEXT NOT NULL DEFAULT 'rm'")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_points_email ON points_ledger(email)")
    cx.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_points_order_ref_reason_scope "
               "ON points_ledger(order_ref, reason, scope)")
    cx.commit()


def balance(cx, email, *, scope="rm"):
    row = cx.execute("SELECT COALESCE(SUM(delta_cents),0) FROM points_ledger "
                     "WHERE email=? AND scope=?", (email, scope)).fetchone()
    return int(row[0] or 0)


def _add(cx, email, delta_cents, reason, order_ref, scope="rm"):
    # balance_after is a NON-authoritative snapshot (balance() is SUM(delta_cents)).
    # INSERT OR IGNORE + the UNIQUE(order_ref,reason,scope) index makes this atomically
    # idempotent across processes: a concurrent duplicate (redirect + webhook settling
    # the same order under gunicorn's 2 workers) inserts exactly one row. Return the
    # re-read true balance so the result is correct whether we inserted or ignored.
    snapshot = balance(cx, email, scope=scope) + int(delta_cents)
    cx.execute("""INSERT OR IGNORE INTO points_ledger(email,delta_cents,reason,order_ref,balance_after,scope)
                  VALUES (?,?,?,?,?,?)""",
               (email, int(delta_cents), reason, order_ref, snapshot, scope))
    cx.commit()
    return balance(cx, email, scope=scope)


def earn(cx, email, *, full_price_cents, earn_pct, order_ref, scope="rm"):
    # earn_pct is a fraction in 0.0–1.0 (e.g. 0.05 = 5%), NOT a whole percent.
    delta = int(round(int(full_price_cents) * float(earn_pct)))
    return _add(cx, email, delta, "earn", order_ref, scope=scope)


def redeem(cx, email, *, value_cents, order_ref, scope="rm"):
    value_cents = int(value_cents)
    if value_cents > balance(cx, email, scope=scope):
        raise ValueError("redeem exceeds balance")
    return _add(cx, email, -value_cents, "redeem", order_ref, scope=scope)


def earned_by_reason(cx, email, reason, *, scope="rm"):
    """Total cents earned (positive credits only) for a given reason — e.g. per-tier
    referral earnings. 0 when none."""
    row = cx.execute(
        "SELECT COALESCE(SUM(delta_cents),0) FROM points_ledger "
        "WHERE email=? AND reason=? AND scope=? AND delta_cents > 0",
        (email, reason, scope)).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def has_entry(cx, *, order_ref, reason, scope="rm"):
    """True if a ledger row already exists for this order_ref + reason (idempotency guard)."""
    row = cx.execute("SELECT 1 FROM points_ledger WHERE order_ref=? AND reason=? AND scope=? LIMIT 1",
                     (order_ref, reason, scope)).fetchone()
    return row is not None


def credit(cx, email: str, *, value_cents: int, reason: str, order_ref: str, scope: str = "rm") -> None:
    """Idempotent direct credit — bypasses earn_pct calculation. Safe to call repeatedly."""
    if not has_entry(cx, order_ref=order_ref, reason=reason, scope=scope):
        _add(cx, email, value_cents, reason, order_ref, scope=scope)


def spend(cx, email, *, value_cents, reason, order_ref, scope="rm"):
    """Idempotent guarded debit (the mirror of credit): no-op if a row with this
    (order_ref, reason, scope) already exists; else redeems value_cents CLAMPED to the
    current balance under a distinct reason. Unlike redeem() this never raises on an
    over-request (it spends what's there) and it carries a caller-chosen reason so the
    (order_ref, reason) pair is the idempotency key. Returns cents actually spent."""
    if has_entry(cx, order_ref=order_ref, reason=reason, scope=scope):
        return 0
    amt = min(int(value_cents), balance(cx, email, scope=scope))
    if amt <= 0:
        return 0
    _add(cx, email, -amt, reason, order_ref, scope=scope)
    return amt
