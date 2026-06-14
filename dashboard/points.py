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
    bal = balance(cx, email) + int(delta_cents)
    cx.execute("""INSERT INTO points_ledger(email,delta_cents,reason,order_ref,balance_after)
                  VALUES (?,?,?,?,?)""", (email, int(delta_cents), reason, order_ref, bal))
    cx.commit()
    return bal


def earn(cx, email, *, full_price_cents, earn_pct, order_ref):
    delta = int(round(int(full_price_cents) * float(earn_pct)))
    return _add(cx, email, delta, "earn", order_ref)


def redeem(cx, email, *, value_cents, order_ref):
    value_cents = int(value_cents)
    if value_cents > balance(cx, email):
        raise ValueError("redeem exceeds balance")
    return _add(cx, email, -value_cents, "redeem", order_ref)
