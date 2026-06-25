"""1b product savings coupons. A coupon = a single % discount on ONE product,
time-limited. Self-coupons are minted on journey-stage completion and applied at
checkout via the EXISTING coupon_pct path (clamped to the wholesale floor in
dashboard.pricing.compute). Idempotent; safe under the app's single sqlite conn.
Reads build dicts by hand so they don't depend on the connection's row_factory."""
import uuid
from datetime import datetime, timedelta

_COLS = ["code", "product_slug", "pct", "kind", "email", "session_id",
         "minted_at", "expires_at", "redeemed_at", "order_ref"]
_SEL = ",".join(_COLS)


def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _row(r):
    return dict(zip(_COLS, r)) if r else None


def init_coupons_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS coupons (
            code TEXT PRIMARY KEY,
            product_slug TEXT NOT NULL,
            pct INTEGER NOT NULL,
            kind TEXT NOT NULL DEFAULT 'self',
            email TEXT,
            session_id TEXT,
            minted_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            redeemed_at TEXT,
            order_ref TEXT
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_coupons_email ON coupons(email)")
    cx.commit()


def mint_self(cx, *, email, product_slug, pct=15, days=10):
    now = _now()
    # Earn-once: one self-coupon per (email, product_slug) for life. ANY prior
    # coupon — active, redeemed, OR expired-unused — blocks a re-mint and is
    # returned as-is. The 10-day window is use-it-or-lose-it, not a refill.
    existing = _row(cx.execute(
        f"SELECT {_SEL} FROM coupons WHERE email=? AND product_slug=? AND kind='self' "
        "ORDER BY minted_at DESC LIMIT 1",
        (email, product_slug)).fetchone())
    if existing:
        return existing
    code = "SELF-" + uuid.uuid4().hex[:8].upper()
    expires = (datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
               + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cx.execute("INSERT INTO coupons(code,product_slug,pct,kind,email,minted_at,expires_at) "
               "VALUES (?,?,?,?,?,?,?)",
               (code, product_slug, int(pct), "self", email, now, expires))
    cx.commit()
    return _row(cx.execute(f"SELECT {_SEL} FROM coupons WHERE code=?", (code,)).fetchone())


def validate(cx, code, *, product_slug=None):
    now = _now()
    r = _row(cx.execute(
        f"SELECT {_SEL} FROM coupons WHERE code=? AND redeemed_at IS NULL AND expires_at > ?",
        ((code or "").strip(), now)).fetchone())
    if not r:
        return None
    if product_slug is not None and r["product_slug"] != product_slug:
        return None
    return r


def mark_redeemed(cx, code, *, order_ref):
    cur = cx.execute(
        "UPDATE coupons SET redeemed_at=?, order_ref=? WHERE code=? AND redeemed_at IS NULL",
        (_now(), str(order_ref), (code or "").strip()))
    cx.commit()
    return cur.rowcount > 0


def wallet(cx, *, email):
    now = _now()
    rows = cx.execute(
        f"SELECT {_SEL} FROM coupons WHERE email=? AND redeemed_at IS NULL AND expires_at > ? "
        "ORDER BY expires_at ASC", (email, now)).fetchall()
    return [_row(r) for r in rows]
