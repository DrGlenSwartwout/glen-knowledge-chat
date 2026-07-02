"""Persistent per-client special pricing. A client (by email) can have a
negotiated price for specific product slugs (e.g. J.C.'s Functional-Formulation
rate) that auto-applies to orders built for them. Mirrors the per-practitioner
pricing pattern. Pure functions over a sqlite connection (testable).

Precedence in the in-house order pricer: an explicit per-line override the owner
typed on THIS order wins; else this client price; else the standard volume/list.
"""
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS client_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            slug TEXT NOT NULL,
            price_cents INTEGER NOT NULL,
            note TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(email, slug)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_client_prices_email ON client_prices(email)")
    cx.commit()


def set_price(cx, email, slug, price_cents, note=None):
    """Upsert a client's special price for a slug. price_cents < 0 is rejected;
    use remove() to clear one."""
    email, slug = _norm(email), (slug or "").strip()
    pc = int(price_cents)
    if not email or not slug or pc < 0:
        raise ValueError("email, slug, and non-negative price_cents required")
    cx.execute(
        "INSERT INTO client_prices (email, slug, price_cents, note, updated_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(email, slug) DO UPDATE SET "
        "price_cents=excluded.price_cents, note=excluded.note, updated_at=excluded.updated_at",
        (email, slug, pc, note, _now()))
    cx.commit()


def get_price(cx, email, slug):
    """This client's special price for slug in cents, or None if none set."""
    email, slug = _norm(email), (slug or "").strip()
    if not email or not slug:
        return None
    row = cx.execute("SELECT price_cents FROM client_prices WHERE email=? AND slug=?",
                     (email, slug)).fetchone()
    return int(row[0]) if row else None


def price_map(cx, email):
    """{slug: price_cents} for a client — one read to price a whole cart."""
    email = _norm(email)
    if not email:
        return {}
    return {r[0]: int(r[1]) for r in cx.execute(
        "SELECT slug, price_cents FROM client_prices WHERE email=?", (email,)).fetchall()}


def list_for(cx, email):
    """A client's special prices as [{slug, price_cents, note, updated_at}]."""
    email = _norm(email)
    if not email:
        return []
    return [{"slug": r[0], "price_cents": int(r[1]), "note": r[2], "updated_at": r[3]}
            for r in cx.execute(
                "SELECT slug, price_cents, note, updated_at FROM client_prices "
                "WHERE email=? ORDER BY slug", (email,)).fetchall()]


def remove(cx, email, slug):
    cur = cx.execute("DELETE FROM client_prices WHERE email=? AND slug=?",
                     (_norm(email), (slug or "").strip()))
    cx.commit()
    return cur.rowcount > 0


def clients_with_prices(cx):
    """[{email, count}] — every client who has any special price, for the console list."""
    return [{"email": r[0], "count": int(r[1])} for r in cx.execute(
        "SELECT email, COUNT(*) FROM client_prices GROUP BY email ORDER BY email").fetchall()]
