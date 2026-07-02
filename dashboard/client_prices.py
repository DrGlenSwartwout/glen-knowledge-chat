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


# Reserved slug for a client's FLAT rate across all Functional Formulations. It
# never matches a real product slug, and is filtered out of the per-SKU views.
FF_FLAT_SLUG = "__all_ff__"


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
    """{slug: price_cents} of PER-SKU prices for a client (the FF-flat reserved
    slug is excluded) — one read to price a whole cart."""
    email = _norm(email)
    if not email:
        return {}
    return {r[0]: int(r[1]) for r in cx.execute(
        "SELECT slug, price_cents FROM client_prices WHERE email=? AND slug!=?",
        (email, FF_FLAT_SLUG)).fetchall()}


def set_ff_flat(cx, email, price_cents):
    """This client's flat rate for ALL Functional Formulations."""
    set_price(cx, email, FF_FLAT_SLUG, price_cents, note="all-FFs flat rate")


def get_ff_flat(cx, email):
    """This client's flat FF rate in cents, or None."""
    return get_price(cx, email, FF_FLAT_SLUG)


def remove_ff_flat(cx, email):
    return remove(cx, email, FF_FLAT_SLUG)


def list_for(cx, email):
    """A client's PER-SKU special prices as [{slug, price_cents, note, updated_at}]
    (the FF-flat reserved slug is excluded — read it with get_ff_flat)."""
    email = _norm(email)
    if not email:
        return []
    return [{"slug": r[0], "price_cents": int(r[1]), "note": r[2], "updated_at": r[3]}
            for r in cx.execute(
                "SELECT slug, price_cents, note, updated_at FROM client_prices "
                "WHERE email=? AND slug!=? ORDER BY slug", (email, FF_FLAT_SLUG)).fetchall()]


def remove(cx, email, slug):
    cur = cx.execute("DELETE FROM client_prices WHERE email=? AND slug=?",
                     (_norm(email), (slug or "").strip()))
    cx.commit()
    return cur.rowcount > 0


def clients_with_prices(cx):
    """[{email, count}] — every client who has any special price, for the console list."""
    return [{"email": r[0], "count": int(r[1])} for r in cx.execute(
        "SELECT email, COUNT(*) FROM client_prices GROUP BY email ORDER BY email").fetchall()]
