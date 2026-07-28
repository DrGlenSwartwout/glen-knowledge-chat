"""Persistent shopping cart (token-keyed). Pure: the caller passes cx.

Deliberately stores NO prices -- only slug/qty/fmt. `_price_cart` recomputes at
checkout, which is what makes a price change between adding and paying resolve
correctly by construction.

Schema note: `carts` is keyed by an app-generated TEXT token and `cart_items` by a
composite primary key, so nothing here needs an autoincrement id. `cur.lastrowid`
RAISES on the Postgres adapter, and this shape never reaches for it.
"""
from datetime import datetime, timezone
import uuid

MAX_QTY = 99


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _norm_email(email):
    return (email or "").strip().lower()


def _clamp(qty):
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        qty = 1
    return max(1, min(qty, MAX_QTY))


def init_cart_tables(cx):
    cx.execute(
        """CREATE TABLE IF NOT EXISTS carts (
             token        TEXT PRIMARY KEY,
             email        TEXT NOT NULL DEFAULT '',
             status       TEXT NOT NULL DEFAULT 'open',
             checkout_ref TEXT NOT NULL DEFAULT '',
             created_at   TEXT NOT NULL,
             updated_at   TEXT NOT NULL
           )"""
    )
    cx.execute(
        """CREATE TABLE IF NOT EXISTS cart_items (
             token    TEXT NOT NULL,
             slug     TEXT NOT NULL,
             fmt      TEXT NOT NULL DEFAULT '',
             qty      INTEGER NOT NULL,
             source   TEXT NOT NULL DEFAULT '',
             added_at TEXT NOT NULL,
             PRIMARY KEY (token, slug, fmt)
           )"""
    )
    # One open cart per identified member. Partial index works on SQLite and Postgres.
    cx.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_carts_open_email "
        "ON carts(email) WHERE status='open' AND email<>''"
    )
    cx.commit()


def get_or_create(cx, token, email=""):
    """Get or create an open cart. Returns token (may differ from argument).

    Resolution order:
    1. If open cart exists for `token`, return `token`.
    2. If `email` is non-empty and owns an open cart, return that existing token.
    3. If `token` exists but is non-open status, mint fresh token and create cart.
    4. Otherwise, create cart with `token` and return `token`.

    Callers MUST use the returned token, not the argument.
    """
    token = (token or "").strip()
    if not token:
        raise ValueError("token required")

    # 1. Check if open cart exists for this token
    row = cx.execute(
        "SELECT token FROM carts WHERE token=? AND status='open'", (token,)
    ).fetchone()
    if row:
        return token

    # 2. If email is given, check if it owns an open cart
    norm_email = _norm_email(email)
    if norm_email:
        row = cx.execute(
            "SELECT token FROM carts WHERE email=? AND status='open' LIMIT 1", (norm_email,)
        ).fetchone()
        if row:
            return row[0]

    # 3. If token already exists (in any status), mint a fresh token
    row = cx.execute(
        "SELECT token FROM carts WHERE token=?", (token,)
    ).fetchone()
    if row:
        token = uuid.uuid4().hex

    # 4. Create the cart
    now = _now_iso()
    try:
        cx.execute(
            "INSERT INTO carts(token, email, status, checkout_ref, created_at, updated_at) "
            "VALUES (?,?,'open','',?,?)",
            (token, norm_email, now, now),
        )
        cx.commit()
        return token
    except Exception:
        # Race: another request just created the email's cart. Re-read step 2.
        if norm_email:
            row = cx.execute(
                "SELECT token FROM carts WHERE email=? AND status='open' LIMIT 1", (norm_email,)
            ).fetchone()
            if row:
                return row[0]
        # If re-read still finds nothing, the error was something else
        raise


def open_token_for_email(cx, email):
    email = _norm_email(email)
    if not email:
        return ""
    row = cx.execute(
        "SELECT token FROM carts WHERE email=? AND status='open' LIMIT 1", (email,)
    ).fetchone()
    return row[0] if row else ""


def _touch(cx, token):
    cx.execute("UPDATE carts SET updated_at=? WHERE token=?", (_now_iso(), token))


def add_item(cx, token, slug, qty=1, fmt="", source=""):
    slug = (slug or "").strip().lower()
    if not slug:
        raise ValueError("slug required")
    fmt = (fmt or "").strip().lower()
    qty = _clamp(qty)
    row = cx.execute(
        "SELECT qty FROM cart_items WHERE token=? AND slug=? AND fmt=?", (token, slug, fmt)
    ).fetchone()
    new_qty = _clamp((row[0] if row else 0) + qty)
    if row:
        cx.execute(
            "UPDATE cart_items SET qty=? WHERE token=? AND slug=? AND fmt=?",
            (new_qty, token, slug, fmt),
        )
    else:
        cx.execute(
            "INSERT INTO cart_items(token, slug, fmt, qty, source, added_at) VALUES (?,?,?,?,?,?)",
            (token, slug, fmt, new_qty, (source or "").strip(), _now_iso()),
        )
    _touch(cx, token)
    cx.commit()
    return new_qty


def set_qty(cx, token, slug, fmt, qty):
    slug = (slug or "").strip().lower()
    fmt = (fmt or "").strip().lower()
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        qty = 0
    if qty <= 0:
        cx.execute(
            "DELETE FROM cart_items WHERE token=? AND slug=? AND fmt=?", (token, slug, fmt)
        )
    else:
        cx.execute(
            "UPDATE cart_items SET qty=? WHERE token=? AND slug=? AND fmt=?",
            (_clamp(qty), token, slug, fmt),
        )
    _touch(cx, token)
    cx.commit()


def items(cx, token):
    rows = cx.execute(
        "SELECT slug, qty, fmt, source FROM cart_items WHERE token=? ORDER BY added_at, slug",
        (token,),
    ).fetchall()
    return [
        {"slug": r[0], "qty": int(r[1]), "format": r[2] or "", "source": r[3] or ""}
        for r in rows
    ]


def merge(cx, anon_token, email):
    """Fold an anonymous cart onto a member email and return the surviving token.

    Quantity rule: the HIGHER of the two wins, never the sum. The same bottle added
    on a phone and then a laptop is one intent repeated; summing would charge double.

    Idempotent: merging an already-merged or unknown token is a no-op that still
    returns the member's open cart token.
    """
    email = _norm_email(email)
    if not email:
        raise ValueError("email required")
    anon_token = (anon_token or "").strip()

    member_token = open_token_for_email(cx, email)

    anon_open = False
    if anon_token:
        row = cx.execute(
            "SELECT status, email FROM carts WHERE token=?", (anon_token,)
        ).fetchone()
        anon_open = bool(row) and row[0] == "open"
        if anon_open and (row[1] or "") == email:
            return anon_token          # already this member's cart

    if not anon_open:
        return member_token or get_or_create(cx, _new_token_for(email), email=email)

    if not member_token:
        try:
            cx.execute(
                "UPDATE carts SET email=?, updated_at=? WHERE token=?",
                (email, _now_iso(), anon_token),
            )
            cx.commit()
            return anon_token
        except Exception:
            # Race: another request just created the member's cart. Re-read and fold.
            member_token = open_token_for_email(cx, email)
            if member_token:
                # Fold the anon cart into the now-existing member cart
                for it in items(cx, anon_token):
                    row = cx.execute(
                        "SELECT qty FROM cart_items WHERE token=? AND slug=? AND fmt=?",
                        (member_token, it["slug"], it["format"]),
                    ).fetchone()
                    if row:
                        if int(it["qty"]) > int(row[0]):
                            cx.execute(
                                "UPDATE cart_items SET qty=? WHERE token=? AND slug=? AND fmt=?",
                                (_clamp(it["qty"]), member_token, it["slug"], it["format"]),
                            )
                    else:
                        cx.execute(
                            "INSERT INTO cart_items(token, slug, fmt, qty, source, added_at) "
                            "VALUES (?,?,?,?,?,?)",
                            (member_token, it["slug"], it["format"], _clamp(it["qty"]),
                             it["source"], _now_iso()),
                        )
                cx.execute("DELETE FROM cart_items WHERE token=?", (anon_token,))
                cx.execute(
                    "UPDATE carts SET status='merged', updated_at=? WHERE token=?",
                    (_now_iso(), anon_token),
                )
                _touch(cx, member_token)
                cx.commit()
                return member_token
            # If re-read still finds nothing, the error was something else
            raise

    for it in items(cx, anon_token):
        row = cx.execute(
            "SELECT qty FROM cart_items WHERE token=? AND slug=? AND fmt=?",
            (member_token, it["slug"], it["format"]),
        ).fetchone()
        if row:
            if int(it["qty"]) > int(row[0]):
                cx.execute(
                    "UPDATE cart_items SET qty=? WHERE token=? AND slug=? AND fmt=?",
                    (_clamp(it["qty"]), member_token, it["slug"], it["format"]),
                )
        else:
            cx.execute(
                "INSERT INTO cart_items(token, slug, fmt, qty, source, added_at) "
                "VALUES (?,?,?,?,?,?)",
                (member_token, it["slug"], it["format"], _clamp(it["qty"]),
                 it["source"], _now_iso()),
            )
    cx.execute("DELETE FROM cart_items WHERE token=?", (anon_token,))
    cx.execute(
        "UPDATE carts SET status='merged', updated_at=? WHERE token=?",
        (_now_iso(), anon_token),
    )
    _touch(cx, member_token)
    cx.commit()
    return member_token


def _new_token_for(email):
    """Deterministic fallback token when a member needs a cart and has none.
    Never used for anonymous carts, which get a random token from the route layer."""
    import hashlib
    return "cart:" + hashlib.sha1(_norm_email(email).encode()).hexdigest()[:24]


def mark_ordered(cx, token, checkout_ref):
    cx.execute(
        "UPDATE carts SET status='ordered', checkout_ref=?, updated_at=? WHERE token=?",
        ((checkout_ref or "").strip(), _now_iso(), token),
    )
    cx.commit()
