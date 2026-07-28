"""Persistent shopping cart (token-keyed). Pure: the caller passes cx.

Deliberately stores NO prices -- only slug/qty/fmt. `_price_cart` recomputes at
checkout, which is what makes a price change between adding and paying resolve
correctly by construction.

Schema note: `carts` is keyed by an app-generated TEXT token and `cart_items` by a
composite primary key, so nothing here needs an autoincrement id. `cur.lastrowid`
RAISES on the Postgres adapter, and this shape never reaches for it.
"""
from datetime import datetime, timezone

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
    token = (token or "").strip()
    if not token:
        raise ValueError("token required")
    row = cx.execute(
        "SELECT token FROM carts WHERE token=? AND status='open'", (token,)
    ).fetchone()
    if row:
        return token
    now = _now_iso()
    cx.execute(
        "INSERT INTO carts(token, email, status, checkout_ref, created_at, updated_at) "
        "VALUES (?,?,'open','',?,?)",
        (token, _norm_email(email), now, now),
    )
    cx.commit()
    return token


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
