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
    # When a cart entered 'checking_out' (see claim_for_checkout). NULL for a cart
    # that was never claimed. Lets a stale claim (process died mid-checkout: deploy
    # restart, OOM, SIGKILL between the claim and mark_ordered) be told apart from a
    # genuinely in-flight one, instead of locking the member out of checkout forever.
    try:
        cx.execute("ALTER TABLE carts ADD COLUMN claimed_at TEXT")
    except Exception:
        pass  # already present
    cx.commit()


def get_or_create(cx, token, email=""):
    """Get or create an open cart. Returns token (may differ from argument).

    Resolution order:
    1. If an open cart exists for `token` AND this visitor owns it, return `token`.
    2. If `email` is non-empty and owns an open cart, return that existing token.
    3. If `token` exists but is non-open status, mint fresh token and create cart.
    4. Otherwise, create cart with `token` and return `token`.

    Step 1's ownership test is the same guard `merge` already carries: an open cart
    stamped with a DIFFERENT member's email is never handed back just because the
    caller presented its token. The token travels in a browser cookie, and this
    codebase explicitly models shared household devices -- without the test, the
    next person on that browser (anonymous, or signed in as someone else) writes
    into the previous member's cart, and that member is charged for it.

    Callers MUST use the returned token, not the argument.
    """
    token = (token or "").strip()
    if not token:
        raise ValueError("token required")
    norm_email = _norm_email(email)

    # 1. Check if an open cart exists for this token AND belongs to this visitor
    row = cx.execute(
        "SELECT email FROM carts WHERE token=? AND status='open'", (token,)
    ).fetchone()
    if row:
        owner = _norm_email(row[0])
        if not owner or owner == norm_email:
            return token
        # else: open, but owned by a DIFFERENT member -- fall through and give this
        # visitor a cart of their own rather than that member's.

    # 2. If email is given, check if it owns an open cart
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
        # MANDATORY on Postgres -- do NOT remove as "dead code because SQLite works
        # without it". psycopg leaves the transaction in an ABORTED state after a
        # failed statement, so the very next statement (the recovery SELECT below)
        # raises InFailedSqlTransaction instead of returning the winner's token, and
        # this whole self-heal becomes inert in production while staying green in a
        # SQLite-only test suite. The INSERT above is the first statement after a
        # commit, so this rollback discards nothing.
        try:
            cx.rollback()
        except Exception:
            pass
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

    A cart that is open but already owned by a DIFFERENT member's email is never
    touched -- not reassigned, not folded from, not marked merged. It is treated
    exactly like an unknown token and left completely alone. Without this guard a
    shared browser's stale `rm_cart` cookie (still pointing at the PREVIOUS member's
    open cart) would get folded onto and closed under the NEW member's identity,
    charging one member for another member's items.
    """
    email = _norm_email(email)
    if not email:
        raise ValueError("email required")
    anon_token = (anon_token or "").strip()

    member_token = open_token_for_email(cx, email)

    anon_mergeable = False
    if anon_token:
        row = cx.execute(
            "SELECT status, email FROM carts WHERE token=?", (anon_token,)
        ).fetchone()
        if row and row[0] == "open":
            owner_email = row[1] or ""
            if owner_email == email:
                return anon_token          # already this member's cart
            if not owner_email:
                anon_mergeable = True      # genuinely anonymous -- safe to fold
            # else: open but owned by a DIFFERENT member -- never touched, fall through

    if not anon_mergeable:
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
            # MANDATORY on Postgres -- see the identical note in `get_or_create`.
            # Without it the aborted transaction makes `open_token_for_email`'s
            # SELECT raise, and the anonymous cart is lost instead of folded.
            # The UPDATE above is the first statement after a commit, so this
            # rollback discards nothing.
            try:
                cx.rollback()
            except Exception:
                pass
            # Race: another request just created the member's cart. Re-read and fold.
            member_token = open_token_for_email(cx, email)
            if member_token:
                _fold_cart_items(cx, anon_token, member_token)
                return member_token
            # If re-read still finds nothing, the error was something else
            raise

    _fold_cart_items(cx, anon_token, member_token)
    return member_token


def _fold_cart_items(cx, src_token, dst_token):
    """Fold `src_token`'s items into `dst_token` (higher quantity wins per line,
    never summed), then delete src's items and mark its cart 'merged'. The one
    fold implementation shared by `merge` (both its normal path and its
    race-recovery path) and by stale-claim recovery in the checkout route, rather
    than each writing its own subtly different version."""
    for it in items(cx, src_token):
        row = cx.execute(
            "SELECT qty FROM cart_items WHERE token=? AND slug=? AND fmt=?",
            (dst_token, it["slug"], it["format"]),
        ).fetchone()
        if row:
            if int(it["qty"]) > int(row[0]):
                cx.execute(
                    "UPDATE cart_items SET qty=? WHERE token=? AND slug=? AND fmt=?",
                    (_clamp(it["qty"]), dst_token, it["slug"], it["format"]),
                )
        else:
            cx.execute(
                "INSERT INTO cart_items(token, slug, fmt, qty, source, added_at) "
                "VALUES (?,?,?,?,?,?)",
                (dst_token, it["slug"], it["format"], _clamp(it["qty"]),
                 it["source"], _now_iso()),
            )
    cx.execute("DELETE FROM cart_items WHERE token=?", (src_token,))
    cx.execute(
        "UPDATE carts SET status='merged', claimed_at=NULL, updated_at=? WHERE token=?",
        (_now_iso(), src_token),
    )
    _touch(cx, dst_token)
    cx.commit()


def _new_token_for(email):
    """Deterministic fallback token when a member needs a cart and has none.
    Never used for anonymous carts, which get a random token from the route layer."""
    import hashlib
    return "cart:" + hashlib.sha1(_norm_email(email).encode()).hexdigest()[:24]


def checking_out_for_email(cx, email):
    """True if this member already has a cart mid-checkout (status='checking_out').

    Checked by the route BEFORE calling `merge`/`claim_for_checkout`: `merge`'s own
    fallback (mint a fresh cart when the member has no OPEN cart) cannot tell "no
    cart at all" apart from "the real cart is just busy" -- a concurrent second
    request would otherwise get handed a brand-new EMPTY cart instead of being
    refused, orphaning the real (claimed) cart's items."""
    email = _norm_email(email)
    if not email:
        return False
    row = cx.execute(
        "SELECT 1 FROM carts WHERE email=? AND status='checking_out' LIMIT 1", (email,)
    ).fetchone()
    return bool(row)


def stale_claim_for_email(cx, email, cutoff_iso):
    """Token of a `checking_out` cart for this email whose claim is OLDER than
    `cutoff_iso` (i.e. genuinely abandoned -- the process behind it died before
    releasing or completing it), else "". ISO-8601 UTC timestamps (as produced by
    `_now_iso()` everywhere in this module) compare correctly as plain strings, so
    no datetime parsing is needed here.

    Compares against COALESCE(claimed_at, updated_at), not claimed_at alone: a row
    claimed before the `claimed_at` column existed (an old deploy's in-flight
    checkout during the upgrade window) would have a NULL claimed_at forever, which
    would make it permanently un-stale-able -- i.e. permanently un-recoverable --
    under any ordering of deploys. `updated_at` is always set (by `claim_for_checkout`
    itself, and by every other write in this module), so it is always a safe
    fallback anchor for staleness.

    A cart whose claim is NEWER than the cutoff is not returned -- that is a
    genuinely in-flight concurrent checkout, not a stale one, and the caller must
    still refuse it (409), not "recover" it out from under the request that is
    legitimately still running."""
    email = _norm_email(email)
    if not email:
        return ""
    row = cx.execute(
        "SELECT token FROM carts WHERE email=? AND status='checking_out' "
        "AND COALESCE(claimed_at, updated_at) < ? LIMIT 1",
        (email, cutoff_iso),
    ).fetchone()
    return row[0] if row else ""


def release_or_fold_stale_claim(cx, token, email):
    """Recovery for a `checking_out` cart confirmed to have NO order behind it (see
    app.py's `_recover_stale_cart_claim`). Never blindly reopens it: if the member
    came back while it was stuck and added an item, `get_or_create`/`add_item`
    attaches their email to a brand-new SECOND open cart (there is no way for those
    routes to know the old one is merely stuck rather than gone) -- and a bare
    `UPDATE carts SET status='open'` on the stale cart would then violate
    `ux_carts_open_email` (one open cart per email). Left to a swallowed exception,
    that silently strands the stale cart's items forever, which is exactly the
    "release it back to open" half of the fix failing to happen for the customer
    most likely to hit it: the one who tried again after getting stuck.

    - No other open cart for this email -> release this one back to 'open' (the
      common case: the member has not tried again yet).
    - Another open cart already exists -> fold this cart's items into it (same
      higher-quantity-wins rule as `merge`, via the shared `_fold_cart_items`) and
      mark this one 'merged' instead of 'open', so the unique index is never at
      risk and nothing is orphaned."""
    email = _norm_email(email)
    other_open = ""
    if email:
        row = cx.execute(
            "SELECT token FROM carts WHERE email=? AND status='open' AND token<>? LIMIT 1",
            (email, token),
        ).fetchone()
        other_open = row[0] if row else ""
    if other_open:
        _fold_cart_items(cx, token, other_open)
    else:
        release_claim(cx, token)


def claim_for_checkout(cx, token):
    """Atomically claim an open cart for checkout. Returns True iff THIS call won the
    race (rowcount==1). A losing caller must not proceed to price/charge -- another
    checkout for this same cart is already in flight, and letting both continue is
    what produces two orders and two Stripe sessions for one cart."""
    now = _now_iso()
    cur = cx.execute(
        "UPDATE carts SET status='checking_out', claimed_at=?, updated_at=? "
        "WHERE token=? AND status='open'",
        (now, now, token),
    )
    cx.commit()
    return cur.rowcount == 1


def release_claim(cx, token):
    """Revert a claimed-but-not-completed cart back to open, so a checkout that failed
    after claiming (bad address, pricing error, unexpected exception, or a recovered
    stale claim that turned out to have no order behind it) does not lock the customer
    out of their own cart."""
    cx.execute(
        "UPDATE carts SET status='open', claimed_at=NULL, updated_at=? "
        "WHERE token=? AND status='checking_out'",
        (_now_iso(), token),
    )
    cx.commit()


def mark_ordered(cx, token, checkout_ref):
    cx.execute(
        "UPDATE carts SET status='ordered', checkout_ref=?, claimed_at=NULL, updated_at=? "
        "WHERE token=?",
        ((checkout_ref or "").strip(), _now_iso(), token),
    )
    cx.commit()
