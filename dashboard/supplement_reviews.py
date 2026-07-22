"""Free product review store. A client submits a supplement they take; Dr. Glen's
formulation-analyzer produces a review that lands in their portal. One row per
(email, product). State machine mirrors the biofield report flow:

    requested -> ai_draft -> confirmed          (+ rejected side-state)

- requested: client asked; nothing generated yet.
- ai_draft:  analyzer wrote a review; client sees "in progress", not the text.
- confirmed: Glen approved in console; review is visible on the portal.

Transitions never downgrade (a confirmed review can't be walked back or overwritten),
so a re-run or a late analyzer draft can't un-publish an approved review. LOG_DB (SQLite).
Distinct from dashboard/product_reviews.py, which is customer testimonials of products."""
import datetime
import os
import re

from dashboard import db

_RANK = {"requested": 0, "ai_draft": 1, "confirmed": 2}


def enabled():
    """True when SUPPLEMENT_REVIEW_ENABLED is set truthy. Default off (dark)."""
    return (os.environ.get("SUPPLEMENT_REVIEW_ENABLED", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(e):
    return (e or "").strip().lower()


def _key(name, brand):
    """Stable dedupe key for a product: case- and whitespace-insensitive name|brand."""
    raw = "%s|%s" % ((name or "").strip().lower(), (brand or "").strip().lower())
    return re.sub(r"\s+", " ", raw)


def init_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS supplement_reviews (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT NOT NULL,
            product_name  TEXT NOT NULL,
            product_brand TEXT,
            product_key   TEXT NOT NULL,
            source        TEXT,
            status        TEXT NOT NULL,
            review_text   TEXT,
            requested_at  TEXT,
            drafted_at    TEXT,
            confirmed_at  TEXT,
            updated_at    TEXT,
            UNIQUE(email, product_key)
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_suprev_status ON supplement_reviews(status)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_suprev_email ON supplement_reviews(email)")
    cx.commit()


def create_request(cx, email, product_name, product_brand="", source="portal"):
    e = _norm(email)
    name = (product_name or "").strip()
    if not e or not name:
        return {"created": False, "id": None, "status": None}
    key = _key(name, product_brand)
    row = cx.execute("SELECT id, status FROM supplement_reviews WHERE email=? AND product_key=?",
                     (e, key)).fetchone()
    if row:
        return {"created": False, "id": row[0], "status": row[1]}
    now = _now()
    try:
        cur = cx.execute(
            "INSERT INTO supplement_reviews "
            "(email, product_name, product_brand, product_key, source, status, requested_at, updated_at) "
            "VALUES (?,?,?,?,?, 'requested', ?, ?)",
            (e, name, (product_brand or "").strip(), key, (source or "").strip(), now, now))
        cx.commit()
        return {"created": True, "id": cur.lastrowid, "status": "requested"}
    except db.IntegrityError:
        # lost a UNIQUE race → return the existing row rather than 500ing.
        row = cx.execute("SELECT id, status FROM supplement_reviews WHERE email=? AND product_key=?",
                         (e, key)).fetchone()
        return {"created": False, "id": row[0] if row else None,
                "status": row[1] if row else None}


def set_draft(cx, review_id, review_text):
    """Attach the analyzer's review and move requested -> ai_draft. Never touches a
    confirmed review (idempotent + never-downgrade)."""
    row = cx.execute("SELECT status FROM supplement_reviews WHERE id=?", (review_id,)).fetchone()
    if not row:
        return {"status": None}
    cur = row[0]
    if _RANK.get(cur, 0) >= _RANK["confirmed"]:
        return {"status": cur}  # confirmed: leave content and status alone
    cx.execute("UPDATE supplement_reviews SET review_text=?, status='ai_draft', drafted_at=?, updated_at=? WHERE id=?",
               (review_text, _now(), _now(), review_id))
    cx.commit()
    return {"status": "ai_draft"}


def set_status(cx, review_id, status, by=None):
    """Promote/finalize. Refuses to move a review to a lower rank (never-downgrade);
    'rejected' is allowed only from a non-confirmed state."""
    row = cx.execute("SELECT status FROM supplement_reviews WHERE id=?", (review_id,)).fetchone()
    if not row:
        return {"status": None}
    cur = row[0]
    if status == cur:
        return {"status": cur}
    if status == "rejected":
        if cur == "confirmed":
            return {"status": cur}
        cx.execute("UPDATE supplement_reviews SET status='rejected', updated_at=? WHERE id=?",
                   (_now(), review_id))
        cx.commit()
        return {"status": "rejected"}
    if status in _RANK:
        if _RANK[status] <= _RANK.get(cur, -1):
            return {"status": cur}  # never downgrade
        confirmed_at = _now() if status == "confirmed" else None
        cx.execute("UPDATE supplement_reviews SET status=?, confirmed_at=COALESCE(?, confirmed_at), updated_at=? WHERE id=?",
                   (status, confirmed_at, _now(), review_id))
        cx.commit()
        return {"status": status}
    return {"status": cur}


def get(cx, review_id):
    row = cx.execute(
        "SELECT id, email, product_name, product_brand, source, status, review_text, "
        "requested_at, drafted_at, confirmed_at FROM supplement_reviews WHERE id=?",
        (review_id,)).fetchone()
    return _row(row) if row else None


def list_for_email(cx, email):
    rows = cx.execute(
        "SELECT id, email, product_name, product_brand, source, status, review_text, "
        "requested_at, drafted_at, confirmed_at FROM supplement_reviews WHERE email=? ORDER BY id",
        (_norm(email),)).fetchall()
    return [_row(r) for r in rows]


def pending_queue(cx, limit=100):
    """Rows needing operator attention (requested or ai_draft), oldest first."""
    rows = cx.execute(
        "SELECT id, email, product_name, product_brand, source, status, review_text, "
        "requested_at, drafted_at, confirmed_at FROM supplement_reviews "
        "WHERE status IN ('requested','ai_draft') ORDER BY id LIMIT ?", (int(limit),)).fetchall()
    return [_row(r) for r in rows]


def _row(r):
    return {"id": r[0], "email": r[1], "product_name": r[2], "product_brand": r[3],
            "source": r[4], "status": r[5], "review_text": r[6],
            "requested_at": r[7], "drafted_at": r[8], "confirmed_at": r[9]}
