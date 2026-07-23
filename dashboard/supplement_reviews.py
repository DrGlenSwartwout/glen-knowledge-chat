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

_RANK = {"listed": -1, "requested": 0, "ai_draft": 1, "confirmed": 2}


def enabled():
    """True when SUPPLEMENT_REVIEW_ENABLED is set truthy. Default off (dark)."""
    return (os.environ.get("SUPPLEMENT_REVIEW_ENABLED", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm(e):
    return (e or "").strip().lower()


def product_key(name, brand=""):
    """Canonical stable dedupe key for a product: case- and whitespace-insensitive
    name|brand. This is the SINGLE SOURCE OF TRUTH for this normalization —
    dashboard.remedies_block._product_key and dashboard._remedies_product_key
    in app.py both delegate here rather than reimplementing it, so the three
    entry points can never drift apart."""
    raw = "%s|%s" % ((name or "").strip().lower(), (brand or "").strip().lower())
    return re.sub(r"\s+", " ", raw)


def _key(name, brand):
    """Stable dedupe key for a product: case- and whitespace-insensitive name|brand.
    Delegates to product_key() (kept for backward compatibility with existing
    call sites in this module)."""
    return product_key(name, brand)


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
    _cols = {r[1] for r in cx.execute("PRAGMA table_info(supplement_reviews)")}
    if "reason" not in _cols:
        cx.execute("ALTER TABLE supplement_reviews ADD COLUMN reason TEXT")
    if "importance" not in _cols:
        cx.execute("ALTER TABLE supplement_reviews ADD COLUMN importance INTEGER")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_suprev_status ON supplement_reviews(status)")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_suprev_email ON supplement_reviews(email)")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS supplement_review_access (
            email      TEXT PRIMARY KEY,
            enabled    INTEGER NOT NULL DEFAULT 1,
            set_by     TEXT,
            updated_at TEXT
        )
    """)
    cx.commit()


def access_enabled(cx, email):
    """Per-client access to the free product review. Default ON: a client is
    enabled unless there is an explicit opt-out row (enabled=0). None-raising."""
    e = _norm(email)
    if not e:
        return True
    try:
        row = cx.execute("SELECT enabled FROM supplement_review_access WHERE email=?", (e,)).fetchone()
    except Exception:
        return True
    return True if row is None else bool(row[0])


def set_access(cx, email, enabled, by=None):
    """Grant (True) or revoke (False) a client's access to the free review. Upsert."""
    e = _norm(email)
    if not e:
        return {"email": None, "enabled": None}
    val = 1 if enabled else 0
    cx.execute(
        "INSERT INTO supplement_review_access (email, enabled, set_by, updated_at) VALUES (?,?,?,?) "
        "ON CONFLICT(email) DO UPDATE SET enabled=excluded.enabled, set_by=excluded.set_by, updated_at=excluded.updated_at",
        (e, val, (by or ""), _now()))
    cx.commit()
    return {"email": e, "enabled": bool(val)}


def create_request(cx, email, product_name, product_brand="", source="portal"):
    e = _norm(email)
    name = (product_name or "").strip()
    if not e or not name:
        return {"created": False, "id": None, "status": None}
    key = _key(name, product_brand)
    row = cx.execute("SELECT id, status FROM supplement_reviews WHERE email=? AND product_key=?",
                     (e, key)).fetchone()
    if row:
        if row[1] == "listed":
            now = _now()
            cx.execute(
                "UPDATE supplement_reviews SET status='requested', requested_at=?, updated_at=? WHERE id=?",
                (now, now, row[0]))
            cx.commit()
            return {"created": False, "id": row[0], "status": "requested"}
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


def _clamp_importance(importance):
    if importance is None:
        return None
    try:
        v = int(importance)
    except (TypeError, ValueError):
        return None
    return max(1, min(10, v))


def add_listed(cx, email, product_name, product_brand="", reason="", importance=None, source="portal"):
    """Client adds a supplement to their externally-maintained stack, no review
    requested yet. Idempotent on (email, product_key): if a row already exists
    (any status) it is returned untouched."""
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
            "(email, product_name, product_brand, product_key, source, status, reason, importance, updated_at) "
            "VALUES (?,?,?,?,?, 'listed', ?, ?, ?)",
            (e, name, (product_brand or "").strip(), key, (source or "").strip(),
             (reason or "").strip() or None, _clamp_importance(importance), now))
        cx.commit()
        return {"created": True, "id": cur.lastrowid, "status": "listed"}
    except db.IntegrityError:
        # lost a UNIQUE race → return the existing row rather than 500ing.
        row = cx.execute("SELECT id, status FROM supplement_reviews WHERE email=? AND product_key=?",
                         (e, key)).fetchone()
        return {"created": False, "id": row[0] if row else None,
                "status": row[1] if row else None}


def set_meta(cx, email, product_key, reason=None, importance=None):
    """Update the client-set reason/importance on an existing row. Leaves
    status/review untouched. importance clamped to 1..10 or None."""
    e = _norm(email)
    row = cx.execute("SELECT id FROM supplement_reviews WHERE email=? AND product_key=?",
                     (e, product_key)).fetchone()
    if not row:
        return {"updated": False}
    cx.execute(
        "UPDATE supplement_reviews SET reason=?, importance=?, updated_at=? WHERE id=?",
        ((reason or "").strip() or None, _clamp_importance(importance), _now(), row[0]))
    cx.commit()
    return {"updated": True, "id": row[0]}


def remove(cx, email, product_key):
    """Delete a row only if its status is 'listed' or 'requested' — never an
    ai_draft/confirmed review, which represents Glen's work."""
    e = _norm(email)
    row = cx.execute("SELECT id, status FROM supplement_reviews WHERE email=? AND product_key=?",
                     (e, product_key)).fetchone()
    if not row or row[1] not in ("listed", "requested"):
        return {"removed": False}
    cx.execute("DELETE FROM supplement_reviews WHERE id=?", (row[0],))
    cx.commit()
    return {"removed": True}


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
        "requested_at, drafted_at, confirmed_at, reason, importance FROM supplement_reviews WHERE id=?",
        (review_id,)).fetchone()
    return _row(row) if row else None


def list_for_email(cx, email):
    rows = cx.execute(
        "SELECT id, email, product_name, product_brand, source, status, review_text, "
        "requested_at, drafted_at, confirmed_at, reason, importance FROM supplement_reviews WHERE email=? ORDER BY id",
        (_norm(email),)).fetchall()
    return [_row(r) for r in rows]


def pending_queue(cx, limit=100):
    """Rows needing operator attention (requested or ai_draft), oldest first."""
    rows = cx.execute(
        "SELECT id, email, product_name, product_brand, source, status, review_text, "
        "requested_at, drafted_at, confirmed_at, reason, importance FROM supplement_reviews "
        "WHERE status IN ('requested','ai_draft') ORDER BY id LIMIT ?", (int(limit),)).fetchall()
    return [_row(r) for r in rows]


def _row(r):
    return {"id": r[0], "email": r[1], "product_name": r[2], "product_brand": r[3],
            "source": r[4], "status": r[5], "review_text": r[6],
            "requested_at": r[7], "drafted_at": r[8], "confirmed_at": r[9],
            "reason": r[10], "importance": r[11]}
