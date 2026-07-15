"""Client photo store — one photo per client, keyed by lowercased email.

Central store for Slice 1 of the client-photos feature: the Biofield Intake page
and console/reveal surfaces read a client's photo by email; FMP-export upload,
portal self-upload, and GHL pull all write here. Persistence only — no HTTP, no
rendering. See docs/superpowers/specs/2026-07-14-client-photos-design.md.
"""
import datetime


def _now():
    return datetime.datetime.utcnow().isoformat() + "Z"


def _norm(email):
    return (email or "").strip().lower()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS client_photos("
        "email TEXT PRIMARY KEY, image_blob BLOB, content_type TEXT, "
        "source TEXT, updated_at TEXT)")


def put(cx, email, blob, content_type, source="upload"):
    """Upsert a client's photo. Returns the normalized email, or None if no email."""
    e = _norm(email)
    if not e or not blob:
        return None
    init_table(cx)
    cx.execute(
        "INSERT INTO client_photos(email, image_blob, content_type, source, updated_at) "
        "VALUES(?,?,?,?,?) ON CONFLICT(email) DO UPDATE SET "
        "image_blob=excluded.image_blob, content_type=excluded.content_type, "
        "source=excluded.source, updated_at=excluded.updated_at",
        (e, blob, (content_type or "image/jpeg"), source, _now()))
    cx.commit()
    return e


def get(cx, email):
    """Return {'blob': bytes, 'content_type': str} for the email, or None."""
    e = _norm(email)
    if not e:
        return None
    init_table(cx)
    row = cx.execute(
        "SELECT image_blob, content_type FROM client_photos WHERE email=?", (e,)
    ).fetchone()
    if not row or row[0] is None:
        return None
    return {"blob": row[0], "content_type": row[1] or "image/jpeg"}


def has(cx, email):
    return get(cx, email) is not None
