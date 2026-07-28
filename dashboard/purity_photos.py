"""Client-uploaded supplement LABEL photos, kept so Glen can verify a
client-submitted purity screen against the actual photo at confirm time (catch a
photo/product mismatch). One photo per product_key (latest wins).

image_blob is BYTEA, not BLOB: runtime pgcompat does not translate BLOB, so a
BLOB column fails on Postgres; BYTEA round-trips bytes on SQLite too (see
dashboard/body_map_photos.py). Writes use SELECT-then-INSERT/UPDATE + a Python
timestamp -- no ON CONFLICT / datetime('now') -- matching the other purity
modules for cross-backend safety.
"""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS purity_photos (
        product_key TEXT PRIMARY KEY,
        email TEXT, image_blob BYTEA, content_type TEXT, updated_at TEXT)""")
    cx.commit()


def save(cx, product_key, email, blob, content_type):
    """Upsert the latest label photo for a product. False on empty key/blob."""
    key = (product_key or "").strip()
    if not key or not blob:
        return False
    init_table(cx)
    now = _now()
    exists = cx.execute("SELECT 1 FROM purity_photos WHERE product_key=?", (key,)).fetchone()
    if exists:
        cx.execute("UPDATE purity_photos SET email=?, image_blob=?, content_type=?, "
                   "updated_at=? WHERE product_key=?",
                   ((email or "").strip().lower(), blob, content_type or "image/jpeg", now, key))
    else:
        cx.execute("INSERT INTO purity_photos "
                   "(product_key, email, image_blob, content_type, updated_at) VALUES (?,?,?,?,?)",
                   (key, (email or "").strip().lower(), blob, content_type or "image/jpeg", now))
    cx.commit()
    return True


def get(cx, product_key):
    key = (product_key or "").strip()
    if not key:
        return None
    r = cx.execute("SELECT product_key, email, image_blob, content_type, updated_at "
                   "FROM purity_photos WHERE product_key=?", (key,)).fetchone()
    return dict(r) if r else None


def keys_with_photos(cx):
    return {row[0] for row in cx.execute("SELECT product_key FROM purity_photos").fetchall()}
