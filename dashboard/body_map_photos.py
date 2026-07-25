# dashboard/body_map_photos.py
"""Per-slot Body Map photo + saved alignment store.

A slot is (email, system, side); each holds one photo and an optional
{mx,my,tx,ty} similarity transform in the map's fixed 600x600 viewBox space
(resolution-independent). Persistence only -- no HTTP, no rendering.

`image_blob` is BYTEA, not BLOB: runtime pgcompat does not translate BLOB, so a
BLOB column fails outright on Postgres. BYTEA round-trips bytes on SQLite. See
docs/superpowers/specs/2026-07-25-bodymap-multiphoto-alignment-foundation-design.md

This is a SEPARATE table from client_photos (the identity portrait). client_photos
is never touched here; the face-slot fallback to it lives in the HTTP layer.
"""
import json
import math
from datetime import datetime, timezone

_TKEYS = ("mx", "my", "tx", "ty")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def _side(side):
    return (side or "").strip().lower()


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS body_map_photos (
        email TEXT, system TEXT, side TEXT,
        image_blob BYTEA, content_type TEXT, transform_json TEXT,
        source TEXT, updated_at TEXT,
        PRIMARY KEY (email, system, side))""")
    cx.commit()


def _valid_transform(t):
    if not isinstance(t, dict):
        return None
    out = {}
    for k in _TKEYS:
        v = t.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return None
        out[k] = float(v)
    return out


def put(cx, email, system, side, blob, content_type, source):
    """Upsert the slot's photo. A new photo CLEARS any saved transform (it needs
    re-aligning). Returns True on write, False for empty email/system/blob."""
    e, sys_, sd = _norm(email), (system or "").strip(), _side(side)
    if not e or not sys_ or not blob:
        return False
    init_table(cx)
    cx.execute(
        "INSERT INTO body_map_photos"
        "(email, system, side, image_blob, content_type, transform_json, source, updated_at) "
        "VALUES(?,?,?,?,?,NULL,?,?) "
        "ON CONFLICT(email, system, side) DO UPDATE SET "
        "image_blob=excluded.image_blob, content_type=excluded.content_type, "
        "transform_json=NULL, source=excluded.source, updated_at=excluded.updated_at",
        (e, sys_, sd, blob, content_type or "image/jpeg", source or "", _now()))
    cx.commit()
    return True


def get(cx, email, system, side):
    e, sys_, sd = _norm(email), (system or "").strip(), _side(side)
    if not e:
        return None
    init_table(cx)
    r = cx.execute(
        "SELECT image_blob, content_type, transform_json, source FROM body_map_photos "
        "WHERE email=? AND system=? AND side=?", (e, sys_, sd)).fetchone()
    if not r or r[0] is None:
        return None
    try:
        transform = json.loads(r[2]) if r[2] else None
    except (TypeError, ValueError):
        transform = None
    return {"blob": r[0], "content_type": r[1] or "image/jpeg",
            "transform": transform, "source": r[3] or ""}


def set_transform(cx, email, system, side, transform):
    """Save (or clear, when transform is None) the slot's {mx,my,tx,ty}. Rejects
    anything that is not four finite numbers. Returns True on a write/clear,
    False on a malformed transform."""
    e, sys_, sd = _norm(email), (system or "").strip(), _side(side)
    if not e:
        return False
    init_table(cx)
    if transform is None:
        val = None
    else:
        clean = _valid_transform(transform)
        if clean is None:
            return False
        val = json.dumps(clean)
    cx.execute("UPDATE body_map_photos SET transform_json=?, updated_at=? "
               "WHERE email=? AND system=? AND side=?", (val, _now(), e, sys_, sd))
    cx.commit()
    return True


def get_transform(cx, email, system, side):
    rec = get(cx, email, system, side)
    return rec["transform"] if rec else None


def list_for_email(cx, email):
    e = _norm(email)
    if not e:
        return []
    init_table(cx)
    rows = cx.execute(
        "SELECT system, side, transform_json, updated_at FROM body_map_photos "
        "WHERE email=? ORDER BY system, side", (e,)).fetchall()
    return [{"system": r[0], "side": r[1], "has_transform": bool(r[2]),
             "updated_at": r[3]} for r in rows]
