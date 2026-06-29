"""Tokenized per-client portal — data layer.

One row per client = their personal "healing adventure" home page. The token is
the only auth (no login), mirroring the /invoice/<token> pattern. Durable: no
short TTL — this is the client's home, and the token is what they bookmark.

Content (greeting, video, causal-chain layers, reorder items) is stored as JSON
in `content_json`; the route layer enriches reorder slugs with catalog data.
"""

import hashlib
import json
import secrets
from datetime import datetime, timezone


def _hash(token: str) -> str:
    return hashlib.sha256((token or "").strip().encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_client_portal_table(cx) -> None:
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS client_portals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash   TEXT UNIQUE,
            email        TEXT,
            name         TEXT,
            content_json TEXT,
            created_at   TEXT,
            updated_at   TEXT
        )
        """
    )
    cx.execute("CREATE INDEX IF NOT EXISTS ix_client_portals_email ON client_portals(email)")
    cx.commit()


def upsert_portal(cx, email: str, name: str, content: dict):
    """Create or update a client's portal, keyed by email.

    On first create a token is minted and returned. On update the existing row
    (and therefore its token_hash) is preserved so previously-shared links never
    break — and since only the hash is stored, update returns ``None`` for the
    token slot (the caller already holds the link they shared at create time).

    Returns ``(raw_token_or_None, portal_id)``.
    """
    email = (email or "").strip().lower()
    now = _now_iso()
    row = cx.execute("SELECT id FROM client_portals WHERE email=?", (email,)).fetchone()
    payload = json.dumps(content or {})
    if row:
        pid = row[0]
        cx.execute(
            "UPDATE client_portals SET name=?, content_json=?, updated_at=? WHERE id=?",
            (name, payload, now, pid),
        )
        cx.commit()
        return None, pid
    token = secrets.token_urlsafe(32)
    cur = cx.execute(
        "INSERT INTO client_portals (token_hash, email, name, content_json, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        (_hash(token), email, name, payload, now, now),
    )
    cx.commit()
    # Persist the raw token so a later "Publish & email" REUSES this exact link rather
    # than rotating to a new one (which would dead-link the URL already shared/emailed).
    # ensure_token() reads it back from notify_state; only the hash lives in this table.
    try:
        from dashboard import notify_state as _ns
        _ns.init_table(cx)
        _ns.set_token(cx, email, token)
    except Exception:
        pass
    return token, cur.lastrowid


def reissue_token(cx, email):
    """Mint a FRESH token for an existing portal (rotates token_hash; the old link
    stops working). Content is unchanged. Returns the new raw token, or None if
    there is no portal for that email. Use to re-share a link when the original
    token (stored only as a one-way hash) can't be recovered."""
    email = (email or "").strip().lower()
    row = cx.execute("SELECT id FROM client_portals WHERE email=?", (email,)).fetchone()
    if not row:
        return None
    token = secrets.token_urlsafe(32)
    cx.execute("UPDATE client_portals SET token_hash=?, updated_at=? WHERE id=?",
               (_hash(token), _now_iso(), row[0]))
    cx.commit()
    # The freshly-rotated link is now the reusable one — keep notify_state in sync so
    # a subsequent "Publish & email" reuses it instead of rotating again.
    try:
        from dashboard import notify_state as _ns
        _ns.init_table(cx)
        _ns.set_token(cx, email, token)
    except Exception:
        pass
    return token


def ensure_token(cx, email, name=""):
    """Stable raw token for notification links. Creates a pending portal if the
    client has none. The raw token is held in portal_notify_state so the link is
    re-sendable without rotating it each scan. Returns the raw token."""
    from dashboard import notify_state as _ns
    email = (email or "").strip().lower()
    st = _ns.get_state(cx, email)
    if st.get("portal_token"):
        return st["portal_token"]
    if not cx.execute("SELECT 1 FROM client_portals WHERE email=?", (email,)).fetchone():
        upsert_portal(cx, email, name, {"biofield_status": "pending"})
    token = secrets.token_urlsafe(32)
    cx.execute("UPDATE client_portals SET token_hash=?, updated_at=? WHERE email=?",
               (_hash(token), _now_iso(), email))
    _ns.set_token(cx, email, token)
    cx.commit()
    return token


def get_portal_by_token(cx, token: str):
    th = _hash(token)
    row = cx.execute(
        "SELECT email, name, content_json FROM client_portals WHERE token_hash=?", (th,)
    ).fetchone()
    if not row:
        return None
    try:
        content = json.loads(row[2] or "{}")
    except Exception:
        content = {}
    return {"email": row[0], "name": row[1], "content": content}


def get_portal_content_by_email(cx, email):
    """A client's portal content keyed by email (for the unified portal view,
    which resolves identity first and then reads the biofield block). Returns
    ``{"name", "content"}`` or ``None`` when the client has no portal yet."""
    email = (email or "").strip().lower()
    row = cx.execute(
        "SELECT name, content_json FROM client_portals WHERE email=?", (email,)
    ).fetchone()
    if not row:
        return None
    try:
        content = json.loads(row[1] or "{}")
    except Exception:
        content = {}
    return {"name": row[0], "content": content}


def get_biofield_status(cx, email):
    """The biofield review status; legacy/hand-built portals (no field) = 'confirmed'."""
    rec = get_portal_content_by_email(cx, email)
    if not rec:
        return None
    return (rec.get("content") or {}).get("biofield_status") or "confirmed"


def set_biofield_status(cx, email, status):
    """Set content.biofield_status in place. Returns False if no portal for that email."""
    email = (email or "").strip().lower()
    row = cx.execute("SELECT content_json FROM client_portals WHERE email=?", (email,)).fetchone()
    if not row:
        return False
    try:
        content = json.loads(row[0] or "{}")
    except Exception:
        content = {}
    content["biofield_status"] = status
    cx.execute("UPDATE client_portals SET content_json=?, updated_at=? WHERE email=?",
               (json.dumps(content), _now_iso(), email))
    cx.commit()
    return True
