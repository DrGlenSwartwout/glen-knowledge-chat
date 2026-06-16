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
    return token, cur.lastrowid


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
