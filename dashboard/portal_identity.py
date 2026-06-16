"""The portal identity seam.

`resolve_identity` is the single choke point that turns an incoming portal
request into an `Identity` resolved against the unified `people` row. Today only
the token branch is active (the emailed `/portal/<token>` link is the auth, no
login — mirroring `/invoice/<token>`). The session branch is scaffolded but
inactive: it is the documented drop-in point for real client login (next slice),
and it returns the *same* `Identity` shape so the page and APIs never change.

Kept self-contained (takes a `cx`, never imports `app`) so it unit-tests fast
and in isolation, mirroring `dashboard/client_portal.py`.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets

from dashboard import client_portal as _cp
from dashboard.practitioner_portal import _ensure_auth_tokens

CLIENT_SESSION_TTL_DAYS = 30
_SESSION_PURPOSE = "client_session"


def _hash(token: str) -> str:
    return hashlib.sha256((token or "").strip().encode("utf-8")).hexdigest()


@dataclass
class Identity:
    person_id: int
    email: str
    roles: list = field(default_factory=list)
    auth_method: str = "token"  # "token" (active) | "session" (scaffolded)


def _ensure_people_table(cx) -> None:
    """Create a minimal `people` table if absent. In production the full table
    (app._init_people_table) already exists, so `IF NOT EXISTS` is a no-op; this
    only fires for isolated tests / standalone use. Columns are a compatible
    subset of the real schema."""
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS people (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT UNIQUE NOT NULL,
            name       TEXT DEFAULT '',
            roles      TEXT DEFAULT '[]',
            tags       TEXT DEFAULT '[]',
            created_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT ''
        )
        """
    )
    cx.commit()


def _get_or_create_person(cx, email: str, name: str = ""):
    """Resolve an email to (person_id, roles), lazily creating a minimal person
    row when the portal holder isn't in the hub yet. A portal-link holder is a
    `client` by default, so the client-facing blocks render. Richer enrichment
    (tags, address, history) happens elsewhere via app.upsert_person — this only
    guarantees the portal always has a person to hang roles on."""
    row = cx.execute(
        "SELECT id, roles FROM people WHERE email=?", (email,)
    ).fetchone()
    if row:
        try:
            roles = list(json.loads(row[1] or "[]"))
        except Exception:
            roles = []
        return row[0], (roles or ["client"])
    cur = cx.execute(
        "INSERT INTO people (email, name, roles, created_at, updated_at) VALUES (?,?,?,?,?)",
        (email, name or "", json.dumps(["client"]), "", ""),
    )
    cx.commit()
    return cur.lastrowid, ["client"]


def _roles_by_person_id(cx, person_id):
    row = cx.execute("SELECT email, roles FROM people WHERE id=?", (person_id,)).fetchone()
    if not row:
        return None, []
    try:
        roles = list(json.loads(row[1] or "[]"))
    except Exception:
        roles = []
    return row[0], roles


def identity_from_token(cx, token) -> "Identity | None":
    """Active branch: validate a portal token and resolve it to an Identity."""
    portal = _cp.get_portal_by_token(cx, token)
    if not portal:
        return None
    email = (portal.get("email") or "").strip().lower()
    if not email:
        return None
    person_id, roles = _get_or_create_person(cx, email, portal.get("name") or "")
    return Identity(person_id=person_id, email=email, roles=roles, auth_method="token")


# ── Scaffolded session branch — the drop-in point for real client login ───────
# Built and tested, but dark: the funnel never mints a client session and
# resolve_identity ignores session cookies unless client_login_enabled is True
# (gated upstream by CLIENT_LOGIN_ENABLED). Mirrors the practitioner session
# pattern (30-day token in auth_tokens), keyed to a person_id.

def create_client_session(cx, person_id, email="", *, ttl_days=CLIENT_SESSION_TTL_DAYS) -> str:
    tok = secrets.token_urlsafe(32)
    _ensure_auth_tokens(cx)
    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=ttl_days)
    cx.execute(
        "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
        "VALUES (?,?,?,?,?,?)",
        (_hash(tok), email, _SESSION_PURPOSE,
         json.dumps({"person_id": person_id, "email": email}),
         now.isoformat(), exp.isoformat()),
    )
    cx.commit()
    return tok


_MAGIC_PURPOSE = "client_magic_link"
CLIENT_MAGIC_TTL_MIN = 15


def create_client_magic_link(cx, person_id, email="", *, ttl_min=CLIENT_MAGIC_TTL_MIN) -> str:
    """Mint a one-time login link token for a client (emailed by /portal/login).
    Dark: only minted/consumed when CLIENT_LOGIN_ENABLED is on."""
    tok = secrets.token_urlsafe(32)
    _ensure_auth_tokens(cx)
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=ttl_min)
    cx.execute(
        "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
        "VALUES (?,?,?,?,?,?)",
        (_hash(tok), email, _MAGIC_PURPOSE,
         json.dumps({"person_id": person_id, "email": email}),
         now.isoformat(), exp.isoformat()),
    )
    cx.commit()
    return tok


def consume_client_magic_link(cx, token) -> "int | None":
    """Validate a one-time magic-link, mark it consumed, return the person_id."""
    if not token:
        return None
    _ensure_auth_tokens(cx)
    row = cx.execute(
        "SELECT extra, expires_at, consumed_at FROM auth_tokens "
        "WHERE token_hash=? AND purpose=?",
        (_hash(token), _MAGIC_PURPOSE),
    ).fetchone()
    if not row:
        return None
    extra, expires_at, consumed_at = row
    if consumed_at:
        return None
    try:
        if datetime.fromisoformat(expires_at.rstrip("Z")) < datetime.now(timezone.utc):
            return None
    except Exception:
        return None
    cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
               (datetime.now(timezone.utc).isoformat(), _hash(token)))
    cx.commit()
    try:
        return (json.loads(extra) or {}).get("person_id")
    except Exception:
        return None


def identity_from_session(cx, session_token) -> "Identity | None":
    if not session_token:
        return None
    _ensure_auth_tokens(cx)
    row = cx.execute(
        "SELECT extra, expires_at, consumed_at FROM auth_tokens "
        "WHERE token_hash=? AND purpose=?",
        (_hash(session_token), _SESSION_PURPOSE),
    ).fetchone()
    if not row:
        return None
    extra, expires_at, consumed_at = row
    if consumed_at:
        return None
    try:
        if datetime.fromisoformat(expires_at.rstrip("Z")) < datetime.now(timezone.utc):
            return None
    except Exception:
        return None
    try:
        person_id = (json.loads(extra) or {}).get("person_id")
    except Exception:
        return None
    if not person_id:
        return None
    email, roles = _roles_by_person_id(cx, person_id)
    if email is None:
        return None
    return Identity(person_id=person_id, email=email, roles=roles, auth_method="session")


def resolve_identity(cx, *, token=None, session_token=None, client_login_enabled=False):
    """Single choke point. Prefer a logged-in session (only when login is live),
    else fall back to the emailed portal token. The page/APIs call this and never
    look at tokens or cookies directly."""
    if client_login_enabled and session_token:
        ident = identity_from_session(cx, session_token)
        if ident is not None:
            return ident
    if token:
        return identity_from_token(cx, token)
    return None
