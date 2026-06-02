"""Practitioner wholesale portal — logic layer (Phase 3c/3d/3f).

Holds the DB-heavy, testable pieces so the app.py routes stay thin: the cart
(SQLite), magic-link + session tokens (SQLite auth_tokens), two-door registration
(Supabase practitioners row), and portal-data assembly. Imports no app module, so
it loads without the app's heavy deps.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from dashboard import wholesale_pricing as pricing

_LOG_DB = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent))) / "chat_log.db"

MAGIC_TTL_MIN = 15
SESSION_TTL_DAYS = 30


def _db_path() -> str:
    return str(_LOG_DB)


def _hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _utcnow(now=None) -> datetime:
    return now or datetime.utcnow()


# ── schema ────────────────────────────────────────────────────────────────────

def _ensure_auth_tokens(cx) -> None:
    cx.execute(
        "CREATE TABLE IF NOT EXISTS auth_tokens ("
        "token_hash TEXT PRIMARY KEY, email TEXT, purpose TEXT NOT NULL, extra TEXT, "
        "created_at TEXT NOT NULL, expires_at TEXT NOT NULL, consumed_at TEXT)"
    )


def init_cart_table(db_path: Optional[str] = None) -> None:
    with sqlite3.connect(db_path or _db_path()) as cx:
        cx.execute(
            "CREATE TABLE IF NOT EXISTS wholesale_cart ("
            "practitioner_id TEXT NOT NULL, slug TEXT NOT NULL, qty INTEGER NOT NULL "
            "CHECK (qty > 0), updated_at TEXT NOT NULL, "
            "PRIMARY KEY (practitioner_id, slug))"
        )
        cx.commit()


# ── cart ──────────────────────────────────────────────────────────────────────

def cart_set(practitioner_id, slug, qty, *, db_path=None, now=None) -> None:
    """Set a line's qty; qty<=0 removes it."""
    p = db_path or _db_path()
    init_cart_table(p)
    ts = (now or datetime.now(timezone.utc)).isoformat()
    with sqlite3.connect(p) as cx:
        if int(qty) <= 0:
            cx.execute("DELETE FROM wholesale_cart WHERE practitioner_id=? AND slug=?",
                       (str(practitioner_id), slug))
        else:
            cx.execute(
                "INSERT INTO wholesale_cart (practitioner_id, slug, qty, updated_at) "
                "VALUES (?,?,?,?) ON CONFLICT(practitioner_id, slug) "
                "DO UPDATE SET qty=excluded.qty, updated_at=excluded.updated_at",
                (str(practitioner_id), slug, int(qty), ts),
            )
        cx.commit()


def cart_items(practitioner_id, *, db_path=None) -> List[dict]:
    p = db_path or _db_path()
    init_cart_table(p)
    with sqlite3.connect(p) as cx:
        rows = cx.execute(
            "SELECT slug, qty FROM wholesale_cart WHERE practitioner_id=? ORDER BY slug",
            (str(practitioner_id),),
        ).fetchall()
    return [{"slug": r[0], "qty": r[1]} for r in rows]


def cart_clear(practitioner_id, *, db_path=None) -> None:
    p = db_path or _db_path()
    init_cart_table(p)
    with sqlite3.connect(p) as cx:
        cx.execute("DELETE FROM wholesale_cart WHERE practitioner_id=?", (str(practitioner_id),))
        cx.commit()


# ── order history (local record, written at checkout) ─────────────────────────

def _ensure_orders_table(cx) -> None:
    cx.execute(
        "CREATE TABLE IF NOT EXISTS wholesale_orders ("
        "invoice_id TEXT PRIMARY KEY, practitioner_id TEXT NOT NULL, doc_number TEXT, "
        "total_cents INTEGER, credit_cents INTEGER, created_at TEXT NOT NULL)"
    )
    cx.execute("CREATE INDEX IF NOT EXISTS wholesale_orders_prac "
               "ON wholesale_orders (practitioner_id, created_at DESC)")


def record_order(practitioner_id, *, invoice_id, doc_number=None, total_cents=0,
                 credit_cents=0, now=None, db_path=None) -> None:
    """Persist a placed order so the portal can show history without a live QBO
    call. Idempotent on invoice_id (a retry won't duplicate)."""
    if not invoice_id:
        return
    p = db_path or _db_path()
    ts = (now or datetime.now(timezone.utc)).isoformat()
    with sqlite3.connect(p) as cx:
        _ensure_orders_table(cx)
        cx.execute(
            "INSERT INTO wholesale_orders "
            "(invoice_id, practitioner_id, doc_number, total_cents, credit_cents, created_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(invoice_id) DO NOTHING",
            (str(invoice_id), str(practitioner_id), doc_number,
             int(total_cents or 0), int(credit_cents or 0), ts),
        )
        cx.commit()


def order_history(practitioner_id, *, limit=20, db_path=None) -> List[dict]:
    p = db_path or _db_path()
    with sqlite3.connect(p) as cx:
        _ensure_orders_table(cx)
        rows = cx.execute(
            "SELECT invoice_id, doc_number, total_cents, credit_cents, created_at "
            "FROM wholesale_orders WHERE practitioner_id=? ORDER BY created_at DESC LIMIT ?",
            (str(practitioner_id), int(limit)),
        ).fetchall()
    return [{"invoice_id": r[0], "doc_number": r[1], "total_cents": r[2],
             "credit_cents": r[3], "created_at": r[4]} for r in rows]


# ── AI assistant cross-sell (resolve adjacent formulations to cart slugs) ──────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PAIRINGS_CACHE = None


def _load_pairings() -> dict:
    global _PAIRINGS_CACHE
    if _PAIRINGS_CACHE is None:
        try:
            _PAIRINGS_CACHE = json.loads((_DATA_DIR / "upsell-pairings.json").read_text())
        except Exception:
            _PAIRINGS_CACHE = {}
    return _PAIRINGS_CACHE


def name_to_slug(name, catalog) -> Optional[str]:
    """Resolve a product NAME to a products.json slug (exact or fuzzy substring).
    Mirrors app.py:_resolve_buy_slug so the assistant can add to the cart."""
    if not name:
        return None
    nl = name.strip().lower()
    for slug, p in (catalog or {}).items():
        pn = (p.get("name") or "").lower()
        if pn and (nl == pn or (len(nl) > 4 and (nl in pn or pn in nl))):
            return slug
    return None


def assist_cross_sell(slug, *, catalog=None, pairings=None) -> List[dict]:
    """Adjacent in-catalog formulations for a matched slug, from upsell-pairings.
    Returns [{name, slug}] for resolvable, in-catalog complements (others dropped)."""
    cat = catalog if catalog is not None else pricing._load_catalog()
    pr = pairings if pairings is not None else _load_pairings()
    names = (pr.get("pairings") or {}).get(slug) or []
    out, seen = [], set()
    for nm in names:
        s = name_to_slug(nm, cat)
        if s and s != slug and s not in seen:
            seen.add(s)
            out.append({"name": nm, "slug": s})
    return out


# ── tokens (SQLite auth_tokens, shared with the app's magic-link table) ────────

def _insert_token(tok, purpose, extra, ttl_seconds, now=None, db_path=None) -> None:
    n = _utcnow(now)
    exp = n + timedelta(seconds=ttl_seconds)
    with sqlite3.connect(db_path or _db_path()) as cx:
        _ensure_auth_tokens(cx)
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (_hash(tok), (extra.get("email") or ""), purpose, json.dumps(extra),
             n.isoformat(), exp.isoformat()),
        )
        cx.commit()


def create_magic_link_token(practitioner_id, email="", *, now=None, db_path=None) -> str:
    tok = secrets.token_urlsafe(32)
    _insert_token(tok, "practitioner_magic_link",
                  {"practitioner_id": str(practitioner_id), "email": email},
                  MAGIC_TTL_MIN * 60, now, db_path)
    return tok


def create_session_token(practitioner_id, *, now=None, db_path=None) -> str:
    tok = secrets.token_urlsafe(32)
    _insert_token(tok, "practitioner_session", {"practitioner_id": str(practitioner_id)},
                  SESSION_TTL_DAYS * 86400, now, db_path)
    return tok


def _valid_token_row(token, purpose, *, now=None, db_path=None):
    if not token:
        return None
    with sqlite3.connect(db_path or _db_path()) as cx:
        _ensure_auth_tokens(cx)
        row = cx.execute(
            "SELECT extra, expires_at, consumed_at FROM auth_tokens "
            "WHERE token_hash=? AND purpose=?",
            (_hash(token), purpose),
        ).fetchone()
    if not row:
        return None
    extra, expires_at, consumed_at = row
    if consumed_at:
        return None
    try:
        if datetime.fromisoformat(expires_at.rstrip("Z")) < _utcnow(now):
            return None
    except Exception:
        return None
    return extra


def consume_magic_link(token, *, now=None, db_path=None) -> Optional[str]:
    """Validate a one-time magic-link token, mark it consumed, return practitioner_id."""
    extra = _valid_token_row(token, "practitioner_magic_link", now=now, db_path=db_path)
    if extra is None:
        return None
    with sqlite3.connect(db_path or _db_path()) as cx:
        cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
                   (_utcnow(now).isoformat(), _hash(token)))
        cx.commit()
    try:
        return (json.loads(extra) or {}).get("practitioner_id")
    except Exception:
        return None


def practitioner_id_from_session(token, *, now=None, db_path=None) -> Optional[str]:
    """Return the practitioner_id for a valid (non-expired, non-consumed) session token."""
    extra = _valid_token_row(token, "practitioner_session", now=now, db_path=db_path)
    if extra is None:
        return None
    try:
        return (json.loads(extra) or {}).get("practitioner_id")
    except Exception:
        return None


# ── registration (two-door) ───────────────────────────────────────────────────

def validate_registration(payload: dict) -> Tuple[Optional[dict], Optional[str]]:
    """Pure validation. Returns (clean_dict, None) or (None, error_message)."""
    email = (payload.get("email") or "").strip().lower()
    name = (payload.get("name") or "").strip()
    role = (payload.get("portal_role") or "").strip().lower()
    if "@" not in email or "." not in email:
        return None, "A valid email is required."
    if not name:
        return None, "Your name is required."
    if role not in ("licensed", "coach"):
        return None, "Choose how you're joining: licensed practitioner or coach."
    if role == "licensed" and not (payload.get("license_number") or "").strip():
        return None, "A license number is required for licensed practitioners."
    if role == "coach" and not (payload.get("resale_license_number") or "").strip():
        return None, "A resale certificate number is required to join as a coach."
    return {
        "email": email, "name": name, "portal_role": role,
        "practice_name": (payload.get("practice_name") or "").strip() or None,
        "credentials": (payload.get("credentials") or "").strip() or None,
        "phone": (payload.get("phone") or "").strip() or None,
        "website": (payload.get("website") or "").strip() or None,
        "license_state": (payload.get("license_state") or "").strip() or None,
        "license_number": (payload.get("license_number") or "").strip() or None,
        "resale_license_number": (payload.get("resale_license_number") or "").strip() or None,
    }, None


def find_practitioner_id_by_email(email) -> Optional[str]:
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute("SELECT id FROM practitioners WHERE lower(email)=lower(%s) "
                    "AND portal_role IS NOT NULL LIMIT 1", (str(email).strip(),))
        row = cur.fetchone()
    return str(row["id"]) if row else None


def register_practitioner(clean: dict, *, now=None) -> Tuple[str, bool]:
    """Insert or link a practitioners row for a portal registrant. Licensed unlock
    immediately; coaches stay locked until the first module is committed. Returns
    (practitioner_id, wholesale_unlocked)."""
    from db_supabase import supabase_cursor
    unlocked_at = _utcnow(now) if clean["portal_role"] == "licensed" else None
    tier = "panel_in_cert" if clean["portal_role"] == "coach" else "org_member"
    with supabase_cursor() as cur:
        cur.execute("SELECT id FROM practitioners WHERE lower(email)=lower(%s) LIMIT 1",
                    (clean["email"],))
        row = cur.fetchone()
        if row:
            pid = row["id"]
            cur.execute(
                "UPDATE practitioners SET portal_role=%s, license_state=%s, license_number=%s, "
                "resale_license_number=%s, "
                "wholesale_unlocked_at=COALESCE(wholesale_unlocked_at, %s), "
                "name=COALESCE(NULLIF(name,''), %s), practice_name=COALESCE(practice_name, %s), "
                "credentials=COALESCE(credentials, %s), phone=COALESCE(phone, %s), "
                "website=COALESCE(website, %s), updated_at=now() WHERE id=%s",
                (clean["portal_role"], clean["license_state"], clean["license_number"],
                 clean["resale_license_number"], unlocked_at, clean["name"],
                 clean["practice_name"], clean["credentials"], clean["phone"],
                 clean["website"], pid),
            )
        else:
            cur.execute(
                "INSERT INTO practitioners (tier, name, email, practice_name, credentials, "
                "phone, website, portal_role, license_state, license_number, "
                "resale_license_number, wholesale_unlocked_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (tier, clean["name"], clean["email"], clean["practice_name"],
                 clean["credentials"], clean["phone"], clean["website"],
                 clean["portal_role"], clean["license_state"], clean["license_number"],
                 clean["resale_license_number"], unlocked_at),
            )
            pid = cur.fetchone()["id"]
    return str(pid), unlocked_at is not None


def unlock_wholesale(practitioner_id, *, now=None) -> None:
    """Flip a coach to unlocked once their first module is committed."""
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute(
            "UPDATE practitioners SET wholesale_unlocked_at=COALESCE(wholesale_unlocked_at, %s), "
            "updated_at=now() WHERE id=%s",
            (_utcnow(now), str(practitioner_id)),
        )


# ── portal data ───────────────────────────────────────────────────────────────

def portal_data(practitioner_id, *, db_path=None, include_orders=False) -> Optional[dict]:
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute(
            "SELECT id, name, practice_name, email, portal_role, modules_completed, "
            "wallet_balance_cents, wholesale_unlocked_at FROM practitioners WHERE id=%s",
            (str(practitioner_id),),
        )
        row = cur.fetchone()
    if not row:
        return None
    items = cart_items(practitioner_id, db_path=db_path)
    quote = None
    if items:
        try:
            quote = pricing.order_quote(
                items, {"modules_completed": row["modules_completed"]}, db_path=db_path)
        except Exception:
            quote = None
    data = {
        "practitioner_id": str(row["id"]),
        "name": row["name"],
        "practice_name": row["practice_name"],
        "email": row["email"],
        "portal_role": row["portal_role"],
        "modules_completed": row["modules_completed"],
        "wallet_balance_cents": row["wallet_balance_cents"],
        "wholesale_unlocked": row["wholesale_unlocked_at"] is not None,
        "cart": items,
        "quote": quote,
    }
    if include_orders:
        data["order_history"] = order_history(practitioner_id, db_path=db_path)
    return data
