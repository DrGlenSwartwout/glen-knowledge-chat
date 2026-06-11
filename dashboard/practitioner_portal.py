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


# ── drop-ship dispensary (Phase 5) ────────────────────────────────────────────

def _ensure_dispensary_table(cx) -> None:
    cx.execute(
        "CREATE TABLE IF NOT EXISTS dispensary_orders ("
        "invoice_id TEXT PRIMARY KEY, practitioner_id TEXT NOT NULL, customer_email TEXT, "
        "bottles INTEGER, credit_earned_cents INTEGER, created_at TEXT NOT NULL)"
    )
    cx.execute("CREATE INDEX IF NOT EXISTS dispensary_orders_prac "
               "ON dispensary_orders (practitioner_id, created_at DESC)")


def record_dispensary_order(practitioner_id, *, invoice_id, customer_email=None,
                            bottles=0, credit_earned_cents=0, now=None, db_path=None) -> None:
    """Persist a drop-ship sale attributed to a practitioner. Idempotent on invoice_id."""
    if not invoice_id:
        return
    p = db_path or _db_path()
    ts = (now or datetime.now(timezone.utc)).isoformat()
    with sqlite3.connect(p) as cx:
        _ensure_dispensary_table(cx)
        cx.execute(
            "INSERT INTO dispensary_orders "
            "(invoice_id, practitioner_id, customer_email, bottles, credit_earned_cents, created_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(invoice_id) DO NOTHING",
            (str(invoice_id), str(practitioner_id), customer_email,
             int(bottles or 0), int(credit_earned_cents or 0), ts),
        )
        cx.commit()


def dispensary_order_history(practitioner_id, *, limit=20, db_path=None) -> List[dict]:
    p = db_path or _db_path()
    with sqlite3.connect(p) as cx:
        _ensure_dispensary_table(cx)
        rows = cx.execute(
            "SELECT invoice_id, customer_email, bottles, credit_earned_cents, created_at "
            "FROM dispensary_orders WHERE practitioner_id=? ORDER BY created_at DESC LIMIT ?",
            (str(practitioner_id), int(limit)),
        ).fetchall()
    return [{"invoice_id": r[0], "customer_email": r[1], "bottles": r[2],
             "credit_earned_cents": r[3], "created_at": r[4]} for r in rows]


def practitioner_id_by_dispensary_code(code) -> Optional[str]:
    if not code:
        return None
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute("SELECT id FROM practitioners WHERE dispensary_code=%s LIMIT 1",
                    (str(code).strip(),))
        row = cur.fetchone()
    return str(row["id"]) if row else None


def get_or_create_dispensary_code(practitioner_id, *, _gen=None) -> str:
    """Return the practitioner's dispensary code, generating + persisting one on first use."""
    from db_supabase import supabase_cursor
    pid = str(practitioner_id)
    with supabase_cursor() as cur:
        cur.execute("SELECT dispensary_code FROM practitioners WHERE id=%s", (pid,))
        row = cur.fetchone()
        if row and row.get("dispensary_code"):
            return row["dispensary_code"]
        code = (_gen or (lambda: secrets.token_urlsafe(6)))()
        cur.execute("UPDATE practitioners SET dispensary_code=%s WHERE id=%s "
                    "AND dispensary_code IS NULL", (code, pid))
        cur.execute("SELECT dispensary_code FROM practitioners WHERE id=%s", (pid,))
        row2 = cur.fetchone()
        return (row2 and row2.get("dispensary_code")) or code


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
    """Resolve a product NAME to a products.json slug (exact or fuzzy substring,
    matching either the catalog name or its pinecone_title) so the assistant can
    add it to the cart."""
    if not name:
        return None
    nl = name.strip().lower()
    for slug, p in (catalog or {}).items():
        for cand in (p.get("name"), p.get("pinecone_title")):
            pn = (cand or "").lower()
            if pn and (nl == pn or (len(nl) > 4 and (nl in pn or pn in nl))):
                return slug
    return None


def is_orderable(slug, catalog=None) -> bool:
    """A product is wholesale-orderable only if it exists and is not info_only
    (external products like EMF/Kloud on the Centropix store are not)."""
    cat = catalog if catalog is not None else pricing._load_catalog()
    p = cat.get(slug)
    return bool(p) and not p.get("info_only")


def resolve_named_products(items, catalog=None) -> List[dict]:
    """From [{name, why}] (products the assistant named), keep those resolvable to a
    catalog slug, deduped. Returns [{name, why, slug}] so each gets an Add button."""
    cat = catalog if catalog is not None else pricing._load_catalog()
    out, seen = [], set()
    for it in (items or []):
        nm = (it.get("name") or "").strip()
        slug = name_to_slug(nm, cat)
        if not slug or slug in seen:
            continue
        if (cat.get(slug) or {}).get("info_only"):
            continue   # external/non-wholesale (e.g. EMF/Kloud on Centropix) — no Add button
        seen.add(slug)
        out.append({"name": nm, "why": (it.get("why") or "").strip(), "slug": slug})
    return out


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


# ── Tier-2 wholesale application + approval ────────────────────────────────────
# A third path to wholesale, ALONGSIDE licensed-on-register and coach-on-module:
# a resale-license holder applies (agreeing to the ToS), Glen/Rae approve, and the
# same wholesale_unlocked_at gate is set. application_status tracks the apply path.

def validate_wholesale_application(payload: dict) -> Tuple[Optional[dict], Optional[str]]:
    """Pure validation for the public wholesale application. Returns (clean, None)
    or (None, error). Requires name, email, a resale license number, and ToS."""
    email = (payload.get("email") or "").strip().lower()
    name = (payload.get("name") or "").strip()
    resale = (payload.get("resale_license_number") or "").strip()
    if "@" not in email or "." not in email:
        return None, "A valid email is required."
    if not name:
        return None, "Your name is required."
    if not resale:
        return None, "A resale license / certificate number is required."
    if not bool(payload.get("tos")):
        return None, "Please agree to the Terms to apply."
    return {
        "email": email, "name": name,
        "resale_license_number": resale,
        "license_state": (payload.get("license_state") or "").strip() or None,
        "practice_name": (payload.get("practice_name") or "").strip() or None,
        "credentials": (payload.get("credentials") or "").strip() or None,
        "phone": (payload.get("phone") or "").strip() or None,
        "website": (payload.get("website") or "").strip() or None,
    }, None


def submit_wholesale_application(clean: dict, *, now=None) -> Tuple[str, bool]:
    """Create or update a practitioners row in 'pending' state (no auto-unlock).
    Links to an existing row by email if present (does not overwrite an existing
    licensed/coach role, and never clears an already-granted unlock). Returns
    (practitioner_id, already_unlocked)."""
    from db_supabase import supabase_cursor
    submitted_at = _utcnow(now)
    with supabase_cursor() as cur:
        cur.execute("SELECT id, wholesale_unlocked_at FROM practitioners "
                    "WHERE lower(email)=lower(%s) LIMIT 1", (clean["email"],))
        row = cur.fetchone()
        if row:
            pid = row["id"]
            if row["wholesale_unlocked_at"] is not None:
                return str(pid), True   # already has wholesale; nothing to do
            cur.execute(
                "UPDATE practitioners SET portal_role=COALESCE(portal_role, 'reseller'), "
                "application_status='pending', application_submitted_at=%s, "
                "resale_license_number=COALESCE(resale_license_number, %s), "
                "license_state=COALESCE(license_state, %s), "
                "name=COALESCE(NULLIF(name,''), %s), practice_name=COALESCE(practice_name, %s), "
                "credentials=COALESCE(credentials, %s), phone=COALESCE(phone, %s), "
                "website=COALESCE(website, %s), updated_at=now() WHERE id=%s",
                (submitted_at, clean["resale_license_number"], clean["license_state"],
                 clean["name"], clean["practice_name"], clean["credentials"],
                 clean["phone"], clean["website"], pid),
            )
        else:
            cur.execute(
                "INSERT INTO practitioners (tier, name, email, practice_name, credentials, "
                "phone, website, portal_role, license_state, resale_license_number, "
                "application_status, application_submitted_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,'reseller',%s,%s,'pending',%s) RETURNING id",
                ("org_member", clean["name"], clean["email"], clean["practice_name"],
                 clean["credentials"], clean["phone"], clean["website"],
                 clean["license_state"], clean["resale_license_number"], submitted_at),
            )
            pid = cur.fetchone()["id"]
    return str(pid), False


def list_pending_applications() -> List[dict]:
    """Pending wholesale applicants, newest first, for the admin queue."""
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute(
            "SELECT id, name, email, practice_name, portal_role, resale_license_number, "
            "license_state, application_submitted_at FROM practitioners "
            "WHERE application_status='pending' ORDER BY application_submitted_at DESC NULLS LAST")
        rows = cur.fetchall() or []
    return [{
        "id": str(r["id"]), "name": r["name"], "email": r["email"],
        "practice_name": r["practice_name"], "portal_role": r["portal_role"],
        "resale_license_number": r["resale_license_number"],
        "license_state": r["license_state"],
        "application_submitted_at": (r["application_submitted_at"].isoformat()
                                     if r["application_submitted_at"] else None),
    } for r in rows]


def decide_application(practitioner_id, *, approve: bool, notes="", now=None) -> Optional[dict]:
    """Approve (sets application_status='approved' + grants wholesale) or reject
    (sets 'rejected', leaves the gate NULL). Returns {email, name} of the applicant
    for the decision email, or None if the id was not found."""
    from db_supabase import supabase_cursor
    ts = _utcnow(now)
    with supabase_cursor() as cur:
        if approve:
            cur.execute(
                "UPDATE practitioners SET application_status='approved', "
                "wholesale_unlocked_at=COALESCE(wholesale_unlocked_at, %s), "
                "reviewed_at=%s, approval_notes=%s, updated_at=now() "
                "WHERE id=%s RETURNING email, name",
                (ts, ts, (notes or "").strip() or None, str(practitioner_id)))
        else:
            cur.execute(
                "UPDATE practitioners SET application_status='rejected', "
                "reviewed_at=%s, approval_notes=%s, updated_at=now() "
                "WHERE id=%s RETURNING email, name",
                (ts, (notes or "").strip() or None, str(practitioner_id)))
        row = cur.fetchone()
    if not row:
        return None
    return {"email": row["email"], "name": row["name"]}


# ── portal data ───────────────────────────────────────────────────────────────

def portal_data(practitioner_id, *, db_path=None, include_orders=False) -> Optional[dict]:
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute(
            "SELECT id, name, practice_name, email, portal_role, modules_completed, "
            "wallet_balance_cents, wholesale_unlocked_at, application_status, "
            "application_submitted_at, approval_notes, resale_license_number "
            "FROM practitioners WHERE id=%s",
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
        "application_status": row["application_status"],
        "application_submitted_at": (row["application_submitted_at"].isoformat()
                                     if row["application_submitted_at"] else None),
        "approval_notes": row["approval_notes"],
        "resale_license_number": row["resale_license_number"],
        "cart": items,
        "quote": quote,
    }
    if include_orders:
        data["order_history"] = order_history(practitioner_id, db_path=db_path)
        disp = dispensary_order_history(practitioner_id, db_path=db_path)
        data["dispensary_orders"] = disp
        data["dispensary_credit_total_cents"] = sum(
            int(o.get("credit_earned_cents") or 0) for o in disp)
        try:
            data["dispensary_code"] = get_or_create_dispensary_code(practitioner_id)
        except Exception:
            data["dispensary_code"] = None
    return data
