"""Console practitioner-admin — logic layer.

Backs the /console/practitioners page: a console-gated "add practitioner" form
and a roster (grouped Coaches / Practitioners) with per-row stats and edit
actions. The Supabase reads/writes live here too, but the pure pieces — input
validation, SQLite activity aggregation, and the row-merge that feeds the UI —
are split out so they're testable without a database.

Reuses dashboard.practitioner_portal for the SQLite order tables, magic-link
tokens, and the geocoder where possible.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from dashboard import practitioner_portal as pp

VALID_ROLES = ("coach", "licensed")
N_MODULES = 12


# ── validation ──────────────────────────────────────────────────────────────────

def validate_new_practitioner(payload: dict) -> Tuple[Optional[dict], Optional[str]]:
    """Pure validation for the add-practitioner form. Returns (clean, None) or
    (None, error_message). Cert level (0-12) is independent of wholesale access."""
    email = (payload.get("email") or "").strip().lower()
    name = (payload.get("name") or "").strip()
    role = (payload.get("role") or "").strip().lower()
    if "@" not in email or "." not in email:
        return None, "A valid email is required."
    if not name:
        return None, "Your name is required."
    if role not in VALID_ROLES:
        return None, "Choose a role: coach or licensed practitioner."
    raw_level = payload.get("level", 0)
    try:
        level = int(raw_level)
    except (TypeError, ValueError):
        return None, "Cert level must be a number 0–12."
    level = max(0, min(N_MODULES, level))
    return {
        "email": email,
        "name": name,
        "portal_role": role,
        "credentials": (payload.get("credentials") or "").strip() or None,
        "wholesale_access": bool(payload.get("wholesale_access")),
        "level": level,
        "list_in_finder": bool(payload.get("list_in_finder")),
        "city": (payload.get("city") or "").strip() or None,
        "state": (payload.get("state") or "").strip().upper() or None,
        "send_invite": bool(payload.get("send_invite")),
    }, None


# ── SQLite activity aggregation ──────────────────────────────────────────────────

def aggregate_activity(db_path: str) -> dict:
    """Per-practitioner order + dispensary aggregates, keyed by practitioner_id.
    Returns {pid: {orders, spent_cents, last_order, disp_count, disp_credit_cents,
    disp_bottles}}. A practitioner with no activity is simply absent."""
    out: dict = {}
    with sqlite3.connect(db_path) as cx:
        pp._ensure_orders_table(cx)
        pp._ensure_dispensary_table(cx)
        for pid, n, spent, last in cx.execute(
            "SELECT practitioner_id, COUNT(*), COALESCE(SUM(total_cents),0), MAX(created_at) "
            "FROM wholesale_orders GROUP BY practitioner_id"
        ).fetchall():
            out.setdefault(pid, {}).update(
                {"orders": n, "spent_cents": int(spent or 0), "last_order": last})
        for pid, n, credit, bottles in cx.execute(
            "SELECT practitioner_id, COUNT(*), COALESCE(SUM(credit_earned_cents),0), "
            "COALESCE(SUM(bottles),0) FROM dispensary_orders GROUP BY practitioner_id"
        ).fetchall():
            out.setdefault(pid, {}).update(
                {"disp_count": n, "disp_credit_cents": int(credit or 0),
                 "disp_bottles": int(bottles or 0)})
    return out


# ── row merge (feeds the roster UI) ──────────────────────────────────────────────

_ACTIVITY_DEFAULTS = {
    "orders": 0, "spent_cents": 0, "last_order": None,
    "disp_count": 0, "disp_credit_cents": 0, "disp_bottles": 0,
}

_LIST_COLS = (
    "id, name, email, portal_role, credentials, modules_completed, "
    "wallet_balance_cents, wholesale_unlocked_at, application_status, "
    "show_contact, city, state"
)
_TIER_FOR_ROLE = {"coach": "panel_in_cert", "licensed": "org_member"}


# ── Supabase reads/writes (thin; monkeypatched in route tests) ───────────────────

def list_practitioners(q: Optional[str] = None) -> List[dict]:
    """Every portal practitioner (portal_role set), newest-active first. Optional
    free-text filter across name/email/credentials/city."""
    from db_supabase import supabase_cursor
    sql = f"SELECT {_LIST_COLS} FROM practitioners WHERE portal_role IS NOT NULL"
    params: list = []
    term = (q or "").strip()
    if term:
        sql += (" AND (name ILIKE %s OR email ILIKE %s OR credentials ILIKE %s "
                "OR city ILIKE %s)")
        params += [f"%{term}%"] * 4
    sql += " ORDER BY name NULLS LAST"
    with supabase_cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in (cur.fetchall() or [])]


def create_or_update_practitioner(clean: dict, *, now=None) -> str:
    """Insert or link (by email) a portal practitioner from the add form. Cert level
    and wholesale access are set independently. Returns the practitioner_id."""
    from db_supabase import supabase_cursor
    ts = now or datetime.now(timezone.utc)
    unlocked = ts if clean["wholesale_access"] else None
    tier = _TIER_FOR_ROLE.get(clean["portal_role"], "org_member")
    with supabase_cursor() as cur:
        cur.execute("SELECT id FROM practitioners WHERE lower(email)=lower(%s) LIMIT 1",
                    (clean["email"],))
        row = cur.fetchone()
        if row:
            pid = row["id"]
            cur.execute(
                "UPDATE practitioners SET portal_role=%s, credentials=COALESCE(%s, credentials), "
                "modules_completed=%s, "
                "wholesale_unlocked_at=CASE WHEN %s THEN COALESCE(wholesale_unlocked_at, %s) "
                "ELSE NULL END, "
                "show_contact=%s, city=COALESCE(%s, city), state=COALESCE(%s, state), "
                "name=COALESCE(NULLIF(name,''), %s), updated_at=now() WHERE id=%s",
                (clean["portal_role"], clean["credentials"], clean["level"],
                 clean["wholesale_access"], ts, clean["list_in_finder"],
                 clean["city"], clean["state"], clean["name"], pid))
        else:
            cur.execute(
                "INSERT INTO practitioners (tier, name, email, portal_role, credentials, "
                "modules_completed, wholesale_unlocked_at, show_contact, city, state) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (tier, clean["name"], clean["email"], clean["portal_role"],
                 clean["credentials"], clean["level"], unlocked,
                 clean["list_in_finder"], clean["city"], clean["state"]))
            pid = cur.fetchone()["id"]
    return str(pid)


def set_level_and_access(pid: str, level: int, wholesale_access: bool, *, now=None) -> None:
    """Set cert level (0-12) and toggle wholesale access independently. Granting
    access keeps any existing unlock timestamp; revoking clears it."""
    from db_supabase import supabase_cursor
    ts = now or datetime.now(timezone.utc)
    lvl = max(0, min(N_MODULES, int(level)))
    with supabase_cursor() as cur:
        cur.execute(
            "UPDATE practitioners SET modules_completed=%s, "
            "wholesale_unlocked_at=CASE WHEN %s THEN COALESCE(wholesale_unlocked_at, %s) "
            "ELSE NULL END, updated_at=now() WHERE id=%s",
            (lvl, bool(wholesale_access), ts, str(pid)))


def set_finder_visibility(pid: str, show: bool) -> None:
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute("UPDATE practitioners SET show_contact=%s, updated_at=now() WHERE id=%s",
                    (bool(show), str(pid)))


def geocode_and_set_location(pid: str, city: Optional[str], state: Optional[str]) -> None:
    """Set city/state and (best-effort) city-level lat/lng via Mapbox so the
    practitioner places in the finder. lat/lng/quality only set if geocoding hits."""
    from db_supabase import supabase_cursor
    from scrapers.practitioner_finder.geocode import geocode_place
    city = (city or "").strip() or None
    state = (state or "").strip().upper() or None
    lat = lng = None
    if city or state:
        place = ", ".join([p for p in (city, state, "USA") if p])
        try:
            lat, lng = geocode_place(place, "US")
        except Exception:
            lat = lng = None
    with supabase_cursor() as cur:
        if lat is not None and lng is not None:
            cur.execute(
                "UPDATE practitioners SET city=%s, state=%s, country='US', lat=%s, lng=%s, "
                "geocode_quality='city', updated_at=now() WHERE id=%s",
                (city, state, lat, lng, str(pid)))
        else:
            cur.execute("UPDATE practitioners SET city=%s, state=%s, updated_at=now() WHERE id=%s",
                        (city, state, str(pid)))


def build_rows(practitioners: List[dict], activity: dict) -> List[dict]:
    """Merge Supabase practitioner records with their SQLite activity, deriving the
    booleans + section the roster UI groups on. Activity defaults to zeros."""
    rows = []
    for p in practitioners:
        pid = str(p.get("id"))
        act = {**_ACTIVITY_DEFAULTS, **(activity.get(pid) or {})}
        role = p.get("portal_role")
        rows.append({
            "id": pid,
            "name": p.get("name"),
            "email": p.get("email"),
            "portal_role": role,
            "credentials": p.get("credentials"),
            "level": int(p.get("modules_completed") or 0),
            "wallet_balance_cents": int(p.get("wallet_balance_cents") or 0),
            "wholesale_access": p.get("wholesale_unlocked_at") is not None,
            "application_status": p.get("application_status"),
            "finder_listed": bool(p.get("show_contact")),
            "city": p.get("city"),
            "state": p.get("state"),
            "section": "coach" if role == "coach" else "practitioner",
            **act,
        })
    return rows
