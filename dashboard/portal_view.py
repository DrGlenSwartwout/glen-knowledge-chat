"""Role-aware portal view assembler.

`get_portal_view(cx, person_id)` composes ONE payload from the unified person
row plus orders, points, and the existing biofield portal content. The page and
APIs render whichever blocks come back; visibility is driven by roles, and any
absent/unavailable data hides its block rather than erroring.

Self-contained (takes a `cx`, never imports `app`) so it unit-tests in isolation.
Order/points/biofield reads are defensive: a failure degrades to an empty block.
"""
import json

from dashboard import client_portal as _cp
from dashboard import portal_offers as _po

# roles → human-friendly badge labels. Roles not listed fall back to Title Case.
_BADGE = {
    "client": "Client",
    "student": "Student",
    "practitioner": "Practitioner",
    "affiliate": "Affiliate",
    "wholesale": "Wholesale",
}

_ADDRESS_KEYS = ("address1", "address2", "city", "state", "zip", "country")


def _safe_points_cents(cx, email):
    try:
        from dashboard import points as _pts
        return int(_pts.balance(cx, email))
    except Exception:
        return 0


def _orders_block(cx, email, roles):
    """Order history, visible to clients (the default role). Summarized to what
    the shell needs; full detail stays in the order/invoice surfaces."""
    visible = ("client" in roles) or (not roles)
    if not visible:
        return {"visible": False, "items": []}
    items = []
    try:
        import sqlite3
        from dashboard import orders as _o
        cx.row_factory = sqlite3.Row
        for o in _o.list_orders_by_email(cx, email, limit=50):
            if (o.get("status") or "") == "cancelled":
                continue  # clients never see cancelled orders
            items.append({
                "id": o.get("id"),
                "date": o.get("created_at", ""),
                "total_cents": int(o.get("total_cents") or 0),
                "status": o.get("status", ""),
            })
    except Exception:
        items = []
    return {"visible": True, "items": items}


def _biofield_block(cx, email):
    """The 'healing adventure' map — reuses the existing client_portals content,
    now rendered as one section of the shell rather than the whole page."""
    try:
        rec = _cp.get_portal_content_by_email(cx, email)
    except Exception:
        rec = None
    content = (rec or {}).get("content") or {}
    has = bool(content.get("greeting") or content.get("layers") or content.get("video"))
    if not has:
        return {"visible": False}
    # Legacy portals (no biofield_status) are treated as confirmed → render fully.
    status = content.get("biofield_status") or "confirmed"
    confirmed = status == "confirmed"
    layers = []
    for L in (content.get("layers") or []):
        item = {"n": L.get("n"), "title": L.get("title", ""), "meaning": L.get("meaning", "")}
        if confirmed:  # unconfirmed remedies NEVER leave the server
            item["remedy"] = L.get("remedy", "")
            item["dosing"] = L.get("dosing", "")
        layers.append(item)
    return {
        "visible": True,
        "status": status,
        "blurred": not confirmed,
        "greeting": content.get("greeting", ""),
        "video": content.get("video") or {},
        "layers": layers,
        "pricing_note": content.get("pricing_note", "") if confirmed else "",
    }


def _upgrade_block(cx, email, roles, enabled_keys):
    """The single next eligible ladder rung, or disabled when none/flags off."""
    if not enabled_keys:
        return {"enabled": False}
    try:
        offers = _po.next_offers(cx, email, roles, enabled_keys=enabled_keys)
    except Exception:
        offers = []
    if not offers:
        return {"enabled": False}
    return {"enabled": True, "offer": offers[0]}


def get_portal_view(cx, person_id, *, offers_enabled_keys=None):
    import sqlite3
    cx.row_factory = sqlite3.Row
    prow = cx.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
    if not prow:
        return None
    p = {k: prow[k] for k in prow.keys()}
    email = (p.get("email") or "").strip().lower()
    try:
        roles = list(json.loads(p.get("roles") or "[]"))
    except Exception:
        roles = []

    name = (p.get("name") or "").strip() or \
        ((p.get("first_name", "") or "") + " " + (p.get("last_name", "") or "")).strip()
    account = {
        "name": name,
        "email": email,
        "address": {k: (p.get(k) or "") for k in _ADDRESS_KEYS},
        "points_cents": _safe_points_cents(cx, email),
        "roles": roles,
        "role_badges": [_BADGE.get(r, r.replace("_", " ").title()) for r in roles],
    }
    return {
        "person_id": person_id,
        "roles": roles,
        "account": account,
        "orders": _orders_block(cx, email, roles),
        "biofield": _biofield_block(cx, email),
        "upgrade": _upgrade_block(cx, email, roles, offers_enabled_keys),
    }
