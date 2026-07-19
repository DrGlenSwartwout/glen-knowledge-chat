"""Role-aware portal view assembler.

`get_portal_view(cx, person_id)` composes ONE payload from the unified person
row plus orders, points, and the existing biofield portal content. The page and
APIs render whichever blocks come back; visibility is driven by roles, and any
absent/unavailable data hides its block rather than erroring.

Self-contained (takes a `cx`, never imports `app`) so it unit-tests in isolation.
Order/points/biofield reads are defensive: a failure degrades to an empty block.
"""
import datetime
import json

from dashboard import affiliate_dashboard as _ad
from dashboard import client_portal as _cp
from dashboard import entity_refs as _er
from dashboard import portal_biofield_reports as _pbr
from dashboard import portal_offers as _po


def _product_exists(slug):
    """True when `slug` is a live, non-superseded product page — the only case
    where a remedy earns a click-through link. A catalog miss or a superseded
    slug degrades to a pop-up only (never a wrong/dead link)."""
    try:
        from dashboard import products as _pr
        return bool(slug) and slug in _pr.load_products() and not _pr.superseded_slug(slug)
    except Exception:
        return False


def entity_refs_remedy(cx, name):
    """Seam (monkeypatchable in tests): remedy -> {name, info, href}."""
    return _er.remedy_ref(cx, name, product_exists=_product_exists)


def entity_refs_function(cx, title):
    """Seam (monkeypatchable in tests): layer title/function -> {name, info, href}."""
    return _er.function_ref(cx, title)

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


def _biofield_block(cx, email, scan_date=None, unlocked=True):
    """The 'healing adventure' map — per-scan from portal_biofield_reports
    (newest by default, or an explicit scan_date), falling back to the legacy
    client_portals content as a single confirmed report when no rows exist.

    `unlocked` = the client has PAID for this content (paid Biofield Analysis or
    active membership). When False, remedies/dosing/pricing stay blurred even on a
    'confirmed' report — a free E4L reveal published to the portal doesn't hand out
    the product list until they pay. Caller computes it (this module stays pure)."""
    try:
        _pbr.init_table(cx)
        dates = _pbr.list_report_dates(cx, email)
    except Exception:
        dates = []
    if dates:
        picked = scan_date if (scan_date in dates) else dates[0]
        rep = _pbr.get_report(cx, email, picked) or {}
        content = rep.get("content") or {}
        status = rep.get("status") or "confirmed"
        today = datetime.date.today().isoformat()
        actionable = (status != "confirmed") and _pbr.is_actionable(picked, today)
        return _assemble_biofield(cx, content, status, scan_date=picked,
                                  scan_dates=dates, actionable=actionable, unlocked=unlocked)
    # Legacy fallback: single confirmed report, no tabs.
    try:
        rec = _cp.get_portal_content_by_email(cx, email)
    except Exception:
        rec = None
    content = (rec or {}).get("content") or {}
    if not (content.get("greeting") or content.get("layers") or content.get("video")):
        return {"visible": False}
    # Legacy portals (no biofield_status) are treated as confirmed → render fully.
    status = content.get("biofield_status") or "confirmed"
    return _assemble_biofield(cx, content, status, scan_date=None,
                              scan_dates=[], actionable=False, unlocked=unlocked)


def _assemble_biofield(cx, content, status, *, scan_date, scan_dates, actionable, unlocked=True):
    # A report's remedies un-blur only when it's confirmed AND the client has paid.
    # `status` is still reported truthfully (the report exists); only the gated
    # content and `blurred` flag depend on payment.
    show = (status == "confirmed") and bool(unlocked)
    layers = []
    for L in (content.get("layers") or []):
        item = {"n": L.get("n"), "title": L.get("title", ""), "meaning": L.get("meaning", "")}
        fn = entity_refs_function(cx, item["title"])
        item["function_info"] = fn["info"]
        item["function_href"] = fn["href"] or ""
        if show:  # unconfirmed OR unpaid remedies NEVER leave the server
            item["remedy"] = L.get("remedy", "")
            item["dosing"] = L.get("dosing", "")
            rr = entity_refs_remedy(cx, item["remedy"]) if item["remedy"] else {"info": "", "href": None}
            item["remedy_info"] = rr["info"]
            item["remedy_href"] = rr["href"] or ""
        layers.append(item)
    return {"visible": True, "status": status, "blurred": not show,
            "actionable": actionable, "scan_date": scan_date, "scan_dates": scan_dates,
            "greeting": content.get("greeting", ""), "video": content.get("video") or {},
            "layers": layers, "pricing_note": content.get("pricing_note", "") if show else ""}


def _reveal_as_report_content(reveal):
    """Normalize a biofield_reveals row into the portal report-content shape so the
    existing assemblers render it identically to a System B report. Remedy/dosing
    are strings; the caller's blur gate decides whether they leave the server."""
    greeting = ((reveal.get("interpretation") or {}).get("greeting") or "").strip()
    layers = []
    raw = reveal.get("layers") or []
    if raw:
        for L in raw:
            rem = L.get("remedy") if isinstance(L.get("remedy"), dict) else {}
            layers.append({
                "n": L.get("n"),
                "title": L.get("title", "") or "",
                "meaning": (L.get("meaning") or L.get("summary") or ""),
                "remedy": (rem.get("name") or ""),
                "dosing": (rem.get("dosing") or ""),
            })
    else:
        for i, r in enumerate(reveal.get("remedies") or []):
            if not isinstance(r, dict):
                continue
            layers.append({"n": i + 1, "title": "", "meaning": (r.get("meaning") or ""),
                           "remedy": (r.get("name") or ""), "dosing": (r.get("dosing") or "")})
    return {"greeting": greeting, "layers": layers, "video": {}}


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


def _ambassador_block(cx, email, quiz_url, public_base_url):
    """Affiliate/ambassador status for the personal portal, by email. None-raising.
    enrolled -> referral links (from slug); pending -> under review; else signup CTA."""
    em = (email or "").strip().lower()
    base = (public_base_url or "").rstrip("/")
    signup = {"status": "none", "signup_url": f"{base}/affiliate/apply-form"}
    if not em:
        return signup
    try:
        row = cx.execute(
            "SELECT slug, status FROM affiliate_signups WHERE lower(email)=? LIMIT 1",
            (em,)).fetchone()
    except Exception:
        return signup
    if not row:
        if _ad.autoenroll_enabled():
            made = _ad.ensure_affiliate(cx, em, name="")
            if made and made.get("slug"):
                row = (made["slug"], made.get("status") or "approved")
        if not row:
            return signup
    slug, status = row[0], (row[1] or "")
    if status != "approved":
        return {"status": "pending"}
    block = {
        "status": "enrolled",
        "slug": slug,
        "referral_url": f"{quiz_url}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz",
        "recruit_url": f"{base}/affiliate?ref={slug}",
    }
    try:
        block["dashboard"] = _ad.build_dashboard(cx, slug, quiz_url=quiz_url, public_base_url=public_base_url)
    except Exception:
        pass
    return block


def _practitioner_finder_block(address, enabled):
    """Prefill data for the embedded /practitioner-finder card. Zip beats city
    (more precise); country defaults to US. An absent address yields an empty
    location so the finder falls back to its own type-to-search default."""
    address = address or {}
    location = (address.get("zip") or "").strip() or (address.get("city") or "").strip()
    country = (address.get("country") or "").strip() or "US"
    return {"enabled": bool(enabled), "location": location, "country": country}


def _consult_block(cx, email):
    """Ready/booked status + objective stage checklist for the portal's Biofield
    Consult card. Defensive: any failure (missing tables, import error) falls
    back to a safe not-ready/not-booked default so the consult block never
    breaks the rest of the portal payload."""
    from dashboard import consult as _consult
    try:
        _consult.init_consult_tables(cx)
        ready = _consult.consult_is_ready(cx, email)
        paid = _consult.has_paid_purchase(cx, email, _consult.CONSULT["test_slug"])
        booked_start = None
        try:
            row = cx.execute("SELECT start_ts FROM evox_bookings WHERE lower(email)=? "
                             "AND session_type='biofield-consult' AND status='booked' "
                             "ORDER BY start_ts DESC LIMIT 1", (email,)).fetchone()
            booked_start = row[0] if row else None
        except Exception:
            pass
        return {"ready": ready, "booked": booked_start is not None,
                "booked_start": booked_start,
                "stages": {"test_paid": paid, "ready": ready}}
    except Exception:
        return {"ready": False, "booked": False, "booked_start": None, "stages": {}}


def _onboarding_block(cx, email):
    """Whether the member has a booked new-member welcome call. Membership
    eligibility is decided by the client JS via /api/onboarding/state (this
    layer has no app import); here `eligible` is a render hint that is True only
    when a booking already exists. Defensive: any failure falls back to a safe
    not-eligible/not-booked default so it never breaks the portal payload."""
    from dashboard import onboarding as _ob
    try:
        row = _ob.existing_onboarding(cx, email)
        start = row["start_ts"] if row else None
        return {"eligible": start is not None, "booked_start": start}
    except Exception:
        return {"eligible": False, "booked_start": None}


def get_portal_view(cx, person_id, *, offers_enabled_keys=None, scan_date=None,
                    quiz_url="", public_base_url="", finder_enabled=False,
                    biofield_unlocked=True):
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
        "biofield": _biofield_block(cx, email, scan_date=scan_date, unlocked=biofield_unlocked),
        "upgrade": _upgrade_block(cx, email, roles, offers_enabled_keys),
        "ambassador": _ambassador_block(cx, email, quiz_url, public_base_url),
        "practitioner_finder": _practitioner_finder_block(account["address"], finder_enabled),
        "consult": _consult_block(cx, email),
        "onboarding": _onboarding_block(cx, email),
    }
