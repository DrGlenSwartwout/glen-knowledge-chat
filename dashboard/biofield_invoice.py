"""Assemble a client's pickup invoice from the authored Biofield intake and raise it
on prod (Orders board). Pure line-assembly + injected prod calls; mirrors
biofield_fee.py. Prod is the pricing authority — this module sends [{slug,qty}] only.
"""
import json as _json
import os
import urllib.parse
import urllib.request

BIOFIELD_SLUG = "biofield-analysis"


def resolve_line_slug(name, catalog):
    """A remedy NAME -> a sellable catalog slug by EXACT (case-insensitive) match.
    No fuzzy matching: on an invoice a near-name substitution could bill the wrong
    SKU (ES1 vs ES13, Vitamin A vs Vitamin D). A non-exact name returns None and the
    caller lists it as skipped for manual add against the real catalog."""
    name = (name or "").strip().lower()
    if not name:
        return None
    for it in catalog or []:
        if (it.get("name") or "").strip().lower() == name:
            return it.get("slug") or None
    return None


def doses_per_day(freq_text):
    """A per-client frequency phrase -> doses/day, or None if unrecognized.
    Handles 'daily', 'a day', 'twice a day', 'two times a day', '3 times a day', '2x'."""
    import re
    t = (freq_text or "").strip().lower()
    if not t:
        return None
    m = re.search(r"(\d+)\s*(?:x|times?)\b", t)          # "3 times a day", "2x"
    if m:
        return int(m.group(1))
    words = {"once": 1, "one": 1, "twice": 2, "two": 2, "thrice": 3, "three": 3, "four": 4}
    for w, n in words.items():
        if re.search(rf"\b{w}\b", t) and re.search(r"\b(times?|x|a day|per day|daily|day)\b", t):
            return n
    if re.search(r"\b(daily|a day|per day|each day|every day)\b", t):
        return 1
    return None


def bottles_needed(freq_text, doses_per_bottle, program_days=30):
    """Bottles for the program = ceil(doses/day * days / doses_per_bottle), >= 1.
    Falls back to 1 when the frequency is unparseable OR doses_per_bottle is missing
    (e.g. infoceuticals carry no doses_per_bottle -> qty 1)."""
    import math
    try:
        dpb = int(doses_per_bottle)
    except (TypeError, ValueError):
        dpb = 0
    dpd = doses_per_day(freq_text)
    if not dpd or dpb <= 0:
        return 1
    return max(1, math.ceil(dpd * program_days / dpb))


def build_invoice_lines(client, remedies, catalog, include_fee=True):
    """Biofield Analysis is lines[0] (unless include_fee is False — e.g. the client
    already PAID for the analysis, so we invoice remedies only); then one line per
    resolvable remedy (order preserved). A remedy is a name string (qty 1) or a
    {"name","qty"} dict (qty = bottles needed). Unresolvable names go to 'skipped',
    never mispriced."""
    lines = [{"slug": BIOFIELD_SLUG, "qty": 1}] if include_fee else []
    skipped = []
    for r in remedies or []:
        if isinstance(r, dict):
            name, qty = (r.get("name") or "").strip(), r.get("qty")
        else:
            name, qty = (r or "").strip(), 1
        if not name:
            continue
        try:
            qty = max(1, int(qty))
        except (TypeError, ValueError):
            qty = 1
        slug = resolve_line_slug(name, catalog)
        if slug:
            lines.append({"slug": slug, "qty": qty})
        else:
            skipped.append(name)
    return {"lines": lines, "skipped": skipped}


def _console():
    key = os.environ.get("CONSOLE_SECRET")
    if not key:
        return None, None
    base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
    return base, key


def default_fetch_catalog():
    base, key = _console()
    if not base:
        return []
    try:
        url = f"{base}/api/console/biofield-portal/catalog?key=" + urllib.parse.quote(key)
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        with urllib.request.urlopen(req, timeout=8) as r:
            resp = _json.loads(r.read().decode() or "{}")
        return resp.get("products") or []
    except Exception:
        return []


def default_create_order(customer, lines, replace_open=False):
    """Create the hand-off invoice on prod. With replace_open=True (a re-hand-off),
    prod first cancels the client's prior OPEN hand-off drafts (proposed, unpaid, not
    yet published) so a repeated hand-off UPDATES rather than piling up duplicates;
    published/paid orders are never touched."""
    base, key = _console()
    if not base:
        return {"ok": False, "error": "The console connection is not configured."}
    try:
        body = {"customer": {"name": customer.get("name") or "", "email": customer.get("email") or ""},
                "lines": lines, "pickup": True, "replace_open": bool(replace_open),
                "invoice_note": "Biofield Analysis and remedies. Payable by check."}
        url = f"{base}/api/orders/manual?key=" + urllib.parse.quote(key)
        req = urllib.request.Request(url, data=_json.dumps(body).encode(), method="POST",
                                     headers={"X-Console-Key": key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = _json.loads(r.read().decode() or "{}")
        if not resp.get("ok"):
            return {"ok": False, "error": resp.get("error") or "Order creation failed."}
        totals = resp.get("totals") or {}
        accepted = [ (l or {}).get("slug") for l in (resp.get("lines") or []) if (l or {}).get("slug") ]
        return {"ok": True, "order_id": resp.get("order_id"),
                "external_ref": resp.get("external_ref"),
                "total_cents": totals.get("total_cents"),
                "cancelled": resp.get("cancelled") or [],
                "accepted_slugs": accepted, "error": None}
    except Exception:
        return {"ok": False, "error": "Couldn't reach the console to create the order."}


def default_invoice_link(order_id):
    base, key = _console()
    if not base or not order_id:
        return {"ok": False, "error": "link unavailable"}
    try:
        url = (f"{base}/api/console/order/{int(order_id)}/invoice-link?key="
               + urllib.parse.quote(key))
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = _json.loads(r.read().decode() or "{}")
        if resp.get("ok") and resp.get("link"):
            return {"ok": True, "print_url": resp["link"], "error": None}
        return {"ok": False, "error": "link unavailable"}
    except Exception:
        return {"ok": False, "error": "link unavailable"}


def default_orders_link(order_id):
    """The prod Orders-board deep link for an order (Edit action), key-carried the
    same way the local console tools bounce to prod. '' when unconfigured."""
    base, key = _console()
    if not base or not order_id:
        return ""
    return f"{base}/console/orders?order={int(order_id)}&key={urllib.parse.quote(key)}"


def default_publish_invoice(order_id):
    """POST the prod publish-to-portal endpoint for an order, so it shows as a pay
    card on the client's portal. Returns {ok, link} or {ok:False, error}."""
    base, key = _console()
    if not base or not order_id:
        return {"ok": False, "error": "publish unavailable (no console config)"}
    try:
        url = (f"{base}/api/console/order/{int(order_id)}/publish-to-portal?key="
               + urllib.parse.quote(key))
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json", "X-Console-Key": key})
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = _json.loads(r.read().decode() or "{}")
        return resp if isinstance(resp, dict) else {"ok": False, "error": "bad response"}
    except Exception:
        return {"ok": False, "error": "publish failed"}


def default_biofield_paid(email):
    """Has this client already PAID for a Biofield Analysis? Asks prod for a paid,
    non-cancelled order carrying the biofield-analysis line. Returns {paid, order_id,
    paid_at} (paid False when none / unreachable) so the raise can drop the fee line."""
    base, key = _console()
    if not base or not (email or "").strip():
        return {"paid": False}
    try:
        url = (f"{base}/api/console/biofield-analysis-paid?email="
               + urllib.parse.quote(email) + "&key=" + urllib.parse.quote(key))
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        with urllib.request.urlopen(req, timeout=8) as r:
            resp = _json.loads(r.read().decode() or "{}")
        return resp if isinstance(resp, dict) else {"paid": False}
    except Exception:
        return {"paid": False}


def default_handoff_push(email, name, content, scan_date=""):
    """Hand off to Rae: push a portal-seed `content` (built from the authored chain)
    to prod as a portal ai_draft (staged for Rae to review + publish from the console).
    Reuses /admin/portal/upsert. Returns {ok, ...} or {ok:False, error}."""
    base, key = _console()
    if not base or not (email or "").strip():
        return {"ok": False, "error": "handoff unavailable (no console config / email)"}
    payload = dict(content or {})
    payload["biofield_status"] = "ai_draft"
    # Stamp this hand-off's report as the client's CURRENT one so it wins over a stale
    # AI reveal (which owns its own per-scan report row) regardless of scan date.
    if (scan_date or "").strip():
        payload["current_scan_date"] = scan_date.strip()
    body = _json.dumps({"email": email, "name": name or "", "content": payload,
                        "scan_date": scan_date or "", "scan_id": ""}).encode("utf-8")
    try:
        req = urllib.request.Request(f"{base}/admin/portal/upsert", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "X-Console-Key": key})
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = _json.loads(r.read().decode() or "{}")
        return resp if isinstance(resp, dict) else {"ok": True}
    except Exception:
        return {"ok": False, "error": "handoff push failed"}
