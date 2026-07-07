"""Assemble a client's pickup invoice from the authored Biofield intake and raise it
on prod (Orders board). Pure line-assembly + injected prod calls; mirrors
biofield_fee.py. Prod is the pricing authority — this module sends [{slug,qty}] only.
"""
import difflib
import json as _json
import os
import urllib.parse
import urllib.request

BIOFIELD_SLUG = "biofield-analysis"


def resolve_line_slug(name, catalog):
    """A remedy NAME -> a sellable catalog slug. Exact (case-insensitive) first,
    then a difflib close match (cutoff 0.82). None when nothing matches."""
    name = (name or "").strip().lower()
    if not name:
        return None
    by_name = {}
    for it in catalog or []:
        n = (it.get("name") or "").strip().lower()
        if n and n not in by_name:
            by_name[n] = it.get("slug")
    if name in by_name:
        return by_name[name] or None
    match = difflib.get_close_matches(name, list(by_name.keys()), n=1, cutoff=0.82)
    return (by_name[match[0]] or None) if match else None


def build_invoice_lines(client, remedies, catalog):
    """Biofield Analysis is always lines[0]; then one qty-1 line per resolvable
    remedy (order preserved). Unresolvable names go to 'skipped', never mispriced."""
    lines = [{"slug": BIOFIELD_SLUG, "qty": 1}]
    skipped = []
    for rname in remedies or []:
        rname = (rname or "").strip()
        if not rname:
            continue
        slug = resolve_line_slug(rname, catalog)
        if slug:
            lines.append({"slug": slug, "qty": 1})
        else:
            skipped.append(rname)
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


def default_create_order(customer, lines):
    base, key = _console()
    if not base:
        return {"ok": False, "error": "No console configured (CONSOLE_SECRET missing)."}
    try:
        body = {"customer": {"name": customer.get("name") or "", "email": customer.get("email") or ""},
                "lines": lines, "pickup": True,
                "invoice_note": "Biofield Analysis and remedies. Payable by check."}
        url = f"{base}/api/orders/manual?key=" + urllib.parse.quote(key)
        req = urllib.request.Request(url, data=_json.dumps(body).encode(), method="POST",
                                     headers={"X-Console-Key": key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = _json.loads(r.read().decode() or "{}")
        if not resp.get("ok"):
            return {"ok": False, "error": resp.get("error") or "Order creation failed."}
        totals = resp.get("totals") or {}
        return {"ok": True, "order_id": resp.get("order_id"),
                "external_ref": resp.get("external_ref"),
                "total_cents": totals.get("total_cents"), "error": None}
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
