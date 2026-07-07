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
