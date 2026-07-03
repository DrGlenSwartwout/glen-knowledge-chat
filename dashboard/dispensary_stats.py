"""Dispensary product-dispense ranking + practice-type FF recommendations.

Pure ranker + defensive DB collector + curated-recommendations resolver.
Self-contained (never imports app); DB reads never raise (a failure yields an
empty result), matching the portal_view.py convention.
"""
import json
import os
import sqlite3
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"


def _catalog(catalog=None) -> dict:
    if catalog is not None:
        return catalog
    try:
        return json.loads((_DATA / "products.json").read_text()).get("products", {})
    except Exception:
        return {}


def _name(slug, cat) -> str:
    return (cat.get(slug) or {}).get("name") or slug


def _url(slug) -> str:
    return f"/begin/product/{slug}"


def _log_db(db_path=None) -> str:
    if db_path:
        return db_path
    base = os.environ.get("DATA_DIR") or str(Path(__file__).resolve().parent.parent)
    return str(Path(base) / "chat_log.db")


# ── Section 1: ranked dispense table ───────────────────────────────────────

def rank_dispense_rows(dispensed, dropshipped, patient_portal, *, catalog=None):
    """Merge three {slug: units} channel maps into ranked rows.
    Row: {slug, name, url, dispensed, dropshipped, patient_portal, total}.
    Sorted by total desc, then name."""
    cat = _catalog(catalog)
    slugs = set(dispensed) | set(dropshipped) | set(patient_portal)
    rows = []
    for s in slugs:
        d = int(dispensed.get(s, 0))
        ds_ = int(dropshipped.get(s, 0))
        pp = int(patient_portal.get(s, 0))
        rows.append({"slug": s, "name": _name(s, cat), "url": _url(s),
                     "dispensed": d, "dropshipped": ds_, "patient_portal": pp,
                     "total": d + ds_ + pp})
    rows.sort(key=lambda r: (-r["total"], r["name"].lower(), r["slug"]))
    return rows


def _items_for_invoices(cx, invoice_ids, source):
    """{slug: units} summed across the given orders' items_json, scoped to the
    given `source` (orders is UNIQUE(source, external_ref)); newest row wins."""
    out = {}
    for inv in invoice_ids:
        try:
            row = cx.execute(
                "SELECT items_json FROM orders WHERE external_ref=? AND source=? ORDER BY id DESC LIMIT 1",
                (inv, source)).fetchone()
        except Exception:
            continue
        if not row or not row[0]:
            continue
        try:
            parsed = json.loads(row[0])
        except Exception:
            continue
        for it in parsed:
            try:
                s = it.get("slug")
                if s:
                    out[s] = out.get(s, 0) + int(it.get("qty") or 0)
            except Exception:
                continue  # one malformed line never drops the rest of the invoice
    return out


def dispense_stats(practitioner_id, *, db_path=None, catalog=None):
    """Collect the practitioner's per-product units across channels and rank them.
    Dispensed = their own wholesale_orders (orders.items_json, source='wholesale').
    Drop-shipped + Patient portal are DEFERRED ({}) — those sale flows record only
    aggregate bottles today (no per-product slug), so per-product ranking of them
    is not yet supported; the UI marks both columns 'coming soon'. Never raises."""
    try:
        with sqlite3.connect(_log_db(db_path)) as cx:
            w = [r[0] for r in cx.execute(
                "SELECT invoice_id FROM wholesale_orders WHERE practitioner_id=?", (str(practitioner_id),))]
            dispensed = _items_for_invoices(cx, w, "wholesale")
    except Exception:
        return []
    return rank_dispense_rows(dispensed, {}, {}, catalog=catalog)


# ── Section 2: curated practice-type recommendations ───────────────────────

def _practice_tokens(practice_type):
    """Candidate match tokens from a free-form credentials string, so a compound
    value like 'OD, FAAO' or 'Health Coach, RN' still resolves to its key."""
    pt = (practice_type or "").strip().lower()
    toks = {pt}
    for part in pt.replace("/", ",").split(","):
        part = part.strip()
        if part:
            toks.add(part)                 # comma part preserves multi-word keys ("health coach")
            toks.update(part.split())      # single-word tokens ("od" from "od faao")
    return toks


def recommended_ffs(practice_type, *, exclude_slugs=(), recs_path=None, catalog=None):
    """Resolve practice_type to its curated FF list, else 'default'; drop
    exclude_slugs; return [{slug, name, url, blurb}]. Matching tokenizes the
    (free-form) credentials so 'OD, FAAO' resolves to the 'OD' list."""
    try:
        path = recs_path or str(_DATA / "practice_recommendations.json")
        recs = json.loads(Path(path).read_text())
    except Exception:
        return []
    toks = _practice_tokens(practice_type)
    key = next((k for k in recs if k != "default" and k.lower() in toks), "default")
    cat = _catalog(catalog)
    ex = {s for s in (exclude_slugs or ())}
    out = []
    for r in recs.get(key, []):
        s = r.get("slug")
        if not s or s in ex:
            continue
        out.append({"slug": s, "name": _name(s, cat), "url": _url(s), "blurb": r.get("blurb", "")})
    return out
