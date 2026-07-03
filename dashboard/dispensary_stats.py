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
    rows.sort(key=lambda r: (-r["total"], r["name"].lower()))
    return rows


def _items_for_invoices(cx, invoice_ids):
    """{slug: units} summed across the given orders' items_json (by external_ref)."""
    out = {}
    for inv in invoice_ids:
        row = cx.execute("SELECT items_json FROM orders WHERE external_ref=? LIMIT 1", (inv,)).fetchone()
        if not row or not row[0]:
            continue
        try:
            for it in json.loads(row[0]):
                s = it.get("slug")
                if s:
                    out[s] = out.get(s, 0) + int(it.get("qty") or 0)
        except Exception:
            continue
    return out


def dispense_stats(practitioner_id, *, db_path=None, catalog=None):
    """Collect the practitioner's per-product units across channels and rank them.
    Dispensed = their own wholesale_orders; Drop-shipped = dispensary_orders;
    Patient portal = {} (deferred). Never raises."""
    try:
        with sqlite3.connect(_log_db(db_path)) as cx:
            w = [r[0] for r in cx.execute(
                "SELECT invoice_id FROM wholesale_orders WHERE practitioner_id=?", (str(practitioner_id),))]
            d = [r[0] for r in cx.execute(
                "SELECT invoice_id FROM dispensary_orders WHERE practitioner_id=?", (str(practitioner_id),))]
            dispensed = _items_for_invoices(cx, w)
            dropshipped = _items_for_invoices(cx, d)
    except Exception:
        return []
    return rank_dispense_rows(dispensed, dropshipped, {}, catalog=catalog)


# ── Section 2: curated practice-type recommendations ───────────────────────

def recommended_ffs(practice_type, *, exclude_slugs=(), recs_path=None, catalog=None):
    """Resolve practice_type (case-insensitive) to its curated FF list, else
    'default'; drop exclude_slugs; return [{slug, name, url, blurb}]."""
    try:
        path = recs_path or str(_DATA / "practice_recommendations.json")
        recs = json.loads(Path(path).read_text())
    except Exception:
        return []
    key = next((k for k in recs if k.lower() == (practice_type or "").strip().lower()), "default")
    cat = _catalog(catalog)
    ex = {s for s in (exclude_slugs or ())}
    out = []
    for r in recs.get(key, []):
        s = r.get("slug")
        if not s or s in ex:
            continue
        out.append({"slug": s, "name": _name(s, cat), "url": _url(s), "blurb": r.get("blurb", "")})
    return out
