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


_PORTAL_SOURCES = ("portal-reorder", "reorder")


def _add_items(out, items_json):
    """Sum an order's items_json ([{slug, qty}, ...]) into out {slug: units}.
    Best-effort; one malformed line never drops the rest."""
    if not items_json:
        return
    try:
        parsed = json.loads(items_json)
    except Exception:
        return
    for it in parsed:
        try:
            s = it.get("slug")
            if s:
                out[s] = out.get(s, 0) + int(it.get("qty") or 0)
        except Exception:
            continue


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
        _add_items(out, row[0] if row else None)
    return out


def patient_portal_items(practitioner_id, *, db_path=None):
    """{slug: units} from the practitioner's clients' own client-portal reorders.
    Attribution (Approach A): a portal-reorder/reorder order counts for a
    practitioner when its email matches a client they own via dispensary_orders
    (the system's existing 'your client' link). Never raises."""
    out = {}
    try:
        with sqlite3.connect(_log_db(db_path)) as cx:
            emails = [r[0] for r in cx.execute(
                "SELECT DISTINCT lower(customer_email) FROM dispensary_orders "
                "WHERE practitioner_id=? AND customer_email IS NOT NULL AND customer_email!=''",
                (str(practitioner_id),))]
            emails = [e for e in emails if e][:500]  # defensive cap on client-list size
            if not emails:
                return {}
            eq = ",".join("?" for _ in emails)
            sq = ",".join("?" for _ in _PORTAL_SOURCES)
            rows = cx.execute(
                "SELECT items_json FROM orders WHERE lower(email) IN (%s) AND source IN (%s)" % (eq, sq),
                tuple(emails) + _PORTAL_SOURCES).fetchall()
            for (ij,) in rows:
                _add_items(out, ij)
    except Exception:
        return {}
    return out


def dispense_stats(practitioner_id, *, db_path=None, catalog=None):
    """Collect the practitioner's per-product units across channels and rank them.
    Dispensed  = their own wholesale_orders  (orders.items_json, source='wholesale').
    Drop-shipped = their dispensary_orders    (orders.items_json, source='dispensary')
                   — patient sales through the practitioner's dispensary link. Sales
                   ingested without line items (e.g. the GrooveKart webhook stub) hold
                   only aggregate bottles and contribute nothing per-product.
    Patient portal = their clients' own client-portal reorders (patient_portal_items;
                     Approach A: attributed by email via dispensary_orders).
    Sales ingested without line items (aggregate-only stubs) contribute nothing.
    Never raises."""
    dispensed, dropshipped = {}, {}
    try:
        with sqlite3.connect(_log_db(db_path)) as cx:
            try:  # each channel degrades independently (a missing table zeroes only its own)
                w = [r[0] for r in cx.execute(
                    "SELECT invoice_id FROM wholesale_orders WHERE practitioner_id=?", (str(practitioner_id),))]
                dispensed = _items_for_invoices(cx, w, "wholesale")
            except Exception:
                dispensed = {}
            try:
                d = [r[0] for r in cx.execute(
                    "SELECT invoice_id FROM dispensary_orders WHERE practitioner_id=?", (str(practitioner_id),))]
                dropshipped = _items_for_invoices(cx, d, "dispensary")
            except Exception:
                dropshipped = {}
    except Exception:
        return []
    patient_portal = patient_portal_items(practitioner_id, db_path=db_path)
    return rank_dispense_rows(dispensed, dropshipped, patient_portal, catalog=catalog)


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
    key = next((k for k in recs if not k.startswith("_") and k != "default" and k.lower() in toks), "default")
    cat = _catalog(catalog)
    ex = {s for s in (exclude_slugs or ())}
    out = []
    for r in recs.get(key, []):
        s = r.get("slug")
        if not s or s in ex:
            continue
        out.append({"slug": s, "name": _name(s, cat), "url": _url(s), "blurb": r.get("blurb", "")})
    return out
