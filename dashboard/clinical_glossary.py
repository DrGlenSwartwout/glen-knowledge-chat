"""Reader over the curated clinical-theory catalogue
(data/clinical_theory_catalog.json) — a manual snapshot of clinicaltheory.com's
dimension pages (Organs, Meridians, Miasms, Chemistry). Pure; returns empty
structures on any load failure.

Phase 2: each remedy name is resolved to one of our product-page slugs by an
EXACT (normalised) name match against the product catalog, plus a small curated
override file for real products whose glossary name differs. No fuzzy matching —
a wrong product link is worse than a plain-text name."""
import json
import os
import re

_REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_FILENAME = "clinical_theory_catalog.json"
_OVERRIDES_FILENAME = "clinical_remedy_overrides.json"


def _path(path=None):
    if path:
        return path
    d = os.environ.get("DATA_DIR")
    if d and os.path.exists(os.path.join(d, _FILENAME)):
        return os.path.join(d, _FILENAME)
    p = os.path.join(_REPO_DATA, _FILENAME)
    return p if os.path.exists(p) else None


def load(path=None):
    p = _path(path)
    if not p:
        return {"dimensions": []}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f) or {"dimensions": []}
    except Exception:
        return {"dimensions": []}


def dimensions(catalog=None):
    """Lightweight list for the hub — no entries."""
    cat = catalog if catalog is not None else load()
    out = []
    for d in cat.get("dimensions", []):
        out.append({
            "key": d.get("key", ""), "title": d.get("title", ""),
            "blurb": d.get("blurb", ""),
            "entry_count": d.get("entry_count", len(d.get("entries", []))),
        })
    return out


def get_dimension(key, catalog=None):
    """Full dimension record (with entries) or None."""
    cat = catalog if catalog is not None else load()
    for d in cat.get("dimensions", []):
        if d.get("key") == key:
            return d
    return None


# --- Phase 2: remedy name -> product slug ------------------------------------

def _norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def product_name_index(products):
    """{normalised product name: slug} from a {slug: {..., name}} map (first wins).
    Also indexes the normalised SLUG (deterministic fallback) so a remedy named
    after the slug rather than the display name still resolves — e.g. product
    "Sleep Synergy" (slug sleep-syntropy) matches a remedy called "Sleep Syntropy".
    Display names take precedence over slug keys."""
    idx = {}
    for slug, p in (products or {}).items():
        nm = _norm_name(p.get("name") if isinstance(p, dict) else "")
        if nm:
            idx.setdefault(nm, slug)
    for slug in (products or {}):
        idx.setdefault(_norm_name(slug), slug)
    return idx


def load_overrides(path=None):
    """Curated {remedy name: product slug} for real products whose glossary name
    doesn't exactly match the catalog. Empty on any failure."""
    p = path
    if not p:
        d = os.environ.get("DATA_DIR")
        cand = [os.path.join(d, _OVERRIDES_FILENAME)] if d else []
        cand.append(os.path.join(_REPO_DATA, _OVERRIDES_FILENAME))
        p = next((c for c in cand if os.path.exists(c)), None)
    if not p:
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f) or {}
        # keys starting with "_" are notes/metadata, not mappings
        return {k: v for k, v in raw.items() if not str(k).startswith("_")}
    except Exception:
        return {}


def remedy_product_slug(name, name_index, overrides=None):
    """Product slug for a remedy name: exact normalised match, then curated
    override (by exact or normalised name); else None. No fuzzy matching."""
    nm = _norm_name(name)
    if not nm:
        return None
    if name_index and nm in name_index:
        return name_index[nm]
    ov = overrides or {}
    return ov.get((name or "").strip()) or ov.get(nm) or None


def with_product_links(dimension, name_index, overrides=None):
    """Return a shallow copy of a dimension where every remedy gains a
    `product_slug` (str or None). Safe on None/empty input."""
    if not dimension:
        return dimension
    d = dict(dimension)
    out_entries = []
    for e in d.get("entries", []):
        e = dict(e)
        rem = []
        for r in e.get("remedies", []):
            r = dict(r)
            r["product_slug"] = remedy_product_slug(r.get("name", ""), name_index, overrides)
            rem.append(r)
        e["remedies"] = rem
        out_entries.append(e)
    d["entries"] = out_entries
    return d
