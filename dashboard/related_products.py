"""Pure logic for the product-page related-products section. No I/O here so it is
unit-testable without importing app.py (which needs pinecone)."""

import re as _re
from urllib.parse import urlsplit as _urlsplit

DO_NOT_RECOMMEND = frozenset({
    "electrolyte-mineral-manna",
    "water-ionizer-5plate", "water-ionizer-9plate", "water-ionizer-15plate",
    "fungifuge",
})


def guardrail_ok(slug, base_slug, products):
    """Auto-list gate: real, sellable, not the product itself, not blocked."""
    if not slug or slug == base_slug:
        return False
    p = products.get(slug)
    if p is None or p.get("inactive"):
        return False
    if slug in DO_NOT_RECOMMEND:
        return False
    return True


def _entry_slug(entry):
    """A manual pick is either a bare slug string or {"slug","reason"}."""
    return entry.get("slug") if isinstance(entry, dict) else entry


def _entry_reason(entry):
    return (entry.get("reason") or "").strip() if isinstance(entry, dict) else ""


def resolve_related(base_slug, *, manual, harvested, semantic, products, cap=12):
    """Merge the three sources into {featured, more, reasons}. Manual picks bypass
    the guardrail (Glen's explicit choice) and lead; auto = harvested then semantic,
    guardrail-filtered, deduped, capped at `cap`. `reasons` maps slug -> Glen's
    optional explanation, only for manual picks that carry one."""
    seen = set()
    featured_manual = []
    reasons = {}
    for entry in manual:
        s = _entry_slug(entry)
        if s and s != base_slug and s not in seen and s in products:
            seen.add(s)
            featured_manual.append(s)
            r = _entry_reason(entry)
            if r:
                reasons[s] = r

    auto = []
    for s in list(harvested) + list(semantic):
        if s in seen or not guardrail_ok(s, base_slug, products):
            continue
        seen.add(s)
        auto.append(s)
        if len(auto) >= cap:
            break

    if not featured_manual and not auto:
        return {"featured": [], "more": [], "reasons": {}}
    featured = featured_manual + auto[:1]
    more = auto[1:]
    return {"featured": featured, "more": more, "reasons": reasons}


_URL_TAIL = _re.compile(r"/(?:remedies/[^/]+|resources)/\d+-([a-z0-9-]+)/?$", _re.I)


def map_storefront_slug(url, catalog_slugs, aliases):
    """remedymatch storefront URL -> catalog slug, or None if unresolvable.
    Only maps URLs on the remedymatch.com host; query strings/fragments are ignored."""
    if not url:
        return None
    parts = _urlsplit(url.strip())
    if "remedymatch.com" not in (parts.netloc or "").lower():
        return None
    m = _URL_TAIL.search(parts.path or "")
    if not m:
        return None
    sf = m.group(1).lower()
    if sf in aliases:
        return aliases[sf]
    if sf in catalog_slugs:
        return sf
    return None
