"""Conservative external-product -> our-equivalent mapping for the "My Remedies"
client-portal tile. The external-stack section lists, per product a client takes
from other companies, an optional pointer to our equivalent formulation — but
only where one genuinely helps clinically. This is NOT a blanket upsell: a
well-chosen external product (including one that already equals our own slug)
is left alone and `suggest_upgrade` returns None.

Matching is via a curated `_UPGRADE_MAP` keyed on a normalized ingredient/category
derived from the product NAME (re-using the `_key`-style normalization pattern
from `dashboard.supplement_reviews`) — NOT a fuzzy name match against arbitrary
brand products. An unmapped product (or a mapped one whose slug is missing from
the resolved catalog) returns None so we never point at a dead product."""
import re

# Curated ingredient/category -> our-equivalent slug + one-line clinical reason.
# Glen extends this map as he confirms additional equivalences; keys are
# normalized ingredient/category strings derived from the product name (see
# `_normalize`), not full product names or brands.
_UPGRADE_MAP = {
    "magnesium glycinate": {
        "slug": "focus-neuro-magnesium-powder",
        "reason": "Focus Neuro-Magnesium pairs glycinate with taurate/threonate forms "
                   "that cross the blood-brain barrier for calm, sleep, and cognition.",
    },
    "fish oil": {
        "slug": "wholomega",
        "reason": "WholOmega is a whole-fish (not fractionated) omega-3, preserving "
                   "the natural fatty-acid and phospholipid ratios lost in typical fish oils.",
    },
    "vitamin d": {
        "slug": "vitamin-d-syntropy",
        "reason": "Vitamin D Synergy pairs D3 with the cofactors (K2, etc.) that route "
                   "calcium correctly, rather than D3 alone.",
    },
    "turmeric": {
        "slug": "curcu-guard",
        "reason": "Curcu Guard is a standardized, absorption-formulated curcumin complex, "
                   "unlike poorly-bioavailable raw turmeric powder.",
    },
    "coq10": {
        "slug": "mitochondrial-biogenesis",
        "reason": "Mitochondrial Biogenesis pairs CoQ10 with cofactors for real "
                   "cellular-energy support, not CoQ10 alone.",
    },
    "zinc": {
        "slug": "zinc-syntropy",
        "reason": "Zinc Synergy balances zinc with copper and other cofactors to avoid "
                   "the copper depletion single-ingredient zinc can cause.",
    },
    "b12": {
        "slug": "sublingual-b12",
        "reason": "Sublingual B12 bypasses digestive absorption issues common with "
                   "oral B12 capsules or tablets.",
    },
    "probiotics": {
        "slug": "microbiome",
        "reason": "Microbiome is a targeted multi-strain formula rather than a "
                   "generic single-strain probiotic.",
    },
    "vitamin c": {
        "slug": "vitamin-c-syntropy",
        "reason": "Synergy C delivers vitamin C with its synergist cofactors for "
                   "better utilization than ascorbic acid alone.",
    },
}

# Our own brand names — a client already on one of these is already on our
# product, so no self-referential "upgrade" should ever be suggested.
_OUR_BRANDS = {"remedy match", "healing oasis"}


def _normalize(name):
    """Lowercased, whitespace-collapsed category key for a product name.
    Reuses the `_key`-style normalization from `dashboard.supplement_reviews`."""
    raw = (name or "").strip().lower()
    return re.sub(r"\s+", " ", raw)


def suggest_upgrade(product_name, product_brand="", *, catalog=None):
    """Return {"slug","name","url","reason"} for our equivalent formulation when
    there is a confident, clinically-justified match for `product_name`, else None.

    `catalog` is injectable for tests; defaults to `products.load_products()`.
    Matching is a direct lookup on `_UPGRADE_MAP` by normalized product name —
    not a fuzzy match — so an unmapped product (or a well-chosen one, including
    one that already equals our own slug) returns None. If the mapped slug is
    absent from the resolved catalog, also return None rather than point at a
    dead product."""
    if _normalize(product_brand) in _OUR_BRANDS:
        return None

    key = _normalize(product_name)
    entry = _UPGRADE_MAP.get(key)
    if not entry:
        return None

    if catalog is None:
        from dashboard import products  # lazy import: keep this module app.py-free
        catalog = products.load_products()

    slug = entry["slug"]
    product = (catalog or {}).get(slug)
    if not product:
        return None

    if _normalize(product_name) == _normalize(product.get("name", "")):
        return None

    return {
        "slug": slug,
        "name": product.get("name", slug),
        "url": product.get("url", ""),
        "reason": entry["reason"],
    }
